#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import glob
import sys
import torch
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from gaussian_splatting.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from gaussian_splatting.utils.sh_utils import SH2RGB
from gaussian_splatting.scene.gaussian_model import BasicPointCloud
import cv2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

    normal: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    center: list  # 新增center属性
    scale: float  # 新增scale属性

def get_center(pts):
    """辅助函数：计算点云的中心（去除异常值）"""
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & \
            (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_info(cam_extrinsics, pcd):
    """辅助函数：计算场景归一化的变换矩阵、缩放因子和中心"""
    poses = []
    pts = torch.from_numpy(pcd.points)  # BasicPointCloud的points属性
    
    # 读取所有相机的位姿
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        W2C = getWorld2View2(R, T)
        C2W = np.linalg.inv(W2C)
        poses.append(torch.from_numpy(C2W).float())
    poses = torch.stack(poses, dim=0)   

    # 筛选前景点云（在相机包围盒内）
    poses_min, poses_max = poses[...,3].min(0)[0], poses[...,3].max(0)[0]
    pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & 
                 (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
    
    # 计算中心和缩放因子
    center = get_center(pts_fg)
    tc = center.reshape(3, 1)
    t = -tc
  
    inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), 
                           torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
    scale = (pts_fg - tc.T).norm(p=2, dim=-1).max()
    return inv_trans, scale, tc

def normalize_scene(pcd, cam_infos_unsorted, inv_trans, scale):
    """辅助函数：对场景（点云和相机）进行归一化"""
    # 归一化点云
    pts = torch.from_numpy(pcd.points).cuda()
    bb = torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None].cuda()
    pts = (inv_trans.cuda() @ bb)[:,:3,0]
    pts = pts / scale
    
    normalized_pcd = BasicPointCloud(
        points=pts.cpu().numpy(), 
        colors=pcd.colors, 
        normals=pcd.normals
    )
    
    # 归一化相机
    norm_cam_infos_unsorted = []
    for cam in cam_infos_unsorted:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        
        # 应用归一化变换
        C2W_norm = (inv_trans @ C2W)
        C2W_norm[...,3] /= scale
        C2W_norm[3,3] = 1
        W2C_norm = np.linalg.inv(C2W_norm)
        
        # 更新相机参数
        R_norm = W2C_norm[:3, :3].transpose()
        t_norm = W2C_norm[:3, 3]
        
        cam_info = CameraInfo(
            uid=cam.uid, 
            R=R_norm, 
            T=t_norm, 
            FovY=cam.FovY, 
            FovX=cam.FovX, 
            image=cam.image,
            image_path=cam.image_path, 
            image_name=cam.image_name, 
            width=cam.width, 
            height=cam.height,
            normal=cam.normal  # 保留法线信息
        )
        norm_cam_infos_unsorted.append(cam_info)
        
    return normalized_pcd, norm_cam_infos_unsorted

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        
        # 处理图像读取
        if not os.path.exists(image_path):
            image = None
        else:
            image = Image.open(image_path)

        # 处理法线图像（如果存在）
        normal_image_path = os.path.join(images_folder, "normal", image_name + "_normal.png")
        normal = None
        if os.path.exists(normal_image_path):
            normal_image = Image.open(normal_image_path)
            normal_im_data = np.array(normal_image.convert("RGBA"))
            bg = np.array([0, 0, 0])
            norm_normal_data = normal_im_data / 255.0
            normal = norm_normal_data[:,:,:3] * norm_normal_data[:, :, 3:4] + bg * (1 - norm_normal_data[:, :, 3:4])

        cam_info = CameraInfo(
            uid=uid, 
            R=R, 
            T=T, 
            FovY=FovY, 
            FovX=FovX, 
            image=image,
            image_path=image_path, 
            image_name=image_name, 
            width=width, 
            height=height,
            normal=normal
        )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, lod, llffhold=8, scale_input=1.0, center_input=[0,0,0]):
    """读取Colmap场景信息（支持scale_input和center_input参数）"""
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        images_folder=os.path.join(path, reading_dir)
    )

    # 处理点云
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    # 根据传入参数处理归一化
    if scale_input != 0.0 or center_input != [0,0,0]:
        tc = torch.tensor(center_input).reshape(3, 1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), -tc], dim=1), 
                               torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        scale = scale_input
    else:
        inv_trans, scale, tc = normalize_info(cam_extrinsics, pcd)
        scale = 1.0  # 默认不缩放

    # 归一化点云和相机
    pcd, cam_infos_unsorted = normalize_scene(pcd, cam_infos_unsorted, inv_trans, scale)

    # 排序相机
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # 划分训练/测试集
    if eval:
        if lod > 0:
            print(f'using lod, using eval')
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f'test_cam_infos: {len(test_cam_infos)}')
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # 计算NeRF归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 返回包含center和scale的SceneInfo
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        center=tc.cpu().numpy().flatten().tolist(),  # 转为list
        scale=scale.item()  # 转为float
    )
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_debug=False, undistorted=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]], dtype=np.float32)
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            normal_image_path = os.path.join(path, "normal", image_name + "_normal.png")
            normal_image = Image.open(normal_image_path) if os.path.exists(normal_image_path) else None
            normal = None
            if normal_image:
                normal_im_data = np.array(normal_image.convert("RGBA"))
                bg = np.array([0, 0, 0])
                norm_normal_data = normal_im_data / 255.0
                normal = norm_normal_data[:,:,:3] * norm_normal_data[:, :, 3:4] + bg * (1 - norm_normal_data[:, :, 3:4])

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], normal=normal))
            
            if is_debug and idx > 50:
                break
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 为Blender场景添加默认的center和scale
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        center=[0,0,0],  # 默认中心
        scale=1.0        # 默认缩放
    )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}