# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from estimater import *
from datareader_modified import *
from pose_initial import *
import argparse
import numpy as np
import cv2

if __name__ == '__main__':
    model_number = 2 # 0 (MOTHERBOARD), 1 (CUBE)
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    if model_number == 0:
        parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/MB_Box/mesh/MB_Box.obj')
        parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/MB_Box')
    if model_number == 1:
        parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/block/mesh/cleanCube.obj')
        parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/block')
    if model_number == 2:
        parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/Gearbox/mesh/gearbox.obj')
        parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/Gearbox')
    if model_number > 2:
        print("Invalid model number! Please provide a valid model number (0 or 1).")
        sys.exit(1)      
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    
    ############################################################
    # Load Mesh
    ############################################################

    mesh = trimesh.load(args.mesh_file, force='mesh')
    if model_number == 2:
        mesh.apply_scale(0.01)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    # Hololens View
    color = cv2.imread("demo_data/test5_calib/hl_view.png")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    K = np.loadtxt("demo_data/test5_calib/K_hl2.txt")
    T_obj_to_rs = np.loadtxt("demo_data/test5_calib/pose.txt")
    T_rs_to_hl2 = np.loadtxt("demo_data/test5_calib/rs_to_Hl2.txt")
    
    # RS View
    # color = cv2.imread("demo_data/test5_calib/rs_view.png")
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # K = np.loadtxt("demo_data/test5_calib/K_f1370224.txt")
    # T_obj_to_rs = np.loadtxt("demo_data/test5_calib/pose.txt")
    # T_rs_to_hl2 = np.loadtxt("demo_data/test5_calib/rs_to_Hl2.txt")
    # T_pose = T_obj_to_rs

    # Only doing this for visualization purposes only
    center_pose = T_obj_to_rs @ np.linalg.inv(to_origin) # T_obj_to_rs OUTPUT OF POSE
    T_pose = np.dot(np.linalg.inv(T_rs_to_hl2), center_pose) # T_rs_to_hl2 MATLAB CALIBRATION
    center_pose_original = T_obj_to_rs @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(K, img=color, ob_in_cam=T_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=T_pose, scale=0.1, K=K, thickness=3,
                        transparency=0, is_input_rgb=True)
    cv2.imshow('1', vis[..., ::-1])
    cv2.waitKey(0)

    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose_original, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose_original, scale=0.1, K=K, thickness=3,
                        transparency=0, is_input_rgb=True)
    cv2.imshow('1', vis[..., ::-1])
    cv2.waitKey(0)

    meow = np.array([-1, 0, 0], [0, 1, 0], [0, 0, 1])
    A = np.array([1, 0, 0], [0, m.cos(90), -m.sin(90)])
 