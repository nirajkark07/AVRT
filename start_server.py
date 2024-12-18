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
from Utils import *
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import os
import time
import math as m
# Add the Yolov10/ultralytics folder to the Python path before importing anything from ultralytics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov10'))

from detection import YOLOv10Detector

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

    ############################################################
    # Load Foundation Pose Weights
    ############################################################

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    ############################################################
    # Instantiate YOLOv10 detector
    ############################################################

    yolo_detector = YOLOv10Detector(model_path="yolov10/best.pt") 

    ############################################################
    # Instantiate Score Network
    ############################################################

    ############################################################
    # Instantiate Socket
    ############################################################

    clientsocket = instantiate_socket() # Create the server and wait for client connection

    ############################################################
    # Realsense Pipeline
    ############################################################

    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)

    frame_count = 0  # Counter for the number of frames
    inital_pose = False

    # Load calibration file
    T_rs_to_hl2 = np.loadtxt("demo_data/test6_calib/rs_to_Hl2.txt")

    # Convert degrees to radians
    angle_90 = m.radians(90)
    angle_minus_90 = m.radians(-90)

    meow = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    Rx = np.array([
        [1, 0, 0], 
        [0, m.cos(angle_90), -m.sin(angle_90)], 
        [0, m.sin(angle_90), m.cos(angle_90)]
    ])

    Rz = np.array([
        [m.cos(angle_minus_90), -m.sin(angle_minus_90), 0], 
        [m.sin(angle_minus_90), m.cos(angle_minus_90), 0], 
        [0, 0, 1]
    ])

    # Matrix multiplication
    T = meow @ Rz @ Rx

    try:
        while True:
            
            # Get frame data
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is 640 x 480
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())*depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            color = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Initialize `vis` to be the original frame (color) by default
            vis = color.copy()

            # If initial pose is false
            if inital_pose == False:
                mask, detected_class = yolo_detector.get_detection_mask(frame=color, conf_threshold=0.7)
                if detected_class == True:
                    pose = est.register(K=reader.K, rgb=color, depth=depth_image, ob_mask=mask,
                                        iteration=args.est_refine_iter)
                    inital_pose = True

            if inital_pose == True:
                pose = est.track_one(rgb=color, depth=depth_image, K=reader.K, iteration=args.track_refine_iter)
                center_pose = pose @ np.linalg.inv(to_origin)
                # hl2_center_pose = np.dot(np.linalg.inv(T_rs_to_hl2), center_pose)
                # hl2_pose = np.linalg.inv(hl2_center_pose)
                # hl2_pose = pose @ T_rs_to_hl2
                check = np.linalg.inv(T_rs_to_hl2)
                hl2_pose = check @ pose
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3,
                                    transparency=0, is_input_rgb=True)
                q,t = transformation_to_quaternion_and_translation(hl2_pose, T)
                send_pose_to_client(clientsocket, q, t)

            
            # Show the frame (with or without the pose visuals)
            cv2.imshow('1', vis[..., ::-1])

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed