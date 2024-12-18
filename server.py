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
import pyrealsense2 as rs
import numpy as np
import cv2

## Helper Functions ###
def transformation_to_quaternion_and_translation(matrix):
    # Ensure it's a 4x4 matrix
    assert matrix.shape == (4, 4), "Input must be a 4x4 transformation matrix."

    # Extract the 3x3 rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Extract the translation vector
    translation_vector = matrix[:3, 3]*100
    translation_vector = np.array([translation_vector[2], translation_vector[0], -translation_vector[1]])

    # Convert the rotation matrix to a quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Returns (x, y, z, w)
    # quaternion = np.array([quaternion[2], quaternion[0], -quaternion[1], quaternion[3]])


    return quaternion, translation_vector

if __name__ == '__main__':
    model_number = 0 # 0 (MOTHERBOARD), 1 (CUBE)
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    if model_number == 0:
        parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/MB_Box/mesh/MB_Box.obj')
        parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/MB_Box')
    if model_number == 1:
        parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/block/mesh/cleanCube.obj')
        parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/block')
    if model_number > 1:
        print("Invalid model number! Please provide a valid model number (0 or 1).")
        sys.exit(1)      
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file, force='mesh')

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    ## GET MASK
    create_mask()

    ## REALSENSE FRAME CAPTURE

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
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

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    frame_count = 0  # Counter for the number of frames
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is 640 x 480
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())*depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            color = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            if frame_count == 24:  # Wait for the 25th frame
                mask = reader.get_mask(0).astype(bool)
                pose = est.register(K=reader.K, rgb=color, depth=depth_image, ob_mask=mask,
                                    iteration=args.est_refine_iter)
                frame_count += 1
            elif frame_count >= 25:  # After the 25th frame
                pose = est.track_one(rgb=color, depth=depth_image, K=reader.K, iteration=args.track_refine_iter)
                center_pose = pose @ np.linalg.inv(to_origin)
                # Multiply Pose with Calibration Data
                # T_pose = np.dot(np.linalg.inv(T_rs_to_hl2), center_pose)
                # mesh_pose = np.linalg.inv(T_pose)
                # Send Calibration Data
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3,
                                    transparency=0, is_input_rgb=True)
                cv2.imshow('1', vis[..., ::-1])
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
                    break
            else:
                frame_count += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed