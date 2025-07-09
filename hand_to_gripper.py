import os
import json
import copy
import time
import open3d as o3d
import numpy as np
from PIL import Image
import argparse
import cv2
# python hand_to_gripper.py --rgb_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/rgb" --depth_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/depth" --json_file "corrected_empty-vase_keypoint_all_take2.json"

parser = argparse.ArgumentParser(description="Gripper pose visualization with Open3D.")
parser.add_argument('--rgb_folder', type=str, required=True, help='Path to RGB image folder')
parser.add_argument('--depth_folder', type=str, required=True, help='Path to depth image folder')
parser.add_argument('--json_file', type=str, required=True, help='Path to corrected JSON file')
parser.add_argument('--output_folder', type=str, default='hand2gripper_output', help='Folder to save output frames')
args = parser.parse_args()

rgb_folder = args.rgb_folder
depth_folder = args.depth_folder
json_file = args.json_file
output_folder = args.output_folder

os.makedirs(output_folder, exist_ok=True)

video_filename = os.path.join('hand2gripper_video.mp4')  # 可改为 .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
video_writer = None

def apply_transformation_to_gripper(mesh, transformation):
    mesh_transformed = copy.deepcopy(mesh)
    tip_center = np.array([0, 0 ,0.2094631])
    translation_to_origin = np.eye(4)
    translation_to_origin[:3, 3] = -tip_center
    translation_to_origin[:3, :3] = np.array([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]])
    mesh_transformed.transform(translation_to_origin)
    mesh_transformed.transform(transformation)
    return mesh_transformed


def calculate_gripper_pose(P1, P2, P3):
    P_center = (P1 + P2) / 2
    width = np.linalg.norm(P1 - P2)

    # calculate rotation
    grasp_direction = P_center - P3
    grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
    Z_axis = grasp_direction

    X_axis = P2 - P1
    X_axis = X_axis / np.linalg.norm(X_axis)

    Y_axis = np.cross(Z_axis, X_axis)
    Y_axis = Y_axis / np.linalg.norm(Y_axis)

    X_axis = np.cross(Y_axis, Z_axis)
    X_axis = X_axis / np.linalg.norm(X_axis)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = X_axis
    rotation_matrix[:3, 1] = Y_axis
    rotation_matrix[:3, 2] = Z_axis

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = P_center

    transform = np.dot(transformation_matrix, rotation_matrix)

    return transform


def project_2d_to_3d(u, v, depth_image, intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    Z = depth_image[round(v), round(u)]
    if Z == 0:
        return np.array([0, 0, 0])
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z]) * np.array([1, -1, -1])


with open(json_file, 'r') as json_file1:
    data = json.load(json_file1)

rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.npy')])

vis = o3d.visualization.Visualizer()
vis.create_window()

cam_intr = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=570.3422241210938, fy=570.3422241210938, cx=319.5, cy=239.5)

gripper_mesh = o3d.io.read_triangle_mesh("robotiq_arg2f_140.obj")
gripper_mesh.paint_uniform_color([1, 0, 0])

render_option = vis.get_render_option()
render_option.point_size = 10.0


for index_to_view in range(len(rgb_files)):
    rgb_file = rgb_files[index_to_view]
    depth_file = depth_files[index_to_view]

    print(f"Processing frame {index_to_view}: {rgb_file}")

    rgb_path = os.path.join(rgb_folder, rgb_file)
    depth_path = os.path.join(depth_folder, depth_file)

    rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
    depth_image = np.load(depth_path)
    depth_image = np.nan_to_num(depth_image, nan=0.0)

    rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb_image),
        depth=o3d.geometry.Image(depth_image),
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    pc_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_intr)
    pc_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # try:
    if True:
        keypoints_left_np = np.array(data[str(rgb_file)]["left"]["corrected_3d_keypoints"])
        keypoints_right_np = np.array(data[str(rgb_file)]["right"]["corrected_3d_keypoints"])

        p1_left = keypoints_left_np[4]
        p2_left = keypoints_left_np[8]
        p3_left = (keypoints_left_np[2] + keypoints_left_np[5]) / 2
        transformation_l = calculate_gripper_pose(p1_left, p2_left, p3_left)
        left_gripper_mesh = apply_transformation_to_gripper(copy.deepcopy(gripper_mesh), transformation_l)
        data[str(rgb_file)]["left"]["gripper_pose"] = transformation_l.tolist()

        p1_right = keypoints_right_np[4]
        p2_right = keypoints_right_np[8]
        p3_right = (keypoints_right_np[2] + keypoints_right_np[5]) / 2
        transformation_r = calculate_gripper_pose(p1_right, p2_right, p3_right)
        right_gripper_mesh = apply_transformation_to_gripper(copy.deepcopy(gripper_mesh), transformation_r)
        data[str(rgb_file)]["right"]["gripper_pose"] = transformation_r.tolist()

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        lines_left = o3d.geometry.LineSet()
        lines_left.points = o3d.utility.Vector3dVector(keypoints_left_np)
        lines_left.lines = o3d.utility.Vector2iVector(connections)
        colors_left = [[1, 0, 0] for _ in range(len(connections))]
        lines_left.colors = o3d.utility.Vector3dVector(colors_left)

        lines_right = o3d.geometry.LineSet()
        lines_right.points = o3d.utility.Vector3dVector(keypoints_right_np)
        lines_right.lines = o3d.utility.Vector2iVector(connections)
        colors_right = [[0, 0, 1] for _ in range(len(connections))]
        lines_right.colors = o3d.utility.Vector3dVector(colors_right)


        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

        vis.clear_geometries()
        vis.add_geometry(pc_o3d)
        vis.add_geometry(left_gripper_mesh)
        vis.add_geometry(right_gripper_mesh)
        vis.add_geometry(lines_left)
        vis.add_geometry(lines_right)
        vis.add_geometry(coordinate_frame)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.02)
        image_path = os.path.join("hand2gripper_output", f"frame_{index_to_view:04d}.png")
        image = vis.capture_screen_float_buffer(False)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        if video_writer is None:
            height, width, _ = image_np.shape
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        video_writer.write(image_np)

with open(json_file, 'w') as json_file:
    json.dump(data, json_file)

vis.destroy_window()

# Generate Video visualization
if video_writer is not None:
    video_writer.release()
    print(f"Video saved to: {video_filename}")

