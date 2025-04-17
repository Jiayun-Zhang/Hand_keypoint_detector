import copy
import open3d as o3d
import numpy as np
from PIL import Image
import os
import json
import time
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

rgb_folder = 'C:/Users/Jiayun/Desktop/pouring/take8/rgb'
depth_folder = 'C:/Users/Jiayun/Desktop/pouring/take8/depth_predicted'
index_to_view = 250
with open("corrected_keypoint_all_take8.json", 'r') as json_file1:
    data = json.load(json_file1)

rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.npy')])

vis = o3d.visualization.Visualizer()
vis.create_window()

cam_intr = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=525., fy=525., cx=319.5, cy=239.5)
gripper_mesh = o3d.io.read_triangle_mesh("robotiq_arg2f_140.obj")

rgb_file = rgb_files[index_to_view]
depth_file = depth_files[index_to_view]
print(rgb_file)

rgb_path = os.path.join(rgb_folder, rgb_file)
depth_path = os.path.join(depth_folder, depth_file)

rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
depth_image = np.load(depth_path)
depth_image = np.nan_to_num(depth_image, nan=0.0)

K = np.array([
                [525., 0, 319.5],
                [0, 525., 239.5],
                [0, 0, 1]])
try:
    keypoints_left_np = np.array(data[str(rgb_file)]["left"]["corrected_3d_keypoints"])
    keypoints_right_np = np.array(data[str(rgb_file)]["right"]["corrected_3d_keypoints"])

    p1_left = keypoints_left_np[4]
    p2_left = keypoints_left_np[8]
    p3_left = (keypoints_left_np[2] + keypoints_left_np[5]) / 2
    transformation_l = calculate_gripper_pose(p1_left, p2_left, p3_left)
    p1_right = keypoints_right_np[4]
    p2_right = keypoints_right_np[8]
    p3_right = (keypoints_right_np[2] + keypoints_right_np[5]) / 2
    transformation_r = calculate_gripper_pose(p1_right, p2_right, p3_right)

    left_gripper_mesh = apply_transformation_to_gripper(copy.deepcopy(gripper_mesh), transformation_l)
    right_gripper_mesh = apply_transformation_to_gripper(copy.deepcopy(gripper_mesh), transformation_r)

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


except KeyError:
    print("KeyError occurred, skipping this file.")
    exit()



rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color=o3d.geometry.Image(rgb_image),
    depth=o3d.geometry.Image(depth_image),
    depth_scale=1.0,
    depth_trunc=3.0,
    convert_rgb_to_intensity=False
)

pc_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_intr)
pc_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


pc_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_intr)
pc_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

vis.clear_geometries()
vis.add_geometry(pc_o3d)
vis.add_geometry(left_gripper_mesh)
vis.add_geometry(right_gripper_mesh)
vis.add_geometry(lines_left)
vis.add_geometry(lines_right)
vis.add_geometry(coordinate_frame)
render_option = vis.get_render_option()
render_option.point_size = 10.0

vis.run()
vis.destroy_window()
