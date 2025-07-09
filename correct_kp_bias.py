import numpy as np
import open3d as o3d
from PIL import Image
import os
import json
import cv2
import re
import argparse

def project_2d_to_3d(u, v, depth_image, intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    Z = depth_image[min(479, round(v)), min(639, round(u))]
    if Z == 0:
        return np.array([0, 0, 0])
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z])
# python correct_kp_bias.py --rgb_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/rgb" --depth_folder "C:/Users/Jiayun/Desktop/data/empty-vase_take2/depth" --json_file "empty-vase_keypoint_all_take2.json"
parser = argparse.ArgumentParser(description='3D keypoint correction and visualization')
parser.add_argument('--rgb_folder', type=str, required=True, help='Path to RGB image folder')
parser.add_argument('--depth_folder', type=str, required=True, help='Path to depth .npy files folder')
parser.add_argument('--json_file', type=str, required=True, help='Path to the keypoint JSON file')
parser.add_argument('--right_distance_threshold', type=str, default=5, help='Used to determine how far away from the previous hand key point is considered an outlier')
parser.add_argument('--left_distance_threshold', type=str, default=5, help='Used to determine how far away from the previous hand key point is considered an outlier')


args = parser.parse_args()

rgb_folder = args.rgb_folder
depth_folder = args.depth_folder
json_file = args.json_file
right_distance_threshold = args.right_distance_threshold
left_distance_threshold = args.left_distance_threshold

with open(json_file, 'r') as json_file1:
    data = json.load(json_file1)

rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.npy')])


vis = o3d.visualization.Visualizer()
vis.create_window()


cam_intr = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=570.3422241210938, fy=570.3422241210938, cx=319.5, cy=239.5)

K = np.array([[570.3422241210938, 0, 319.5],
                [0, 570.3422241210938, 239.5],
                [0, 0, 1]])

video_writer = None
video_filename = 'hand_keypoints.mp4'
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

count = 0
last_left = []
last_right = []
corrected_left = []
corrected_right = []

for rgb_file, depth_file in zip(rgb_files, depth_files):
    rgb_path = os.path.join(rgb_folder, rgb_file)
    depth_path = os.path.join(depth_folder, depth_file)
    keypoints_left = []

    keypoints_right = []
    rgb_image = np.array(Image.open(rgb_path).convert('RGB'))
    depth_image = np.load(depth_path)

    depth_image = np.nan_to_num(depth_image, nan=0.0)

    count += 1
    try:
        keypoints_left = data[str(rgb_file)]["left"]["3d_keypoints"]
        translation_left = np.array(data[str(rgb_file)]["left"]["translation"])
        keypoints_left_np = np.array(keypoints_left) + translation_left
        keypoints_left_np = keypoints_left_np

        # use the depth img to correct the 3d keypoint
        left_2d = data[str(rgb_file)]["left"]["2d_keypoints"]
        real_left_3d_points = []

        for u, v in left_2d:
            left_3d_point = project_2d_to_3d(u, v, depth_image, K)
            real_left_3d_points.append(left_3d_point)
        real_left_3d_points = np.array(real_left_3d_points)
        bias_1 = real_left_3d_points[1] - keypoints_left_np[1]
        bias_2 = real_left_3d_points[3] - keypoints_left_np[3]

        # keypoints_left_np += np.minimum(bias_1, bias_2)
        keypoints_left_np += bias_2
        keypoints_left_np *= np.array([1, -1, -1])

        # Check if the corrected keypoint is an outlier, if is, keep using the last one
        if count>1:
            distances = np.linalg.norm(keypoints_left_np - last_left, axis=1)
            if np.sum(distances) > left_distance_threshold:
                keypoints_left_np = last_left
        last_left = keypoints_left_np

        data[str(rgb_file)]["left"]["corrected_3d_keypoints"] = keypoints_left_np.tolist()

        distances_left = np.linalg.norm(keypoints_left_np[4] - keypoints_left_np[8])
        print(distances_left)
        if distances_left > 0.08:
            data[str(rgb_file)]["left"]["gripper"] = 1
            print("open")
        else:
            data[str(rgb_file)]["left"]["gripper"] = 0
            print("close")

        keypoints_right = data[str(rgb_file)]["right"]["3d_keypoints"]
        translation_right = np.array(data[str(rgb_file)]["right"]["translation"])
        keypoints_right_np = np.array(keypoints_right) + translation_right

        # use the depth img to correct the 3d keypoint
        right_2d = data[str(rgb_file)]["right"]["2d_keypoints"]
        real_right_3d_points = []
        for u, v in right_2d:
            right_3d_point = project_2d_to_3d(u, v, depth_image, K)
            real_right_3d_points.append(right_3d_point)
        real_right_3d_points = np.array(real_right_3d_points)
        # bias_1 = real_right_3d_points[1] - keypoints_right_np[1]
        bias_2 = real_right_3d_points[2] - keypoints_right_np[2]

        # keypoints_right_np += np.minimum(bias_1, bias_2)
        keypoints_right_np += bias_2
        keypoints_right_np *= np.array([1, -1, -1])

        # Check if the corrected keypoint is an outlier, if is, keep using the last one
        if count > 1:
            distances = np.linalg.norm(keypoints_right_np - last_right, axis=1)
            if np.sum(distances) > right_distance_threshold:
                keypoints_right_np = last_right
        last_right = keypoints_right_np

        data[str(rgb_file)]["right"]["corrected_3d_keypoints"] = keypoints_right_np.tolist()


        distances_right = np.linalg.norm(keypoints_right_np[4] - keypoints_right_np[8])
        # print(distances_right)

        if distances_right > 0.09:
            data[str(rgb_file)]["right"]["gripper"] = 1
            print("open")
        else:
            data[str(rgb_file)]["right"]["gripper"] = 0
            print("close")

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

        sphere_radius = 0.005
        spheres_left = []
        for kp in keypoints_left_np:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(kp)
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            spheres_left.append(sphere)

        spheres_right = []
        for kp in keypoints_right_np:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(kp)
            sphere.paint_uniform_color([0, 0, 1])
            spheres_right.append(sphere)


    except KeyError:
        continue

    rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb_image),
        depth=o3d.geometry.Image(depth_image),
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    pc_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_intr)

    pc_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    vis.clear_geometries()
    vis.add_geometry(pc_o3d)
    vis.add_geometry(lines_left)
    vis.add_geometry(lines_right)

    for sphere in spheres_left:
        vis.add_geometry(sphere)
    for sphere in spheres_right:
        vis.add_geometry(sphere)

    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer(False)
    image_np = (np.asarray(image) * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if video_writer is None:
        height, width, _ = image_np.shape
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    video_writer.write(image_np)

video_writer.release()
vis.destroy_window()



image_files = [f for f in os.listdir(rgb_folder) if f.startswith("rgb_frame") and f.endswith(".png")]
image_files_sorted = sorted(image_files, key=lambda x: int(re.search(r"rgb_frame(\d+)\.png", x).group(1)))


last_valid_data = None

for frame in image_files_sorted:
    if frame in data:  # If key in json
        if "corrected_3d_keypoints" not in data[frame]["left"]:
            data[frame]["left"] = last_valid_data["left"]
            # print(frame)
        if "corrected_3d_keypoints" not in data[frame]["right"]:
            data[frame]["right"] = last_valid_data["right"]
            print(frame)
        last_valid_data = data[frame]
    else:
        if last_valid_data is not None:
            data[frame] = last_valid_data.copy()
            print(f"Added missing frame {frame} using previous data.")
            # (data[frame])
        else:
            print(f"Warning: {frame} is missing and no previous data available!")

output_file = json_file
with open("corrected_" + json_file, 'w') as json_file:
    json.dump(data, json_file)