import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_translation(points, translation):
    return points + translation


def project_points_3d_to_2d(points_3d, K):
    #points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_2d_homogeneous = points_3d.dot(K.T)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]
    return points_2d

def plot_points_on_image(image_path, points_2d):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in points_2d:
        cv2.circle(image_rgb, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def mirror_flip_3d_points(points):
    """
    mirror flip y axis
    """
    left_hand_points = points.copy()
    left_hand_points[:, 0] = -left_hand_points[:, 0]
    return left_hand_points

def calculate_gripper_pose(P1, P2, P3):
    # 计算夹爪中心点
    P_center = (P1 + P2) / 2

    # 计算夹爪开口宽度
    width = np.linalg.norm(P1 - P2)

    # 计算抓取方向向量（虎口到夹爪中心）
    grasp_direction = P_center - P3
    grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
    return width, grasp_direction

if __name__ == '__main__':
    K = np.array([
        [12500, 0, 320],
        [0, 12500, 240],
        [0, 0, 1]
    ])
    points = np.array([[[0.0954, 0.0063, 0.0062],
                        [0.0757, -0.0261, 0.0002],
                        [0.0464, -0.0347, 0.0037],
                        [0.0196, -0.0355, 0.0074],
                        [-0.0126, -0.0335, 0.0216],
                        [0.0752, -0.0725, 0.0455],
                        [0.0676, -0.1008, 0.0604],
                        [0.0458, -0.0979, 0.0623],
                        [0.0267, -0.0842, 0.0554],
                        [0.0781, -0.0626, 0.0684],
                        [0.0622, -0.0804, 0.0891],
                        [0.0393, -0.0769, 0.0877],
                        [0.0199, -0.0688, 0.0739],
                        [0.0780, -0.0376, 0.0778],
                        [0.0541, -0.0463, 0.0912],
                        [0.0305, -0.0393, 0.0880],
                        [0.0126, -0.0382, 0.0712],
                        [0.0745, -0.0164, 0.0816],
                        [0.0548, -0.0109, 0.0865],
                        [0.0385, -0.0035, 0.0806],
                        [0.0229, -0.0012, 0.0679]],

                       [[0.0958, 0.0064, 0.0062],
                        [0.0794, -0.0280, 0.0120],
                        [0.0658, -0.0548, 0.0187],
                        [0.0588, -0.0756, 0.0346],
                        [0.0590, -0.0893, 0.0668],
                        [0.1099, -0.0725, 0.0484],
                        [0.0929, -0.0975, 0.0615],
                        [0.0729, -0.0933, 0.0701],
                        [0.0579, -0.0743, 0.0762],
                        [0.1247, -0.0616, 0.0657],
                        [0.0959, -0.0670, 0.0783],
                        [0.0749, -0.0619, 0.0869],
                        [0.0522, -0.0498, 0.0890],
                        [0.1278, -0.0363, 0.0740],
                        [0.1022, -0.0450, 0.0843],
                        [0.0795, -0.0391, 0.0926],
                        [0.0549, -0.0343, 0.0979],
                        [0.1252, -0.0153, 0.0797],
                        [0.1108, -0.0282, 0.0883],
                        [0.0930, -0.0249, 0.0940],
                        [0.0771, -0.0176, 0.0917]]])

    points[0] = mirror_flip_3d_points(points[0])

    pred_cam_t_full = np.array([[-0.13067877, -0.04837944, 10.754792],
                                [0.12562177, 0.12819055, 10.403296]])

    for i, group in enumerate(points):
        translated_group = apply_translation(group, pred_cam_t_full[i])
        projected_points_2d = project_points_3d_to_2d(translated_group, K)

        print(projected_points_2d)

        image_path = 'example_data/rgb_frame000363.png'
        plot_points_on_image(image_path, projected_points_2d)
