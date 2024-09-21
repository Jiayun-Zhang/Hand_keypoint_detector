from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
from back_project import mirror_flip_3d_points, plot_points_on_image, apply_translation, project_points_3d_to_2d, calculate_gripper_pose
from vitpose_model import ViTPoseModel

import json


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default= 'C:/Users/Jiayun/Desktop/hamer-main/demos_new/take1/rgb', help='Folder with input images')
    parser.add_argument('--out_file', type=str, default='keypoint_all_take1.json', help='Output json file name')

    parser.add_argument('--out_folder', type=str, default='C:/Users/Jiayun/Desktop/hamer-main/demos_new/take1/rgb',
                        help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False,
                        help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True,
                        help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False,
                        help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'],
                        help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'],
                        help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    outer_dict = {}

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # 初始化用于存储最高置信度的左手和右手
        best_left_confidence = -np.inf
        best_right_confidence = -np.inf
        best_left_hand_bbox = None
        best_right_hand_bbox = None

        for vitposes in vitposes_out:

            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]


            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.4
            if sum(valid) > 3:
                total_confidence = np.sum(keyp[valid, 2])  # 计算左手关键点的置信度总和
                if total_confidence > best_left_confidence:  # 更新最高置信度的左手
                    best_left_confidence = total_confidence
                    best_left_hand_bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                                           keyp[valid, 0].max(), keyp[valid, 1].max()]

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.4
            if sum(valid) > 3:
                total_confidence = np.sum(keyp[valid, 2])
                if total_confidence > best_right_confidence:
                    best_right_confidence = total_confidence
                    best_right_hand_bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                                            keyp[valid, 0].max(), keyp[valid, 1].max()]

        # 保存最后检测到的最高置信度的左手和右手
        bboxes = []
        is_right = []

        if best_left_hand_bbox is not None:
            bboxes.append(best_left_hand_bbox)
            is_right.append(0)

        if best_right_hand_bbox is not None:
            bboxes.append(best_right_hand_bbox)
            is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2 * batch['right'] - 1)
            # scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            #
            # print(model_cfg.EXTRA.FOCAL_LENGTH)
            # print(model_cfg.MODEL.IMAGE_SIZE)
            # print(img_size.max())
            # scaled_focal_length = 525.5
            scaled_focal_length = torch.tensor(525., device='cuda:0')
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()

            scaled_focal_length = scaled_focal_length.cpu().item()
            print(scaled_focal_length)
            K = np.array([
                [scaled_focal_length, 0, 320],
                [0, scaled_focal_length, 240],
                [0, 0, 1]])

            left = {}
            right = {}

            for i in range(len(batch['right'])):
                if batch['right'][i] == 1:
                    right_hand_keyp = out['pred_keypoints_3d'][i].cpu().numpy()
                    right["3d_keypoints"] = right_hand_keyp.tolist()
                    t_right = pred_cam_t_full[i]
                    right["translation"] = t_right.tolist()
                    right_hand_keyp = apply_translation(right_hand_keyp, t_right)
                    right_hand_points_2d = project_points_3d_to_2d(right_hand_keyp, K)
                    right["2d_keypoints"] = right_hand_points_2d.tolist()
                    right["focal_length"] = scaled_focal_length

                    # thumb_tip = np.array(right["3d_keypoints"][4])
                    # index_tip = np.array(right["3d_keypoints"][8])
                    # index_mcp = np.array(right["3d_keypoints"][5])
                    # thumb_mcp = np.array(right["3d_keypoints"][1])
                    # gripper_base = index_mcp + thumb_mcp / 2
                    # width, grasp_direction = calculate_gripper_pose(thumb_tip, index_tip, gripper_base)
                    # right["thumb_tip"] = thumb_tip.tolist()
                    # right["index_tip"] = index_tip.tolist()
                    # right["gripper_base"] = gripper_base.tolist()
                    # right["gripper_open_width"] = width
                    # right["grasp_direction"] = grasp_direction.tolist()

                if batch['right'][i] == 0:
                    left_hand_keyp = mirror_flip_3d_points(out['pred_keypoints_3d'].cpu().numpy()[i])
                    left["3d_keypoints"] = left_hand_keyp.tolist()
                    t_left = pred_cam_t_full[i]
                    left["translation"] = t_left.tolist()
                    left_hand_keyp = apply_translation(left_hand_keyp, t_left)
                    left_hand_points_2d = project_points_3d_to_2d(left_hand_keyp, K)
                    left["2d_keypoints"] = left_hand_points_2d.tolist()
                    left["focal_length"] = scaled_focal_length

                    # thumb_tip = np.array(left["3d_keypoints"][4])
                    # index_tip = np.array(left["3d_keypoints"][8])
                    # index_mcp = np.array(left["3d_keypoints"][5])
                    # thumb_mcp = np.array(left["3d_keypoints"][1])
                    # gripper_base = index_mcp + thumb_mcp / 2
                    # width, grasp_direction = calculate_gripper_pose(thumb_tip, index_tip, gripper_base)
                    # left["thumb_tip"] = thumb_tip.tolist()
                    # left["index_tip"] = index_tip.tolist()
                    # left["gripper_base"] = gripper_base.tolist()
                    # left["gripper_open_width"] = width
                    # left["grasp_direction"] = grasp_direction.tolist()

            filename = str(img_path.name)
            inner_dict = {}
            inner_dict["left"] = left
            inner_dict["right"] = right
            outer_dict[filename] = inner_dict
    with open(args.out_file, 'w') as json_file:
        json.dump(outer_dict, json_file)


if __name__ == '__main__':
    main()