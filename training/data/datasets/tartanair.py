# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import glob
import re

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


class TartanAirDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        TartanAir_DIR: str = "datasets/tartanair/",
        min_num_images: int = 24,
        len_train: int = 100,
        len_test: int = 10,
        expand_ratio: int = 8,
    ):
        """
        Initialize the TartanAirDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            TartanAir_DIR (str): Directory path to TartanAir data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.TartanAirDIR = TartanAir_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"TartanAir_DIR is {self.TartanAirDIR}")

        # Load or generate sequence list
        txt_path = osp.join(self.TartanAirDIR, "sequence_list.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # Generate sequence list and save to txt            
            sequence_list = glob.glob(osp.join(self.TartanAirDIR, "*/*/*/image_lcam*"))  # root/scene_name/difficulty/trajectory/camera          
            sequence_list = [file_path.split(self.TartanAirDIR)[-1].lstrip('/') for file_path in sequence_list] # scene_name/difficulty/trajectory/camera
            sequence_list = sorted(sequence_list)

            # Save to txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(sequence_list))

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 250.0

        '''
        # prepare camera endpoint.txt
        for seq_name in self.sequence_list:
            endpoint_data_path = osp.join(self.TartanAirDIR, seq_name).replace("image", "endpoint") + ".txt" # endpoint_lcam_front.txt
            if not osp.exists(endpoint_data_path):
                pose_path = osp.join(self.TartanAirDIR, seq_name).replace("image", "pose") + ".txt"
                pose_list = np.loadtxt(pose_path, dtype=np.float32).reshape(-1, 7) # (N,7): x y z qx qy qz qw
                depth_folder = osp.join(self.TartanAirDIR, seq_name).replace("image", "depth")
                depth_file_list = glob.glob(osp.join(depth_folder, "*_depth.png"))
                def key(p):
                    m = re.match(r"(\d+)_", osp.basename(p))
                    return int(m.group(1)) if m else osp.basename(p)
                depth_file_list = sorted(depth_file_list, key=key) # [N] paths to *_depth.png
                depth_list = []
                for depth_file in depth_file_list:
                    depth_list.append(self.read_decode_depth(depth_file)) # list of (H,W) depth maps
                endpoint_txt = self.compute_camera_endpoint(pose_list, depth_list, self.depth_max, K=self.build_tartanair_K())
                np.savetxt(endpoint_data_path, endpoint_txt, fmt="%.6f")
        '''
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: TartanAir Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: TartanAir Data dataset length: {len(self)}")

    def build_tartanair_K(self, width=640, height=640, focal=320.0) -> np.ndarray:
        fx = focal
        fy = focal
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float32)

    def build_tartanair_intrinsics(self, num_images, width=640, height=640, focal=320.0):
        # original tartan air has fixed camera intrinsics
        fx = focal
        fy = focal
        cx = width / 2.0
        cy = height / 2.0
        one_intrinsic = np.array([fx, fy, cx, cy], dtype=np.float32)
        camera_intrinsic = np.tile(one_intrinsic, (num_images, 1))

        return camera_intrinsic

    def read_decode_depth(self, depthpath: str) -> np.ndarray:
        depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)  # HxWx4 uint8
        depth = depth_rgba.view("<f4")                            # reinterpret as float32
        return np.squeeze(depth, axis=-1)                         # HxW float32 (meters)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_index is None or (seq_index>(self.sequence_list_len - 1)):
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index] # scene_name/difficulty/trajectory/camera

        scene, difficulty, traj, camera_folder = seq_name.split("/")
        #AbandonedCable, Data_easy, P003, image_lcam_bottom
        
        # Load camera parameters
        try:
            # extrinsics
            pose_file = camera_folder.replace("image", "pose") + ".txt"
            pose_path = osp.join(self.TartanAirDIR, scene, difficulty, traj, pose_file)
            camera_parameters = np.loadtxt(pose_path)
            # intrinsics
            num_images = len(camera_parameters)
            camera_intrinsic = self.build_tartanair_intrinsics(num_images=num_images)
            # endpoint
            #endpoint_file = camera_folder.replace("image", "endpoint") + ".txt"
            #endpoint_path = osp.join(self.TartanAirDIR, scene, difficulty, traj, endpoint_file)
            #camera_endpoints = np.loadtxt(endpoint_path)

        except Exception as e:
            logging.error(f"Error loading camera parameters for {seq_name}: {e}")
            raise      

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in ids:
            cam_name = camera_folder.replace("image_", "") # e.g., "lcam_left"
            image_filepath = osp.join(self.TartanAirDIR, seq_name, f"{image_idx:06d}_{cam_name}.png")
            depth_folder = camera_folder.replace("image", "depth")
            depth_filepath = osp.join(self.TartanAirDIR, scene, difficulty, traj, depth_folder, f"{image_idx:06d}_{cam_name}_depth.png")

            image = read_image_cv2(image_filepath)
            depth_rgba = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED) # raw RGBA
            depth_map = depth_rgba.view("<f4") # Convert the underlying memory (H, W, 4 * uint8) → (H, W, 1 * float32)
            depth_map = np.squeeze(depth_map, axis=-1) # Remove the redundant last axis → (H, W)
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            # Process camera matrices
            t_world = camera_parameters[image_idx, :3]
            x, y, z, w = camera_parameters[image_idx, 3:7]
            R_cam_to_world = quat_to_rot_np(x, y, z, w)
            R_world_to_cam = R_cam_to_world.T
            t_world_to_cam = -R_world_to_cam @ t_world
            P = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype=R_world_to_cam.dtype)
            R_world_to_cam_opencv = P @ R_world_to_cam
            t_world_to_cam_opencv = P @ t_world_to_cam
            extri_opencv = np.hstack([R_world_to_cam_opencv, t_world_to_cam_opencv.reshape(3,1)])

            intri_opencv = np.eye(3)
            intri_opencv[0, 0] = camera_intrinsic[image_idx][-4]
            intri_opencv[1, 1] = camera_intrinsic[image_idx][-3]
            intri_opencv[0, 2] = camera_intrinsic[image_idx][-2]
            intri_opencv[1, 2] = camera_intrinsic[image_idx][-1]

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_filepath,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue


            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "tartanair"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch