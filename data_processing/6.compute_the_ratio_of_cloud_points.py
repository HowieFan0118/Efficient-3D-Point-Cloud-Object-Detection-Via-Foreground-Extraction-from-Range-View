# -*- coding: utf-8 -*-
"""compute_the_ratio_of_cloud_points.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZptQ_Ei9iq-a9xjfZ_IdB9QfjPoT61C3
"""

!rm -rf waymo-od > /dev/null
!git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
!cd waymo-od && git branch -a
!cd waymo-od && git checkout remotes/origin/master
!pip3 install --upgrade pip

!pip3 install waymo-open-dataset-tf-2-11-0==1.6.1

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import pickle

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from google.colab import drive

drive.mount('/content/gdrive')
#读取序列tfrecord
FILENAME = '/content/gdrive/My Drive/ColabNotebooks/tfrecord/individual_files_training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME)
file_name = os.path.basename(FILENAME)
#规定保存路径
# folder_path = os.path.join('/content/gdrive/My Drive/ColabNotebooks/extracted_points', file_name.split('.')[0])
# os.makedirs(folder_path, exist_ok=True)
extracted_points_folder_path = '/content/gdrive/My Drive/ColabNotebooks/extracted_points/individual_files_training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels'
points_bb_path = '/content/gdrive/My Drive/ColabNotebooks/points_from_howie/filtered/segment-1005081002024129653_5313_150_5333_150_with_camera_labels'

import os
import pickle
total
frame_count = 1
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    (range_images, camera_projections, seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    (point,cp_point) = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)#获取激光点云
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
    points_all = np.concatenate(points, axis=0)

    # 找到frame后，找extractpoints
    file_name = f'points_{frame_count}.pkl'
    file_path = os.path.join(extracted_points_folder_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            array = pickle.load(f)
            first_three_values = np.array([point[:3] for point in array])
            print(first_three_values.shape[0])
            points_inside = []
            # Iterate through all points in points_all
            for point in first_three_values:
                # Check if the point is inside any of the bounding boxes in laser_labels
                for laser_labels in frame.laser_labels:
                    # Check if the point is inside the bounding box
                    if (point[0] >= laser_labels.box.center_x - 0.5 * laser_labels.box.length and
                        point[0] <= laser_labels.box.center_x + 0.5 * laser_labels.box.length and
                        point[1] >= laser_labels.box.center_y - 0.5 * laser_labels.box.width and
                        point[1] <= laser_labels.box.center_y + 0.5 * laser_labels.box.width and
                        point[2] >= laser_labels.box.center_z - 0.5 * laser_labels.box.height and
                        point[2] <= laser_labels.box.center_z + 0.5 * laser_labels.box.height):
                        points_inside.append(point)
            print(np.array(points_inside).shape[0])
    points_bb_exist_path = os.path.join(points_bb_path, file_name)
    if os.path.exists(points_bb_exist_path):
        with open(points_bb_exist_path, 'rb') as f:
            array_bb = pickle.load(f)
            print(array_bb.shape[0])
            #compute
            print("The ratio of points in the trained point cloud to the original point cloud within bounding boxes is"np.array(points_inside).shape[0]/array_bb.shape[0])
    break
    frame_count += 1