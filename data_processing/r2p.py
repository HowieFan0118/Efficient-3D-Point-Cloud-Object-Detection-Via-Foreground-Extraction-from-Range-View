import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import pickle
from tqdm import tqdm
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def r2p(folder_path,dataset):
	progress_bar_ = tqdm(range(199), desc="frame progress")
	frame_count = 1
	for data in dataset:
		frame = open_dataset.Frame()
		frame.ParseFromString(bytearray(data.numpy()))
		(range_images, camera_projections, seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
		(point,cp_point) = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)#获取激光点云
		points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
		points_all = np.concatenate(points, axis=0)

		points_bb = []
	# Iterate through all points in points_all
		for point in points_all:
	    # Check if the point is inside any of the bounding boxes in laser_labels
			for laser_labels in frame.laser_labels:
	        # Check if the point is inside the bounding box
	  			if (point[3] >= laser_labels.box.center_x - 0.5 * laser_labels.box.length and
	            point[3] <= laser_labels.box.center_x + 0.5 * laser_labels.box.length and
	            point[4] >= laser_labels.box.center_y - 0.5 * laser_labels.box.width and
	            point[4] <= laser_labels.box.center_y + 0.5 * laser_labels.box.width and
	            point[5] >= laser_labels.box.center_z - 0.5 * laser_labels.box.height and
	            point[5] <= laser_labels.box.center_z + 0.5 * laser_labels.box.height):
	            # If the point is inside the bounding box, add it to points_bb
							points_bb.append(point)
							break

	# Convert points_bb to a numpy array
		points_bb = np.array(points_bb)
		file_path = os.path.join(folder_path, f"points_{frame_count}.pkl")
		with open(file_path, 'wb') as file:
			pickle.dump(points_bb, file)
		#print(f"Data saved successfully to {file_path}")
		frame_count += 1
		progress_bar_.update(1)

folder_list = os.listdir('raw_data')
progress_bar = tqdm(range(len(folder_list)), desc="TFrecord progress")
for file_name in folder_list:
	FILENAME = os.path.join('raw_data', file_name)
	dataset = tf.data.TFRecordDataset(FILENAME)
	#规定保存路径
	folder_path = os.path.join('filtered', file_name.split('.')[0])
	os.makedirs(folder_path, exist_ok=True)
	r2p(folder_path,dataset)
	progress_bar.update(1)
	