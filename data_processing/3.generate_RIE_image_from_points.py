# -*- coding: utf-8 -*-
"""2.1generate_intensity_elongation_image_from_points.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ziE1aOwBtp4SwUdJmyWP8_v1-qFAUCy5
"""

!rm -rf waymo-od > /dev/null
!git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
!cd waymo-od && git branch -a
!cd waymo-od && git checkout remotes/origin/master
!pip3 install --upgrade pip
!pip3 install opencv-python
!pip3 install --upgrade numpy
!pip3 install --upgrade opencv-python

!pip3 install waymo-open-dataset-tf-2-11-0==1.6.1

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from google.colab import drive

drive.mount('/content/gdrive')
FILENAME = '/content/gdrive/MyDrive/ColabNotebooks/tfrecord/individual_files_training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME)
# 从文件路径中提取文件名
file_name = os.path.basename(FILENAME)
# 创建一个以文件名命名的文件夹
folder_path = os.path.join('/content/gdrive/MyDrive/ColabNotebooks/intensityimage', file_name.split('.')[0])
os.makedirs(folder_path, exist_ok=True)

import numpy as np
from PIL import Image
import tensorflow as tf
import os

def save_rgb_image(range_image, folder_path, save_path):
    """Generates an RGB image from a range image.

    Args:
      range_image: The range image data of type tf.Tensor with shape (1, 64, 2650, 3).
      folder_path: The path to the folder where the image will be saved.
      save_path: The filename for saving the image.
    """
    range_image_tensor = tf.convert_to_tensor(range_image)


    # Normalize the range values to [0, 1] as float32
    normalized_range_image = tf.cast(range_image_tensor, dtype=tf.float32)
    r_channel = normalized_range_image[:, :, :, 0]  # Red channel
    g_channel = normalized_range_image[:, :, :, 1]  # Green channel
    b_channel = normalized_range_image[:, :, :, 2]  # Blue channel

    lidar_image_mask = tf.cast(tf.greater_equal(r_channel, 0), dtype=tf.bool)
    r_channel = (r_channel - tf.reduce_min(r_channel)) / (tf.reduce_max(r_channel) - tf.reduce_min(r_channel))
    r_channel *= 254
    r_channel = tf.where(lidar_image_mask, r_channel, tf.constant(255, dtype=tf.float32))

    lidar_image_mask2 = tf.cast(tf.greater_equal(g_channel, 0), dtype=tf.bool)
    g_channel = (g_channel - tf.reduce_min(g_channel)) / (tf.reduce_max(g_channel) - tf.reduce_min(g_channel))
    g_channel *= 254
    g_channel = tf.where(lidar_image_mask2, g_channel, tf.constant(255, dtype=tf.float32))

    lidar_image_mask3 = tf.cast(tf.greater_equal(g_channel, 0), dtype=tf.bool)
    b_channel = (b_channel - tf.reduce_min(g_channel)) / (tf.reduce_max(b_channel) - tf.reduce_min(b_channel))
    b_channel *= 254
    b_channel = tf.where(lidar_image_mask3, b_channel, tf.constant(255, dtype=tf.float32))

    rgb_image = tf.stack([r_channel, g_channel, b_channel], axis=-1)

    #rgb_image = rgb_image * 255
    rgb_image_int = tf.cast(rgb_image, dtype=tf.int32)

    rgb_image_np = rgb_image_int.numpy()[0]

    # Create PIL image from numpy array
    pil_image = Image.fromarray(rgb_image_np.astype(np.uint8))

    # Resize image to the desired dimensions (64, 2650)
    resized_image = pil_image.resize((2650, 64), Image.ANTIALIAS)

    # Save the resized image
    resized_image.save(os.path.join(folder_path, save_path))



def build_intensity_range_image_from_point_cloud(points_vehicle_frame,
                                       num_points,
                                       extrinsic,
                                       inclination,
                                       range_image_size,
                                       point_features,
                                       dtype=tf.float32,
                                       scope=None):
  """Build virtual range image from point cloud assuming uniform azimuth.

  Args:
    points_vehicle_frame: tf tensor with shape [B, N, 3] in the vehicle frame.
    num_points: [B] int32 tensor indicating the number of points for each frame.
    extrinsic: tf tensor with shape [B, 4, 4].
    inclination: tf tensor of shape [B, H] that is the inclination angle per
      row. sorted from highest value to lowest.
    range_image_size: a size 2 [height, width] list that configures the size of
      the range image.
    point_features: If not None, it is a tf tensor with shape [B, N, 2] that
      represents lidar 'intensity' and 'elongation'.
    dtype: the data type to use.
    scope: tf name scope.

  Returns:
    range_images : [B, H, W, 3] or [B, H, W] tensor. Range images built from the
      given points. Data type is the same as that of points_vehicle_frame. 0.0
      is populated when a pixel is missing.
    ri_indices: tf int32 tensor [B, N, 2]. It represents the range image index
      for each point.
    ri_ranges: [B, N] tensor. It represents the distance between a point and
      sensor frame origin of each point.
  """

  with tf.compat.v1.name_scope(
      scope,
      'BuildRangeImageFromPointCloud',
      values=[points_vehicle_frame, extrinsic, inclination]):
    points_vehicle_frame_dtype = points_vehicle_frame.dtype

    points_vehicle_frame = tf.cast(points_vehicle_frame, dtype)
    extrinsic = tf.cast(extrinsic, dtype)
    inclination = tf.cast(inclination, dtype)
    height, width = range_image_size

    # [B, 4, 4]
    vehicle_to_laser = tf.linalg.inv(extrinsic)
    # [B, 3, 3]
    rotation = vehicle_to_laser[:, 0:3, 0:3]
    # [B, 1, 3]
    translation = tf.expand_dims(vehicle_to_laser[::, 0:3, 3], 1)
    # Points in sensor frame
    # [B, N, 3]
    points = tf.einsum('bij,bkj->bik', points_vehicle_frame,
                       rotation) + translation
    # [B, N]
    xy_norm = tf.norm(tensor=points[..., 0:2], axis=-1)
    # [B, N]
    point_inclination = tf.atan2(points[..., 2], xy_norm)
    # [B, N, H]
    point_inclination_diff = tf.abs(
        tf.expand_dims(point_inclination, axis=-1) -
        tf.expand_dims(inclination, axis=1))
    # [B, N]
    point_ri_row_indices = tf.argmin(
        input=point_inclination_diff, axis=-1, output_type=tf.int32)
    # [B, 1], within [-pi, pi]
    az_correction = tf.expand_dims(
        tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0]), -1)
    # [B, N], within [-2pi, 2pi]
    point_azimuth = tf.atan2(points[..., 1], points[..., 0]) + az_correction

    point_azimuth_gt_pi_mask = point_azimuth > np.pi
    point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
    point_azimuth = point_azimuth - tf.cast(
        point_azimuth_gt_pi_mask, dtype=dtype) * 2 * np.pi
    point_azimuth = point_azimuth + tf.cast(
        point_azimuth_lt_minus_pi_mask, dtype=dtype) * 2 * np.pi

    # [B, N].
    point_ri_col_indices = width - 1.0 + 0.5 - (point_azimuth +
                                                np.pi) / (2.0 * np.pi) * width
    point_ri_col_indices = tf.cast(
        tf.round(point_ri_col_indices), dtype=tf.int32)

    with tf.control_dependencies([
        tf.compat.v1.assert_non_negative(point_ri_col_indices),
        tf.compat.v1.assert_less(point_ri_col_indices, tf.cast(width, tf.int32))
    ]):
      # [B, N, 2]
      ri_indices = tf.stack([point_ri_row_indices, point_ri_col_indices], -1)
      # [B, N]
      ri_ranges = tf.cast(
          tf.norm(tensor=points, axis=-1), dtype=points_vehicle_frame_dtype)
      print('****************************')
      def fn(args):
        """Builds a range image for each frame.

        Args:
          args: a tuple containing:
            - ri_index: [N, 2] int tensor.
            - ri_value: [N] float tensor.
            - num_point: scalar tensor
            - point_feature: [N, 2] float tensor.

        Returns:
          range_image: [H, W]
        """
        if len(args) == 3:
          ri_index, ri_value, num_point = args
        else:
          ri_index, ri_value, num_point, point_feature = args
          ri_value = tf.concat([ri_value[..., tf.newaxis], point_feature],
                               axis=-1)
          ri_value = encode_lidar_features(ri_value)

        # pylint: disable=unbalanced-tuple-unpacking
        ri_index = ri_index[0:num_point, :]
        ri_value = ri_value[0:num_point, ...]
        range_image = scatter_nd_with_pool(ri_index, ri_value, [height, width],
                                           tf.math.unsorted_segment_min)
        if len(args) != 3:
          range_image = decode_lidar_features(range_image)
        return range_image

      elems = [ri_indices, ri_ranges, num_points]
      if point_features is not None:
        elems.append(point_features)
      range_images = tf.map_fn(
          fn, elems=elems, dtype=points_vehicle_frame_dtype, back_prop=False)

      return range_images






_RANGE_TO_METERS = 0.00585532144
def _decode_elongation(elongation):
  """Decodes lidar elongation from uint8 to float.

  Args:
    elongation: A uint8 tensor represents lidar elongation.

  Returns:
    Decoded lidar elongation.
  """
  return tf.cast(elongation, dtype=tf.float32) * _RANGE_TO_METERS
def _decode_intensity(intensity):
  """Decodes lidar intensity from uint16 to float32.

  The given intensity is encoded with _encode_intensity.

  Args:
    intensity: A uint16 tensor represents lidar intensity.

  Returns:
    Decoded intensity with type as float32.
  """
  if intensity.dtype != tf.uint16:
    raise TypeError('intensity must be of type uint16')

  intensity_uint32 = tf.cast(intensity, dtype=tf.uint32)
  intensity_uint32_shifted = tf.bitwise.left_shift(intensity_uint32, 16)
  return tf.bitcast(intensity_uint32_shifted, tf.float32)
def _decode_range(r):
  """Decodes lidar range from integers to float32.

  Args:
    r: A integer tensor.

  Returns:
    Decoded range.
  """
  return tf.cast(r, dtype=tf.float32) * _RANGE_TO_METERS
def decode_lidar_features(lidar_point_feature):
  """Decodes lidar features (range, intensity, enlongation).

  This function decodes lidar point features encoded by 'encode_lidar_features'.

  Args:
    lidar_point_feature: [N, 3] int64 tensor.

  Returns:
    [N, 3] float tensors that encodes lidar_point_feature.
  """

  r, intensity, elongation = tf.unstack(lidar_point_feature, axis=-1)

  decoded_r = _decode_range(r)
  intensity = tf.bitwise.bitwise_and(intensity, int(0xFFFF))
  decoded_intensity = _decode_intensity(tf.cast(intensity, dtype=tf.uint16))
  elongation = tf.bitwise.bitwise_and(elongation, int(0xFF))
  decoded_elongation = _decode_elongation(tf.cast(elongation, dtype=tf.uint8))

  return tf.stack([decoded_r, decoded_intensity, decoded_elongation], axis=-1)
def _combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(input=tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape
def scatter_nd_with_pool(index,
                         value,
                         shape,
                         pool_method=tf.math.unsorted_segment_max):
  """Similar as tf.scatter_nd but allows custom pool method.

  tf.scatter_nd accumulates (sums) values if there are duplicate indices.

  Args:
    index: [N, 2] tensor. Inner dims are coordinates along height (row) and then
      width (col).
    value: [N, ...] tensor. Values to be scattered.
    shape: (height,width) list that specifies the shape of the output tensor.
    pool_method: pool method when there are multiple points scattered to one
      location.

  Returns:
    image: tensor of shape with value scattered. Missing pixels are set to 0.
  """
  if len(shape) != 2:
    raise ValueError('shape must be of size 2')
  height = shape[0]
  width = shape[1]
  # idx: [N]
  index_encoded, idx = tf.unique(index[:, 0] * width + index[:, 1])
  value_pooled = pool_method(value, idx, tf.size(input=index_encoded))
  index_unique = tf.stack(
      [index_encoded // width,
       tf.math.mod(index_encoded, width)], axis=-1)
  shape = [height, width]
  value_shape = _combined_static_and_dynamic_shape(value)
  if len(value_shape) > 1:
    shape = shape + value_shape[1:]

  image = tf.scatter_nd(index_unique, value_pooled, shape)
  return image
def _encode_elongation(elongation):
  """Encodes lidar elongation from float to uint8.

  Args:
    elongation: A float tensor represents lidar elongation.

  Returns:
    Encoded lidar elongation.
  """
  encoded_elongation = elongation / _RANGE_TO_METERS
  with tf.control_dependencies([
      tf.compat.v1.assert_non_negative(encoded_elongation),
      tf.compat.v1.assert_less_equal(encoded_elongation, math.pow(2, 8) - 1.)
  ]):
    return tf.cast(encoded_elongation, dtype=tf.uint8)
def _encode_intensity(intensity):
  """Encodes lidar intensity from float to uint16.

  The integer value stored here is the upper 16 bits of a float. This
  preserves the exponent and truncates the mantissa to 7bits, which gives
  plenty of dynamic range and preserves about 3 decimal places of
  precision.

  Args:
    intensity: A float tensor represents lidar intensity.

  Returns:
    Encoded intensity with type as uint32.
  """
  if intensity.dtype != tf.float32:
    raise TypeError('intensity must be of type float32')

  intensity_uint32 = tf.bitcast(intensity, tf.uint32)
  intensity_uint32_shifted = tf.bitwise.right_shift(intensity_uint32, 16)
  return tf.cast(intensity_uint32_shifted, dtype=tf.uint16)
def encode_lidar_features(lidar_point_feature):
  """Encodes lidar features (range, intensity, enlongation).

  This function encodes lidar point features such that all features have the
  same ordering as lidar range.

  Args:
    lidar_point_feature: [N, 3] float32 tensor.

  Returns:
    [N, 3] int64 tensors that encodes lidar_point_feature.
  """
  if lidar_point_feature.dtype != tf.float32:
    raise TypeError('lidar_point_feature must be of type float32.')

  r, intensity, elongation = tf.unstack(lidar_point_feature, axis=-1)
  encoded_r = tf.cast(_encode_range(r), dtype=tf.uint32)
  encoded_intensity = tf.cast(_encode_intensity(intensity), dtype=tf.uint32)
  encoded_elongation = tf.cast(_encode_elongation(elongation), dtype=tf.uint32)

  encoded_r_shifted = tf.bitwise.left_shift(encoded_r, 16)

  encoded_intensity = tf.cast(
      tf.bitwise.bitwise_or(encoded_r_shifted, encoded_intensity),
      dtype=tf.int64)
  encoded_elongation = tf.cast(
      tf.bitwise.bitwise_or(encoded_r_shifted, encoded_elongation),
      dtype=tf.int64)
  encoded_r = tf.cast(encoded_r, dtype=tf.int64)

  return tf.stack([encoded_r, encoded_intensity, encoded_elongation], axis=-1)
def _encode_range(r):
  """Encodes lidar range from float to uint16.

  Args:
    r: A float tensor represents lidar range.

  Returns:
    Encoded range with type as uint16.
  """
  encoded_r = r / _RANGE_TO_METERS
  with tf.control_dependencies([
      tf.compat.v1.assert_non_negative(encoded_r),
      tf.compat.v1.assert_less_equal(encoded_r, math.pow(2, 16) - 1.)
  ]):
    return tf.cast(encoded_r, dtype=tf.uint16)

# frame_count = 101
# for i, data in enumerate(dataset, 1):
# 	if i < 101:
# 			continue
frame_count = 1
for data in dataset:
	frame = open_dataset.Frame()
	frame.ParseFromString(bytearray(data.numpy()))
	(range_images, camera_projections, seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
	(point,cp_point) = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)#获取激光点云
	#True: range, intensity, elongation, x, y, z
	points_temp, cp_points_temp = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
	points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=False)
	#print(cp_points)
	points_all = np.concatenate(points, axis=0)

	batch = 1
	num_cols = 2650

	intensity_list = []
	elongation_list= []
	for array in points_temp:
		intensity_list.append(array[:, 1].reshape(1, -1, 1))  # 提取强度信息并将其重塑为形状为 [1, N, 1]
		elongation_list.append(array[:, 2].reshape(1, -1, 1))

	intensity_tensor = np.concatenate(intensity_list, axis=1)  # 在第二个维度上连接强度信息张量
	elongation_tensor = np.concatenate(elongation_list, axis=1)

	cp_points_intensity_tensor = np.concatenate((intensity_tensor, elongation_tensor), axis=-1)  # 在最后一个维度上连接强度信息张量和全零张量
	cp_points_intensity_tensor_tf = tf.convert_to_tensor(cp_points_intensity_tensor, dtype=tf.float32)

	points_all = np.array(points_all)

	# 从 frame 中获取 laser_calibrations 中的 beam_inclinations
	beam_inclinations_proto = None
	laser_name = open_dataset.LaserName.TOP
	for calibration in frame.context.laser_calibrations:
			if calibration.name == laser_name:
				beam_inclinations_proto = calibration
				break

	if beam_inclinations_proto is not None:
    # 提取 beam_inclinations，并将其排序从高到低
		beam_inclinations = np.array(beam_inclinations_proto.beam_inclinations[:64], dtype=np.float32)
		sorted_beam_inclinations = np.sort(beam_inclinations)[::-1]  # 从高到低排序
    # 如果不足64个值，补零
		if len(sorted_beam_inclinations) < 64:
			sorted_beam_inclinations = np.pad(sorted_beam_inclinations, (0, 64 - len(sorted_beam_inclinations)), 'constant')
	else:
    # 如果没有找到符合条件的 laser_calibrations，则创建一个空的数组
		sorted_beam_inclinations = np.zeros(64, dtype=np.float32)

# 调整形状为 (1, 64)
	sorted_beam_inclinations = np.expand_dims(sorted_beam_inclinations, axis=0)

# 转换为张量
	inclinations_tensor = tf.constant(sorted_beam_inclinations, dtype=tf.float32)
	inclination = inclinations_tensor
	num_rows = inclination.shape[1]

	laser_name = open_dataset.LaserName.TOP  # 假设你需要的是 TOP激光雷达的 extrinsic
	extrinsic_proto = None
	for calibration in frame.context.laser_calibrations:
		if calibration.name == laser_name:
			extrinsic_proto = calibration.extrinsic
			break

# 将 extrinsic 转换为齐次变换矩阵形式
	extrinsic_values = np.array([
    [extrinsic_proto.transform[0], extrinsic_proto.transform[1], extrinsic_proto.transform[2], extrinsic_proto.transform[3]],
    [extrinsic_proto.transform[4], extrinsic_proto.transform[5], extrinsic_proto.transform[6], extrinsic_proto.transform[7]],
    [extrinsic_proto.transform[8], extrinsic_proto.transform[9], extrinsic_proto.transform[10], extrinsic_proto.transform[11]],
    [extrinsic_proto.transform[12], extrinsic_proto.transform[13], extrinsic_proto.transform[14], extrinsic_proto.transform[15]]
	])
	extrinsic = tf.constant([extrinsic_values], dtype=tf.float32)

	points2 = tf.tile(
        tf.expand_dims(points_all, axis=0), [batch, 1, 1])
	num_points_per_batch2 = points_all.shape[0]
	num_points2 = tf.constant([num_points_per_batch2], dtype=tf.int32)
	points_tensor2 = tf.convert_to_tensor(points2)



	range_image2 = build_intensity_range_image_from_point_cloud(
    points_tensor2, num_points2, extrinsic, inclination, [num_rows, num_cols],cp_points_intensity_tensor_tf)


	save_rgb_image(range_image2, folder_path, f"image_{frame_count}_original.png")
	print(f"Data saved successfully to original {frame_count}")
	#break

	frame_count += 1