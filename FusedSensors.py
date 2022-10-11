from waymo_open_dataset.utils import frame_utils
import numpy as np
import tensorflow as tf


class FusedSensors:
    def __init__(self, frame):
        self.frame = frame
        (self.range_images, self.camera_projections, self.seg_labels,
         self.range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)

    def extract_lidar_data(self):
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            self.frame,
            self.range_images,
            self.camera_projections,
            self.range_image_top_pose)
        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)
        return points_all, cp_points_all

    def extract_ri2_data(self):
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
            self.frame,
            self.range_images,
            self.camera_projections,
            self.range_image_top_pose,
            ri_index=1)

        points_all_ri2 = np.concatenate(points_ri2, axis=0)
        cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)
        return points_all_ri2, cp_points_all_ri2

    def __extract_images(self):
        images = sorted(self.frame.images, key=lambda i: i.name)
        return images

    def get_front_image(self):
        front_cam_index = 0
        raw_img = self.__extract_images()[front_cam_index].image
        return tf.image.decode_jpeg(raw_img)

    def projected_points(self):
        points_all, cp_points_all = self.extract_lidar_data()

        # The distance between lidar points and vehicle frame origin.
        points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
        images = self.__extract_images()
        mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

        cp_points_all_tensor = tf.cast(tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

        projected_points_all_from_raw_data = tf.concat(
            [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
        return projected_points_all_from_raw_data