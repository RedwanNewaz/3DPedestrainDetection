import tensorflow as tf
from FusedSensors import FusedSensors
from waymo_open_dataset import dataset_pb2 as open_dataset
from VizWaymoData import show_point_cloud
import numpy as np



if __name__ == '__main__':
    FILENAME = 'dataset/testing_segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        data = FusedSensors(frame)
        pcl, cp_pcl = data.extract_lidar_data()
        img = data.get_front_image()
        pcl_projection = data.projected_points()

        # show visualization
        show_point_cloud(pcl, frame.laser_labels)
