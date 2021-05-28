# 使用训练好的权重进行物体检测（实际项目只需要此代码进行全流程检测）
# 此代码包括：地面分割-->聚类-->pointnet检测-->label转化为三维坐标框
import os
import glob
import argparse
import sys
from numpy.lib.type_check import imag
import progressbar
import datetime
import struct
import numpy as np
import open3d as o3d
import scipy
import pandas as pd
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt

'''
@description:定义分类映射表 
@param {*}
@return {*}
'''
decoder = {0: 'cyclist', 1: 'misc', 2: 'pedestrian', 3: 'vehicle'}


'''
@description:获取分类对应的物体检测框 
@param {*} segmented_objects
@param {*} object_ids
@param {*} predictions
@param {*} decoder
@return {*}
'''
def get_bounding_boxes(segmented_objects, object_ids, predictions, decoder):
    """
    Draw bounding boxes for surrounding objects according to classification result
        - red for pedestrian
        - blue for cyclist
        - green for vehicle
    Parameters
    ----------
    segmented_objects: open3d.geometry.PointCloud
        Point cloud of segmented objects
    object_ids: numpy.ndarray
        Object IDs as numpy.ndarray
    predictions:
        Object Predictions
    Returns
    ----------
    """
    # parse params:
    points = np.asarray(segmented_objects.points)

    # color cookbook:
    color = {
        # pedestrian as red:
        'pedestrian': np.asarray([0.5, 0.0, 0.0]),
        # cyclist as blue:
        'cyclist': np.asarray([0.0, 0.0, 0.5]),
        # vehicle as green:
        'vehicle': np.asarray([0.0, 0.5, 0.0]),
    }

    bounding_boxes = []
    for class_id in predictions:


        # get color
        class_name = decoder[class_id]

        if (class_name == 'misc'):
            continue

        class_color = color[class_name]
        # show instances:
        for object_id in predictions[class_id]:
            
            if predictions[class_id][object_id] < 0.9:
                continue

            # create point cloud:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                points[object_ids == object_id]
            )
            # create bounding box:
            bounding_box = object_pcd.get_axis_aligned_bounding_box()

            # set color according to confidence:
            confidence = predictions[class_id][object_id]
            bounding_box.color = tuple(
                class_color + (1.0 - confidence)*class_color
            )

            # update:
            bounding_boxes.append(bounding_box)
    
    return bounding_boxes


'''
@description: 将点从相机坐标系转换到像素坐标系
@param {*} X_cam
@param {*} param
@return {*}
'''
def transform_to_pixel(X_cam, param):
    '''
    Transform point from camera frame to pixel frame
    Parameters
    ----------
    X_cam: numpy.ndarray
        points in camera frame
    param: dict
        Vehicle parameters
    Returns
    ----------
    X_pixel: numpy.ndarray
        Points in pixel frame
    '''
    # get params:
    K, b = param['P2'][:,0:3], param['P2'][:,3]

    # project to pixel frame:
    X_pixel = np.dot(
        K, X_cam.T
    ).T + b

    # rectify:
    X_pixel = (X_pixel[:, :2].T / X_pixel[:, 2]).T

    return X_pixel

'''
@description:将点云从雷达坐标系转换到相机坐标系 
@param {*} X_velo
@param {*} param
@return {*}
'''
def transform_to_cam(X_velo, param):
    # get params:
    R0_rect = param['R0_rect']
    R_velo_to_cam, t_velo_to_cam = param['Tr_velo_to_cam'][:,0:3], param['Tr_velo_to_cam'][:,3]

    # project to unrectified camera frame:
    X_cam = np.dot(
        R_velo_to_cam, X_velo.T
    ).T + t_velo_to_cam

    # rectify:
    X_cam = np.dot(
       R0_rect, X_cam.T
    ).T

    return X_cam


def get_orientation_in_camera_frame(X_cam_centered):
    """
    Get object orientation using PCA
    """
    # keep only x-z:
    X_cam_centered = X_cam_centered[:, [0, 2]]

    H = np.cov(X_cam_centered, rowvar=False, bias=True)

    # get eigen pairs:
    eigenvalues, eigenvectors = np.linalg.eig(H)

    idx_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # orientation as arctan2(-z, x):
    return np.arctan2(-eigenvectors[0][1], eigenvectors[0][0])

'''
@description: 读取点云文件
@param {*} filepath
@return {*}
'''
def read_measurements(filepath):
    point_cloud = []
    with open(filepath, 'rb') as f:
        # unpack velodyne frame:
        content = f.read()
        measurements = struct.iter_unpack('ffff', content)
        for i, point in enumerate(measurements):
            x, y, z, intensity = point
            point_cloud.append([x, y, z, intensity])
    point_cloud = np.asarray(point_cloud, dtype=np.float32)

    return point_cloud

'''
@description:读取矫正矩阵 
@param {*} filepath
@return {*}
'''
def read_calib(filepath):
    DIMENSION = {
        'P0': (3, 4),
        'P1': (3, 4),
        'P2': (3, 4),
        'P3': (3, 4),
        'R0_rect': (3, 3),
        'Tr_velo_to_cam': (3, 4),
        'Tr_imu_to_velo': (3, 4)
    }

    param = {}
    # parse calibration data:
    with open(filepath, 'rt') as f:
        # one line per param:
        content = [tuple(i.split(':')) for i in f.read().strip().split('\n')]
        # format param as numpy.ndarray with correct shape
        for name, value in content:
            param[name] = np.asarray(
                [float(v) for v in value.strip().split()]
            ).reshape(
                DIMENSION[name]
            )
    
    return param

'''
@description: 地面分割和聚类
@param {*} point_cloud
@return {*}
'''
def segment_ground_and_objects(point_cloud):

    N, _ = point_cloud.shape

    # 法向量估计
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
    pcd_original.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=9
        )
    )

    # 使用法向量过滤
    normals = np.asarray(pcd_original.normals)
    angular_distance_to_z = np.abs(normals[:, 2])
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/6)

    # 平面分割
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(point_cloud[idx_downsampled])

    ground_model, idx_ground = pcd_downsampled.segment_plane(
        distance_threshold=0.30,
        ransac_n=3,
        num_iterations=1000
    )


    segmented_ground = pcd_downsampled.select_by_index(idx_ground)

    distance_to_ground = np.abs(
        np.dot(point_cloud,np.asarray(ground_model[:3])) + ground_model[3]
    )
    idx_cloud = distance_to_ground > 0.30

    segmented_objects = o3d.geometry.PointCloud()

    idx_segmented_objects = np.logical_and.reduce(
        [
            idx_cloud,
            point_cloud[:, 0] >=   1.95, point_cloud[:, 0] <=  80.00,
            point_cloud[:, 1] >= -30.00, point_cloud[:, 1] <= +30.00
        ]
    )

    segmented_objects.points = o3d.utility.Vector3dVector(
        point_cloud[idx_segmented_objects]
    )
    segmented_objects.normals = o3d.utility.Vector3dVector(
        np.asarray(pcd_original.normals)[idx_segmented_objects]
    )

    segmented_ground.paint_uniform_color([0.0, 0.0, 0.0])
    segmented_objects.paint_uniform_color([0.5, 0.5, 0.5])

    labels = np.asarray(segmented_objects.cluster_dbscan(eps=0.60, min_points=3))

    return segmented_ground, segmented_objects, labels


'''
@description:将预测数据转换为kitti格式 
@param {*} segmented_objects
@param {*} object_ids
@param {*} param
@param {*} predictions
@param {*} decoder
@return {*}
'''
def to_kitti_eval_format(segmented_objects, object_ids, param, predictions, decoder):
    # parse params:
    points = np.asarray(segmented_objects.points)

    # initialize KITTI label:
    label = {
        'type': [],
        'left': [], 'top': [], 'right': [], 'bottom': [],
        'height': [], 'width': [], 'length': [],
        'cx': [], 'cy': [], 'cz': [], 
        'ry': [], 
        # between 0 and 100:
        'score': []
    }
    formatter = lambda x: f'{x:.2f}'
    kitti_type = {
        'vehicle': 'Car',
        'pedestrian': 'Pedestrian',
        'cyclist': 'Cyclist',
        'misc': 'Misc'
    }

    for class_id in predictions:
        # get color
        class_name = decoder[class_id]

        if (class_name == 'misc'):
            continue
        
        # get KITTI type:
        class_name = kitti_type[class_name]

        # show instances:
        for object_id in predictions[class_id]:
            confidence = 100.0 * predictions[class_id][object_id]

            if confidence < 90:
                continue

            # set object type:
            label['type'].append(class_name)

            # transform to camera frame:
            X_velo = points[object_ids == object_id]
            X_cam = transform_to_cam(X_velo, param)

            # transform to pixel frame:
            X_pixel = transform_to_pixel(X_cam, param)

            # set 2D bounding box:
            top_left = X_pixel.min(axis = 0)
            bottom_right = X_pixel.max(axis = 0)

            label['left'].append(formatter(top_left[0]))
            label['top'].append(formatter(top_left[1]))
            label['right'].append(formatter(bottom_right[0]))
            label['bottom'].append(formatter(bottom_right[1]))

            # set object location:
            c_center = X_cam.mean(axis = 0)

            label['cx'].append(formatter(c_center[0]))
            label['cy'].append(formatter(c_center[1]))
            label['cz'].append(formatter(c_center[2]))

            # set object orientation:
            X_cam_centered = X_cam - c_center
            orientation = get_orientation_in_camera_frame(X_cam_centered)
            label['ry'].append(formatter(orientation))

            # project to object frame:
            cos_ry = np.cos(-orientation)
            sin_ry = np.sin(-orientation)

            R_obj_to_cam = np.asarray(
                [
                    [ cos_ry, 0.0, sin_ry],
                    [    0.0, 1.0,    0.0],
                    [-sin_ry, 0.0, cos_ry]
                ]
            )

            X_obj = np.dot(
                R_obj_to_cam.T, (X_cam_centered).T
            ).T

            # set object dimension:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                X_obj
            )
            bounding_box = object_pcd.get_axis_aligned_bounding_box()
            extent = bounding_box.get_extent()

            # height along y-axis:
            label['height'].append(formatter(extent[1]))
            # width along x-axis:
            label['width'].append(formatter(extent[0]))
            # length along z-axis:
            label['length'].append(formatter(extent[2]))

            # set confidence:
            label['score'].append(formatter(confidence))
    
    # format as pandas dataframe:
    label = pd.DataFrame.from_dict(
        label
    )
    
    # set value for unavailable fields:
    label['truncated'] = -1
    label['occluded'] = -1
    # don't evaluate AOS:
    label['alpha'] = -10

    # set column order:
    label = label[
        [
            'type',
            'truncated',
            'occluded',
            'alpha',
            'left', 'top', 'right', 'bottom',
            'height', 'width', 'length',
            'cx', 'cy', 'cz', 'ry',
            'score'
        ]
    ]

    return label


'''
@description: 数据预处理
@param {*}
@return {*}
'''
def preprocess(
    segmented_objects, object_ids,
    config
):
    # parse config:
    points = np.asarray(segmented_objects.points)
    normals = np.asarray(segmented_objects.normals)
    num_objects = max(object_ids) + 1

    # result:
    X = []
    y = []

    # 遍历每一个聚类
    for object_id in range(num_objects):
        # 1. only keep object with enough number of observations:
        if ((object_ids == object_id).sum() <= 4):
            continue
        
        # 2. only keep object within max radius distance:
        object_center = np.mean(points[object_ids == object_id], axis=0)[:2]
        if (np.sqrt((object_center*object_center).sum()) > config['max_radius_distance']):
            continue
        
        # 3. 重采样过程:
        points_ = np.copy(points[object_ids == object_id])
        normals_ = np.copy(normals[object_ids == object_id])
        N, _ = points_.shape

        weights = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(points_, 'euclidean')
        ).mean(axis = 0)
        weights /= weights.sum()
        
        idx = np.random.choice(
            np.arange(N), 
            size = (config['num_sample_points'], ), replace=True if config['num_sample_points'] > N else False,
            p = weights
        )

        # 4. translate to zero-mean:
        points_processed, normals_processed = points_[idx], normals_[idx]
        points_processed -= points_.mean(axis = 0)

        # format as numpy.ndarray:
        X.append(
            np.hstack(
                (points_processed, normals_processed)
            )
        )
        y.append(object_id)

    # format as tf dataset:
    X = np.asarray(X)
    y = np.asarray(y)

    # pad to batch size:
    N = len(y)
    if (N % config['batch_size'] != 0):
        num_repeat = config['batch_size'] - N % config['batch_size']

        X = np.vstack(
            (
                X, 
                np.repeat(
                    X[0], num_repeat, axis=0
                ).reshape(
                    (-1, config['num_sample_points'], 6)
                )
            )
        )
        y = np.hstack(
            (y, np.repeat(y[0], num_repeat))
        )

    # format as tensorflow dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(X, dtype=tf.float32), 
            tf.convert_to_tensor(y, dtype=tf.int64)
        )
    )
    dataset = dataset.batch(batch_size=config['batch_size'], drop_remainder=True)

    return dataset, N


'''
@description: 预测函数
@param {*} segmented_objects
@param {*} object_ids
@param {*} model
@return {*}
'''
def predict(segmented_objects, object_ids, model, config):
    # 数据预处理:
    dataset, N = preprocess(segmented_objects, object_ids, config)

    # 进行预测:
    predictions = {
        class_id: {} for class_id in range(config['num_classes'])
    }
    num_predicted = 0

    for X, y in dataset:
        # predict:
        prob_preds = model(X)
        ids = y.numpy()

        # add to prediction:
        for (object_id, class_id, confidence) in zip(
            # object ID:
            ids,
            # category:
            np.argmax(prob_preds, axis=1),
            # confidence:
            np.max(prob_preds, axis=1)
        ):
            predictions[class_id][object_id] = confidence
            num_predicted += 1
            
            # 跳过为了保证数据整齐性而扩充的部分:
            if (num_predicted == N):
                break
    
    return predictions


'''
@description:获取命令行参数 
@param {*}
@return {*}
'''
def get_arguments():

    parser = argparse.ArgumentParser("Perform two-stage object detection on KITTI dataset.")


    parser.add_argument("--input_dir",type=str,default="/home/teamo/point_process_piplines/KITTI/object/training")

    parser.add_argument(
        "--debug",type=bool, default=True
    )
    parser.add_argument(
        "--max_radius_distance", type=float, default=25.0
    )
    parser.add_argument(
        "--num_sample_points",type=int, default=64
    )
    parser.add_argument(
        "--trained_model",type=str, default="pointnet_model.h5"
    )
    return parser.parse_args()

'''
@description:物体检测主函数 
@param {*}
@return {*}
'''
def detect(
    dataset_dir, index,
    max_radius_distance, num_sample_points,
    debug_mode,
    model
):
    # 设置路径:
    input_velodyne = os.path.join(dataset_dir, 'velodyne', f'{index:06d}.bin')
    input_params = os.path.join(dataset_dir, 'calib', f'{index:06d}.txt')
    output_label = os.path.join("/home/teamo/point_process_piplines/KITTI/object/object_training_datasets", 
                                'predict_result_2', 'data', f'{index:06d}.txt')


    # 1. 读取点云文件和相机矫正文件:
    point_cloud = read_measurements(input_velodyne)
    param = read_calib(input_params)

    # 2. 地面分割和聚类:
    segmented_ground, segmented_objects, object_ids = segment_ground_and_objects(point_cloud[:, 0:3])

    # 3. 使用分类网络进行预测:
    config = {
        # preprocess:
        'max_radius_distance': max_radius_distance,
        'num_sample_points': num_sample_points,
        # predict:
        'msg' : True,
        'batch_size' : 16,
        'num_classes' : 4
    }

    predictions = predict(segmented_objects, object_ids, model, config)

    # debug mode:
    if (debug_mode):
        # print detection results:
        for class_id in predictions:
            # show category:
            print(f'[{decoder[class_id]}]')
            # show instances:
            for object_id in predictions[class_id]:
                print(f'\t[Object ID]: {object_id}, confidence {predictions[class_id][object_id]:.2f}')

        # visualize:
        bounding_boxes = get_bounding_boxes(
            segmented_objects, object_ids, 
            predictions, decoder
        )
        o3d.visualization.draw_geometries(
            [segmented_ground, segmented_objects] + bounding_boxes
        )

      
    
    # 4. 保存为kitti格式:
    label = to_kitti_eval_format(
        segmented_objects, object_ids, param,
        predictions, decoder
    )
    label.to_csv(output_label, sep=' ', header=False, index=False)


if __name__ == "__main__":
    # 获取命令行参数
    args = get_arguments()
    prog = progressbar.ProgressBar()

    # 建立输出文件夹

    # 载入模型
    model = tf.keras.models.load_model(arg.trained_model)

    for label in prog(
        glob.glob(
            os.path.join(args.input_dir, 'label_2', '*.txt')
        )
    ):
        # 获取索引:
        index = int(
            os.path.splitext(
                os.path.basename(label)
            )[0]
            
        )
        print(index)

        # 进行检测:
        detect(
            args.input_dir, index,
            args.max_radius_distance, args.num_sample_points,
            args.debug,
            model
        )