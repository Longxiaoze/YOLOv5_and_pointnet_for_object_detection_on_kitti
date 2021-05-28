# 用于根据kitti点云，相机矫正文件以及左眼图片来生成用于训练pointnet的数据集

import argparse
import glob
import os
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
import progressbar
import shutil
import numpy as np
import struct
import pandas as pd
import open3d as o3d
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt


'''
@description:添加标签 
@param {*} dataset_label
@param {*} category
@param {*} label
@param {*} N
@param {*} center
@return {*}
'''
def add_label(dataset_label, category, label, N, center):

    if label is None:
        # kitti category:
        dataset_label[category]['type'].append('Unknown')
        dataset_label[category]['truncated'].append(-1)
        dataset_label[category]['occluded'].append(-1)

        # bounding box labels:
        dataset_label[category]['height'].append(-1)
        dataset_label[category]['width'].append(-1)
        dataset_label[category]['length'].append(-1)
        dataset_label[category]['ry'].append(-10)
    else:
        # kitti category:
        dataset_label[category]['type'].append(label['type'])
        dataset_label[category]['truncated'].append(label['truncated'])
        dataset_label[category]['occluded'].append(label['occluded'])

        # bounding box labels:
        dataset_label[category]['height'].append(label['height'])
        dataset_label[category]['width'].append(label['width'])
        dataset_label[category]['length'].append(label['length'])
        dataset_label[category]['ry'].append(label['ry'])

    # distance and num. of measurements:
    dataset_label[category]['num_measurements'].append(N)
    vx, vy, vz = center
    dataset_label[category]['vx'].append(vx)
    dataset_label[category]['vy'].append(vy)
    dataset_label[category]['vz'].append(vz)


'''
@description:获取对应的类别标签 
@param {*} object_type
@return {*}
'''
def get_object_category(object_type):

    category = 'vehicle'

    if object_type is None or object_type == 'Misc' or object_type == 'DontCare':
        category = 'misc'
    elif object_type == 'Pedestrian' or object_type == 'Person_sitting':
        category = 'pedestrian'
    elif object_type == 'Cyclist':
        category = 'cyclist'


    return category

'''
@description: 获取对应的点云
@param {*} pcd
@param {*} idx
@param {*} N
@return {*}
'''
def get_object_pcd_df(pcd, idx, num_sample):


        
    df_point_cloud_with_normal = pd.DataFrame(
        data = np.hstack(
            (
                np.asarray(pcd.points)[idx],
                np.asarray(pcd.normals)[idx]
            )
        ),
        index = np.arange(N),
        columns = ['vx', 'vy', 'vz', 'nx', 'ny', 'nz']
    )

    return df_point_cloud_with_normal


'''
@description: 根据矩形框边界对点云进行滤波 
@param {*} X
@param {*} labels
@param {*} dims
@return {*}
'''
def filter_by_bouding_box(X, labels, dims):


    # filter by bounding box in object frame:
    idx_obj = np.all(
        np.logical_and(
            X >= -dims/2,
            X <=  dims/2
        ),
        axis = 1
    )

    if idx_obj.sum() == 0:
        return None

    # get object ID using non-maximum suppression:
    ids, counts = np.unique(
        labels[idx_obj], return_counts=True
    )
    object_id, _ = max(zip(ids, counts), key=lambda x:x[1]) 

    return object_id


'''
@description: 将点云从雷达坐标系转换到相机坐标系 
@param {*} X_velo
@param {*} param
@param {*} t_obj_to_cam
@param {*} ry
@return {*}
'''
def transform_from_velo_to_obj(X_velo, param, t_obj_to_cam, ry):

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

    # project to object frame:
    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)

    R_obj_to_cam = np.asarray(
        [
            [ cos_ry, 0.0, sin_ry],
            [    0.0, 1.0,    0.0],
            [-sin_ry, 0.0, cos_ry]
        ]
    )

    X_obj = np.dot(
        R_obj_to_cam.T, (X_cam - t_obj_to_cam).T
    ).T

    return X_obj

'''
@description: 对点云进行地面分割和聚类 
@param {*} point_cloud
@return {*}
'''
def segment_ground_and_objects(point_cloud):
    N, _ = point_cloud.shape


    # 根据法向量进行滤波
    # 估计每个点的法向量
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
    pcd_original.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=9
        )
    )

    # 提取z轴角度大于30度的点
    normals = np.asarray(pcd_original.normals)
    angular_distance_to_z = np.abs(normals[:, 2])
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/6)

    # 使用open3d的函数进行平面分割
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(point_cloud[idx_downsampled])
    ground_model, idx_ground = pcd_downsampled.segment_plane(
        distance_threshold=0.30,
        ransac_n=3,
        num_iterations=1000
    )

    # 提取到的属于地面的点
    segmented_ground = pcd_downsampled.select_by_index(idx_ground)

    # 计算原始点云中每个点与地面的距离
    distance_to_ground = np.abs(
        np.dot(point_cloud,np.asarray(ground_model[:3])) + ground_model[3]
    )

    # 提取非地面的点
    idx_cloud = distance_to_ground > 0.30

    # limit FOV to front:
    segmented_objects = o3d.geometry.PointCloud()

    # 选取可视范围内的点
    idx_segmented_objects = np.logical_and.reduce(
        [
            idx_cloud,
            point_cloud[:, 0] >=   1.95, point_cloud[:, 0] <=  80.00,
            point_cloud[:, 1] >= -30.00, point_cloud[:, 1] <= +30.00
        ]
    )

    # 最后得到的非地面的点
    segmented_objects.points = o3d.utility.Vector3dVector(
        point_cloud[idx_segmented_objects]
    )
    segmented_objects.normals = o3d.utility.Vector3dVector(
        np.asarray(pcd_original.normals)[idx_segmented_objects]
    )

    # 对地面与非地面的点绘制不同的颜色
    segmented_ground.paint_uniform_color([0.0, 0.0, 0.0])
    segmented_objects.paint_uniform_color([0.5, 0.5, 0.5])

    
    # DBSACN聚类，这里结果的格式为N by 1的数据，-1代表噪声数据
    labels = np.asarray(segmented_objects.cluster_dbscan(eps=0.60, min_points=3))

    return segmented_ground, segmented_objects, labels

'''
@description: 将坐标从相机坐标系转换到雷达坐标系 
@param {*} X_cam
@param {*} param
@return {*}
'''
def transform_from_cam_to_velo(X_cam, param):
    # get params:
    R0_rect = param['R0_rect']
    R_velo_to_cam, t_velo_to_cam = param['Tr_velo_to_cam'][:,0:3], param['Tr_velo_to_cam'][:,3]

    # unrectify:
    X_velo = np.dot(
        R0_rect.T, X_cam.T
    ).T

    # project to velo frame:
    X_velo = np.dot(
        R_velo_to_cam.T, (X_velo - t_velo_to_cam).T
    ).T

    return X_velo

'''
@description:读取label文件，并且将标定框的中心坐标从相机坐标系转换到雷达坐标系 
@param {*} filepath
@param {*} param
@return {*}
'''
def read_label(filepath, param):
    
    # load data:    
    df_label = pd.read_csv(
        filepath,
        sep = ' ', header=None
    )

    # add attribute names:
    df_label.columns = [
        'type',
        'truncated',
        'occluded',
        'alpha',
        'left', 'top', 'right', 'bottom',
        'height', 'width', 'length',
        'cx', 'cy', 'cz', 'ry'
    ]

    # filter label with no dimensions:
    condition = (
        (df_label['height'] >= 0.0) &
        (df_label['width'] >= 0.0) &
        (df_label['length'] >= 0.0)
    )
    df_label = df_label.loc[
        condition, df_label.columns
    ]

    #
    # get object center in velo frame:
    #
    centers_cam = df_label[['cx', 'cy', 'cz']].values
    centers_velo = transform_from_cam_to_velo(centers_cam, param)
    # add height bias:
    df_label['vx'] = df_label['vy'] = df_label['vz'] = 0.0
    df_label[['vx', 'vy', 'vz']] = centers_velo
    df_label['vz'] += df_label['height']/2

    # add radius for point cloud extraction:
    df_label['radius'] = df_label.apply(
        lambda x: np.linalg.norm(
            0.5*np.asarray(
                [x['height'], x['width'], x['length']]
            )
        ),
        axis = 1
    )

    return df_label


'''
@description: 用于读取相机矫正文件和转换矩阵 
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
@description: 用于读取点云数据 
@param {*} filepath
@return {*}
'''
def read_velodyne_bin(filepath):
    point_cloud = []
    with open(filepath, 'rb') as f:
        # unpack velodyne frame:
        content = f.read()
        measurements = struct.iter_unpack('ffff', content)
        # parse:
        for i, point in enumerate(measurements):
            x, y, z, intensity = point
            point_cloud.append([x, y, z, intensity])
    # format for output
    point_cloud = np.asarray(point_cloud, dtype=np.float32)

    return point_cloud

'''
@description: 为每种标签建立对应的字典
@param {*}
@return {*}
'''
def init_label():
    return {
        # original category:
        'type': [],
        'truncated': [],
        'occluded': [],
        # distance and num. of measurements:
        'vx': [], 
        'vy': [], 
        'vz': [], 
        'num_measurements': [],
        # bounding box labels:
        'height': [], 
        'width': [], 
        'length':[], 
        'ry':[]
    }

'''
@description: 用于获取命令行参数 
@param {*}
@return {*}
'''
def get_args():
    parser = argparse.ArgumentParser("generate training set")
    parser.add_argument("--input_dir",type=str,default="./KITTI/object/training")
    parser.add_argument("--output_dir",type=str,default="./KITTI/object/object_detect_datasets")
    parser.add_argument("--max_distance",type=float,default=45.0)#默认对45米内的物体进行检测
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_distance = args.max_distance

    # 获取label文件数量
    N = len(glob.glob(os.path.join(input_dir,"label_2","*.txt")))
    
    # 建立输出文件夹
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    else:
        os.mkdir(output_dir)


    # 为每种分类建立对应的文件夹
    dataset_label = {}
    for cat in ['vehicle', 'pedestrian', 'cyclist', 'misc']:
        os.mkdir(os.path.join(output_dir,cat))
        dataset_label[cat] = init_label()

    progres = progressbar.ProgressBar()

    # 遍历每一帧点云并提取
    for index in progres(range(N)):

        # # 仅用于测试
        # image = plt.imread(
        # os.path.join(input_dir, 'image_2', f'{index:06d}.png')
        # )

        # points_vechile = np.empty(shape=(0,3))
        # points_pedestrian = np.empty(shape=(0,3))
        # points_cyclist = np.empty(shape=(0,3))
        # points_misc = np.empty(shape=(0,3))



        # 读取对应的点云
        point_cloud = read_velodyne_bin(
        os.path.join(input_dir, 'velodyne', f'{index:06d}.bin'))

        # 读取对应的相机矫正文件以及转换矩阵
        param = read_calib(
        os.path.join(input_dir, 'calib', f'{index:06d}.txt'))

        # 读取label文件
        df_label = read_label(
        os.path.join(input_dir, 'label_2', f'{index:06d}.txt'),param)
        
        # 地面分割和聚类
        # 返回的结果为地面点云，非地面点云，对于非地面点云的聚类标签
        segmented_ground, segmented_objects, object_ids = segment_ground_and_objects(point_cloud[:, 0:3])
        
        # 建立搜索树
        search_tree = o3d.geometry.KDTreeFlann(segmented_objects)

        identified = set()
        # 遍历label中的每一行
        for idx, label in df_label.iterrows():
            # 提取对应的参数
            center_velo = np.asarray([label['vx'], label['vy'], label['vz']])

            #print(np.linalg.norm(center_velo))

            if np.linalg.norm(center_velo) > max_distance:
                continue
            
            center_cam = np.asarray([label['cx'], label['cy'], label['cz']])

            # dimensions in camera frame:
            dims = np.asarray([label['length'], label['height'], label['width']])
            

            # 以物体框在雷达坐标系中的中心为原点，搜索指定半径内的点
            [k, idx, _] = search_tree.search_radius_vector_3d(
                center_velo, 
                label['radius']
            )

            if (k > 0):     
                # 半径内的点云
                point_cloud_velo_ = np.asarray(segmented_objects.points)[idx]
                # 半径内点云对应的聚类结果
                object_ids_ = object_ids[idx]


                # 将提取到的点云从雷达坐标系转换到相机坐标系
                point_cloud_obj = transform_from_velo_to_obj(
                    point_cloud_velo_, 
                    param, 
                    center_cam, 
                    label['ry']
                )

                # add bias along height:
                point_cloud_obj[:, 1] += label['height']/2


                # 进行矩形框滤波，并获取对应的聚类id（在矩形框内点数最多的类别的id）
                object_id_ = filter_by_bouding_box(point_cloud_obj, object_ids_, dims)

                if object_id_ is None:
                    continue
                    
                identified.add(object_id_)

                # 获取对应聚类的点云id
                idx_object = np.asarray(idx)[object_ids_ == object_id_]


                # 构建对应的点云dataframe
                N = len(idx_object)
                df_point_cloud_with_normal = get_object_pcd_df(segmented_objects, idx_object, N)

                # 获取类别:
                category = get_object_category(label['type'])
                # 获取对应点云的中心
                center = np.asarray(segmented_objects.points)[idx_object].mean(axis = 0)

                add_label(dataset_label, category, label, N, center)





                # 输出结果
                dataset_index = len(dataset_label[category]['type'])
                df_point_cloud_with_normal.to_csv(
                    os.path.join(output_dir, category, f'{dataset_index:06d}.txt'),
                    index=False, header=None
                )

        # plt.imshow(image)
        # plt.show()

        # 可视化查看
        # pcd_vechile = o3d.geometry.PointCloud()
        # pcd_vechile.points = o3d.utility.Vector3dVector(points_vechile)
        # pcd_vechile.paint_uniform_color([1,0,0])
        # pcd_pedestrian = o3d.geometry.PointCloud()
        # pcd_pedestrian.points = o3d.utility.Vector3dVector(points_pedestrian)
        # pcd_pedestrian.paint_uniform_color([0,1,0])
        # pcd_cyclist = o3d.geometry.PointCloud()
        # pcd_cyclist.points = o3d.utility.Vector3dVector(points_cyclist)
        # pcd_cyclist.paint_uniform_color([0,0,1])

        # o3d.visualization.draw_geometries([segmented_ground,segmented_objects,pcd_vechile,pcd_pedestrian,pcd_cyclist])
    for category in ['vehicle', 'pedestrian', 'cyclist', 'misc']:
        dataset_label[category] = pd.DataFrame.from_dict(
            dataset_label[category]
        )
        dataset_label[category].to_csv(
            os.path.join(output_dir, f'{category}.txt'),
            index=False
        )           
