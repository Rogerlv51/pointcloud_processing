import os
import argparse
import progressbar

import numpy as np
import open3d as o3d
import pandas as pd
import copy
import Utils


# 获取命令行参数
def get_args():
    parase = argparse.ArgumentParser("Registration")
    parase.add_argument("--pointcloud_dir",type=str,default="/workspace/Data/registration_dataset")
    parase.add_argument("--voxel_size",type=float,default=1)


    return parase.parse_args()


if __name__ == "__main__":

    args = get_args()
    # 获取文件夹路径
    datasets_dir = args.pointcloud_dir
    
    # 进度条
    progress = progressbar.ProgressBar()


    # 以pandas.dataframe格式读取结果文件
    registration_results = Utils.Homework.read_registration_results(os.path.join(datasets_dir,"reg_result.txt"))

    # 初始化输出文件结构
    df_output = Utils.Homework.init_output()

    # 体素滤波的尺寸
    voxel_size = args.voxel_size

    # 迭代reg_result中的每一行来获取需要配准的点云文件
    for index,row in progress(list(registration_results.iterrows())):
        
        # 需要配准的点云文件名称
        idx_target = int(row["idx1"])
        idx_source = int(row["idx2"])
        
        # 读取点云,输出格式为open3d的点云格式
        pcd_source = Utils.pointcloud.read_point_cloud_bin(os.path.join(datasets_dir,"point_clouds",f"{idx_source}.bin"))
        pcd_target = Utils.pointcloud.read_point_cloud_bin(os.path.join(datasets_dir,"point_clouds",f"{idx_target}.bin"))

        # 在配准前可视化
        # Utils.Homework.draw_registration_result(pcd_source,pcd_target,np.identity(4))

        # 降采样
        pcd_down_source = pcd_source.voxel_down_sample(voxel_size)
        pcd_down_target = pcd_target.voxel_down_sample(voxel_size)

        # 绘制降采样后的结果
        # Utils.Homework.draw_registration_result(pcd_down_source,pcd_down_target,np.identity(4))

        # 最近邻搜索半径
        radius_feature = voxel_size * 5

        # 计算fpfh描述子
        pcd_fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down_source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        pcd_fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down_target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        # 距离阈值
        distance_threshold = voxel_size * 0.5

        # 使用open3d的RANSAC流程函数进行global registration
        init_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_down_source, pcd_down_target, pcd_fpfh_source, pcd_fpfh_target, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
        # 这三个参数用来评价RANSAC配准的结果是否有效
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(              # 这个参数用来检测配准之后的两个点云的几何不变性       
                0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(                # 这个用于检测配准之后两个点云的距离接近性
                distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(10*2*3.14/180)    # 这个用于检测配准后两个点云的法向量相似性
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))             # RANSAC的最大迭代次数及置信概率

        #  绘制配准后的结果
        # Utils.Homework.draw_registration_result(pcd_down_source, pcd_down_target, init_result.transformation)

        distance_threshold = 0.05 * 0.4

        # 使用ICP优化结果 这里由于已经有了初始值，选择使用原始点云
        final_result = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, distance_threshold, init_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # 绘制最终配准后的结果
        # Utils.Homework.draw_registration_result(pcd_source, pcd_target, final_result.transformation)

        # 往结果dict中加入数据
        Utils.Homework.add_to_output(df_output, idx_target, idx_source, final_result.transformation)
    
    # 将结果写入文件
    Utils.Homework.write_output(
        os.path.join(datasets_dir, 'reg_result_teamo.txt'),
        df_output)