import argparse             # 命令行参数获取
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import Utils
import numpy
import open3d


class FPFH_decriptor:
    def __init__(self) -> None:
        self.__B = 11
        self.__pointclouds = None
        self.__feature_index = None
        self.__seacrch_tree = None
        self.__radius = 1
        self.__keypoints = None
        self.__descriptors = None
        self.__search_tree = None
        pass
    
    
    def set_pointclouds(self,pointclouds:open3d.geometry.PointCloud) -> None:
        self.__pointclouds = pointclouds
        self.__search_tree = o3d.geometry.KDTreeFlann(self.__pointclouds)

    def set_attribute(self,B:int,radius:float) -> None:
        self.__B = B
        self.__radius = radius

    def set_keypoints(self,keypoints:numpy.ndarray) -> None:
        self.__keypoints = keypoints

    def __SFPH(self,keypoint:numpy.ndarray) -> None:
        # 获取点云
        points = np.asarray(self.__pointclouds.points)
        normals = np.asarray(self.__pointclouds.normals)  

        # 获取邻居节点
        [k,idx_neighbors,_] = self.__search_tree.search_radius_vector_3d(keypoint,self.__radius)
        
        # 获取n1
        n1 = normals[idx_neighbors[0]]

        # 移除关键点本身
        idx_neighbors = idx_neighbors[1:]

        # 计算 (p2-p1)/norm(p2-p1)
        diff = points[idx_neighbors] - keypoint
        diff = diff/np.reshape(np.linalg.norm(diff,ord=2,axis=1),(k-1,1))

        u = n1
        v = np.cross(u,diff)
        w = np.cross(u,v)    

        # 计算n2
        n2 = normals[idx_neighbors]

        # 计算alpha
        alpha = np.reshape((v*n2).sum(axis = 1),(k-1,1))

        # 计算phi
        phi = np.reshape((u*diff).sum(axis = 1),(k-1,1))

        # 计算 theta
        theta = np.reshape(np.arctan2((w*n2).sum(axis = 1),(u*n2).sum(axis = 1)),(k-1,1))

        # 计算相应的直方图
        alpha_hist = np.reshape(np.histogram(alpha,self.__B,range=[-1.0,1.0])[0],(1,self.__B))
        phi_hist = np.reshape(np.histogram(phi,self.__B,range=[-1.0,1.0])[0],(1,self.__B))
        theta_hist = np.reshape(np.histogram(theta,self.__B,range=[-3.14,3.14])[0],(1,self.__B))

    
        # 组成描述子
        fpfh = np.hstack((alpha_hist,phi_hist,theta_hist))

        return fpfh

    def describe(self) -> None:

        pointclouds = np.asarray(self.__pointclouds.points)

        self.__descriptors = numpy.ndarray((0,3*self.__B))

        for keypoint in self.__keypoints:

            # 寻找keypoint的邻居点
            [k,idx_neighbors,_] = self.__search_tree.search_radius_vector_3d(keypoint,self.__radius)

            # 移除关键点本身
            idx_neighbors = idx_neighbors[1:]

            # 计算权重
            w = 1.0/np.linalg.norm(keypoint - pointclouds[idx_neighbors],ord = 2,axis = 1)

            # 计算邻居的SPFH
            neighbors_SPFH = np.reshape(np.asarray([self.__SFPH(pointclouds[i]) for i in idx_neighbors]),(k-1,3*self.__B))

            # 计算自身的描述子
            self_SFPH = self.__SFPH(keypoint)

            # 计算最终的FPFH
            neighbors_SPFH = 1.0/(k-1)*np.dot(w,neighbors_SPFH)

            # 获取描述子并归一化
            finial_FPFH = self_SFPH + neighbors_SPFH
            finial_FPFH = finial_FPFH/np.linalg.norm(finial_FPFH)

            self.__descriptors = numpy.vstack((self.__descriptors,finial_FPFH))

    def get_descriptors(self) -> list:
        return self.__descriptors 



# 获取命令行参数
def get_args():

    parase = argparse.ArgumentParser(description="arg parser")
    parase.add_argument("--file_name",type=str,default= "/workspace/Data/modelnet40_normal_resampled/chair/chair_0001.txt")               # 需要提取描述子的的点云文件路径
    parase.add_argument("--radius",type=float,default= 0.05)                                             # PFPH需要的参数
    parase.add_argument("--x",type=float,default=0.084)                                                  # 特征点的x坐标
    parase.add_argument("--y",type=float,default=0.2597)                                                 # 特征点的y坐标
    parase.add_argument("--z",type=float,default=-0.0713)                                                # 特征点的z坐标
    parase.add_argument("--B",type=int,default=11)                                                       # 描述子的分段数

    return parase.parse_args()



if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    radius = args.radius
    file_name = args.file_name
    B = args.B
    
    # 读取点云
    pointclouds = Utils.pointcloud.read_pointcloud(file_name)

    
    # 创建特征点
    key_point1 = np.array([0.4333,-0.7807,-0.4372])
    key_point2 = np.array([-0.4240,-0.7850,-0.4392])
    key_point3 = np.array([0.02323,0.004715,-0.2731])

    keypoints = np.vstack((key_point1,key_point2,key_point3))

    # 创建描述子
    descriptor = FPFH_decriptor()
    descriptor.set_pointclouds(pointclouds)
    descriptor.set_keypoints(keypoints)
    descriptor.describe()
    descriptors = descriptor.get_descriptors()

    print(descriptors.shape)


    plt.plot(range(3*B), descriptors[0,:].T, ls="-.",color="r",marker =",", lw=2, label="keypoint1")
    plt.plot(range(3*B), descriptors[1,:].T, ls="-.",color="g",marker =",", lw=2, label="keypoint2")
    plt.plot(range(3*B), descriptors[2,:].T, ls="-.",color="b",marker =",", lw=2, label="keypoint3")

    plt.legend()

    plt.show()


    
