import pandas as pd
import open3d as o3d
import open3d

class pointcloud:
    def __init__(self):
        pass
    
    # 从文件中读取点云
    @staticmethod
    def read_pointcloud(file_name:str)-> open3d.geometry.PointCloud:
        df = pd.read_csv(file_name,header = None)
        df.columns = ["x","y","z",
                        "nx","ny","nz"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[["x","y","z"]].values)
        pcd.normals = o3d.utility.Vector3dVector(df[["nx","ny","nz"]].values)

        return pcd


