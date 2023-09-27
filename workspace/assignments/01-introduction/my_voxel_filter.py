import open3d as o3d
import numpy as np
import argparse
import random

def voxel_filter(data, r_size, mode = 0):
  data = data[:,:3]
  print("数据总点数为: ", data.shape[0])
  min_data = np.min(data, axis=0)
  max_data = np.max(data, axis=0)

  # compute dimensions of the voxel grid 向上取整
  Dx = np.ceil((max_data[0] - min_data[0])/r_size).astype(np.int32)
  Dy = np.ceil((max_data[1] - min_data[1])/r_size).astype(np.int32)
  Dz = np.ceil((max_data[2] - min_data[2])/r_size).astype(np.int32)
  print(Dx, Dy, Dz)

  # compute voxel grid index for each point
  hx = np.floor((data[:,0] - min_data[0])/r_size).astype(np.int32)
  hy = np.floor((data[:,1] - min_data[1])/r_size).astype(np.int32)
  hz = np.floor((data[:,2] - min_data[2])/r_size).astype(np.int32)
  h = hx + Dx * hy + Dx * Dy * hz
  sort = h.argsort()
  print(sort)
  point_cloud = data[sort]
  print(len(point_cloud))
  filtered_points = []
  voxel = []
  for i in range(len(point_cloud)):
    if i < len(point_cloud) - 1 and h[sort][i] == h[sort][i+1]:
      voxel.append(i)
    else:
      voxel.append(i)
      if mode == 0:
          # centroid
          choice_point = np.mean(point_cloud[i-len(voxel)+1:i+1, :], axis=0)
      else:
          # random
          random_index = random.randint(0,len(voxel)-1)
          choice_point = point_cloud[voxel[random_index],:]

      filtered_points.append(choice_point)
      voxel = []

  filtered_points = np.array(filtered_points, dtype=np.float64)
  print(filtered_points.shape)
  return filtered_points      
  


if __name__ == '__main__':
  filename = "../modelnet40_normal_resampled/airplane/airplane_0001.txt"
  data = np.loadtxt(filename, delimiter=",")
  
  filteded_pt = voxel_filter(data, 0.08, 1)
  point_cloud_o3d = o3d.geometry.PointCloud()
  point_cloud_o3d.points = o3d.utility.Vector3dVector(data[:,0:3])
  o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

  point_cloud_filter = o3d.geometry.PointCloud()
  point_cloud_filter.points = o3d.utility.Vector3dVector(filteded_pt)
  o3d.visualization.draw_geometries([point_cloud_filter]) # 显示原始点云