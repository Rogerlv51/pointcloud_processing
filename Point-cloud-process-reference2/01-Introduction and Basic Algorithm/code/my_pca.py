import open3d as o3d
import numpy as np

def pca(data):   # 事实上这个函数根本不需要加sort操作，默认索引0就是主方向
  X = data.T
  X_mean = np.mean(X, axis=1).reshape(3,1)
  X = X - X_mean
  H = X.dot(X.T)
  eigenvalues, eigenvectors = np.linalg.eig(H)
  return eigenvalues[0], eigenvectors[:,0]

if __name__ == '__main__':
  # 数据集中是以txt存储的x,y,z和方向信息
  filename = "../../modelnet40_normal_resampled/airplane/airplane_0001.txt"
  data = np.loadtxt(filename, delimiter=",")
  
  point_cloud_o3d = o3d.geometry.PointCloud()
  point_cloud_o3d.points = o3d.utility.Vector3dVector(data[:,0:3])
  o3d.visualization.draw_geometries([point_cloud_o3d])

  origin_data = np.array(point_cloud_o3d.points, dtype=np.float32)
  v, u = pca(origin_data)
  print('the main orientation of this pointcloud is: ', u)

  # 绘制主方向轴
  line_set = o3d.geometry.LineSet()
  line_set.points = o3d.utility.Vector3dVector([np.mean(origin_data, axis=0), np.mean(origin_data, axis=0) + u])
  line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
  line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
  o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

  # 循环计算每个点的法向量，这里实际上就是用kd-tree算每个点的法向量
  number_nearest = 50   # 邻域数量
  pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
  normals = []
  for point in point_cloud_o3d.points:
      [k, idx, _] = pcd_tree.search_knn_vector_3d(point, knn=number_nearest)
      w, z = pca(origin_data[idx,:])
      normals.append(z)

  normals = np.array(normals, dtype=np.float64)
  
  # 对点云的法向量进行赋值
  point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
  o3d.visualization.draw_geometries([point_cloud_o3d])