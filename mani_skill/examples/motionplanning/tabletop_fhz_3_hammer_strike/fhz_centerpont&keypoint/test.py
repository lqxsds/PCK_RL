import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 .npy 文件
data = np.load('hammer_pointcloud_full.npy')  # 替换为你的文件路径
x, y, z = data[:, 0], data[:, 1], data[:, 2]  # 假设数据是 (N, 3)

# 创建 3D 散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1)  # s 控制点的大小
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_box_aspect([1,1,1])  # 使三个轴的比例相同
plt.show()