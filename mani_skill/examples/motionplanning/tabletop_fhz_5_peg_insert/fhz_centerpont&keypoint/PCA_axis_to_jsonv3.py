import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load the point cloud
file_path = f'peg_pointcloud_full.npy'  # Replace with your actual file path
points = np.load(file_path)

# Compute the mean and center the points
mean = np.mean(points, axis=0)
centered_points = points - mean

# Apply PCA
pca = PCA(n_components=2)
pca.fit(centered_points)
components = pca.components_

# Prepare data for PCA axes and center point as lists
pca_data = {
    "center_point": [float(mean[0]), float(mean[1]), float(mean[2])],
    "principal_axes": []
}

# Add the two principal components and their endpoints
for i in range(2):
    component = components[i]
    projections = centered_points.dot(component)
    t_min = np.min(projections)
    t_max = np.max(projections)
    p1 = mean + t_min * component
    p2 = mean + t_max * component

    axis_data = {
        "component": [float(component[0]), float(component[1]), float(component[2])],
        "endpoint_1": [float(p1[0]), float(p1[1]), float(p1[2])],
        "endpoint_2": [float(p2[0]), float(p2[1]), float(p2[2])]
    }
    pca_data["principal_axes"].append(axis_data)

# Save the data to a JSON file
json_file_path = f'peg_PCA.json'  # Replace with your desired file path
with open(json_file_path, 'w') as json_file:
    json.dump(pca_data, json_file, indent=4)

print(f"PCA data saved to {json_file_path}")

# Visualize
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.1, s=1, color='b', label='Point Cloud')
colors = ['r', 'g']
markers = ['o', 's']
for i in range(2):
    component = components[i]
    projections = centered_points.dot(component)
    t_min = np.min(projections)
    t_max = np.max(projections)
    p1 = mean + t_min * component
    p2 = mean + t_max * component
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=colors[i], linewidth=2, label=f'PC{i + 1} Axis')
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
               color=colors[i], marker=markers[i], s=100)

# 设置 X 和 Y 轴的显示范围 保持一致 fhz################################################################
x_range = points[:, 0].max() - points[:, 0].min()
y_range = points[:, 1].max() - points[:, 1].min()
range=x_range if x_range > y_range else y_range
range=range/2
ax.set_xlim(np.mean(points[:, 0])-range , np.mean(points[:, 0])+range )
ax.set_ylim(np.mean(points[:, 1])-range , np.mean(points[:, 1])+range )
ax.set_zlim(np.mean(points[:, 2])-range , np.mean(points[:, 2])+range )
# 设置 X 和 Y 轴的显示范围 保持一致 fhz################################################################

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('3D Point Cloud with PCA Axes')
ax.set_box_aspect([1, 1, 1])
plt.show()