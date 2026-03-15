import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import string
import json

# Define number of ticks (adjustable, max 26 due to alphabet size)
n_ticks = 15  # Change this value as needed (e.g., 5, 15, 20), <= 26
if n_ticks > 26:
    print("n_ticks exceeds the number of available letters. Setting to 26.")
    n_ticks = 26

# Load point cloud and perform PCA
name = "peg"
file_path = f'{name}_pointcloud_full.npy'
points = np.load(file_path)
mean = np.mean(points, axis=0)
centered_points = points - mean
pca = PCA(n_components=2)
pca.fit(centered_points)
components = pca.components_

# Print endpoints for each principal axis
print("Coordinates of the Two Points for Each Principal Axis:")
for i in range(2):
    component = components[i]
    projections = centered_points.dot(component)
    t_min = np.min(projections)
    t_max = np.max(projections)
    p1 = mean + t_min * component
    p2 = mean + t_max * component
    print(f"\nPrincipal Component {i + 1} (PC{i + 1}):")
    print(f"  Endpoint 1: ({p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f})")
    print(f"  Endpoint 2: ({p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f})")

# Compute variance explained by each principal component
variance_pc1 = pca.explained_variance_ratio_[0] * 100
variance_pc2 = pca.explained_variance_ratio_[1] * 100

# Initial 3D Plot (No Axis Labels and No Title)
fig_3d_initial = plt.figure(figsize=(8, 6))
ax3d_initial = fig_3d_initial.add_subplot(111, projection='3d')
ax3d_initial.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.5, s=20, color='b', label='Point Cloud')
colors = ['r', 'g']
markers = ['o', 's']
for i in range(2):
    component = components[i]
    projections = centered_points.dot(component)
    t_min = np.min(projections)
    t_max = np.max(projections)
    p1 = mean + t_min * component
    p2 = mean + t_max * component
    ax3d_initial.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      color=colors[i], linewidth=2, label=f'PC{i + 1} Axis')
    ax3d_initial.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                         color=colors[i], marker=markers[i], s=100)
ax3d_initial.legend()
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
x_range = x_max - x_min
y_range = y_max - y_min
max_range = x_range if x_range > y_range else y_range
max_range = max_range / 2
ax3d_initial.set_xlim(np.mean(points[:, 0]) - max_range, np.mean(points[:, 0]) + max_range)
ax3d_initial.set_ylim(np.mean(points[:, 1]) - max_range, np.mean(points[:, 1]) + max_range)
ax3d_initial.set_zlim(np.mean(points[:, 2]) - max_range, np.mean(points[:, 2]) + max_range)
plt.show()

# Initial 2D PCA Projection Plot (No Axis Labels and No Title)
transformed_points = pca.transform(centered_points)
fig_2d_initial = plt.figure(figsize=(8, 6))
ax2d_initial = fig_2d_initial.add_subplot(111)
ax2d_initial.scatter(transformed_points[:, 0], transformed_points[:, 1], alpha=0.5, s=20, color='b')

# Define tick labels based on n_ticks
y_tick_labels = list(string.ascii_lowercase)[:n_ticks]
x_tick_labels = [str(i) for i in range(1, n_ticks + 1)]

# Calculate tick positions to center the plot
x_min, x_max = transformed_points[:, 0].min(), transformed_points[:, 0].max()
y_min, y_max = transformed_points[:, 1].min(), transformed_points[:, 1].max()
x_range = x_max - x_min
y_range = y_max - y_min
max_range = x_range if x_range > y_range else y_range
x_min = np.average(transformed_points[:, 0]) - max_range / 2
x_max = np.average(transformed_points[:, 0]) + max_range / 2
y_min = np.average(transformed_points[:, 1]) - max_range / 2
y_max = np.average(transformed_points[:, 1]) + max_range / 2

x_ticks = np.linspace(x_min, x_max, n_ticks)
y_ticks = np.linspace(y_min, y_max, n_ticks)
ax2d_initial.set_xticks(x_ticks)
ax2d_initial.set_yticks(y_ticks)
ax2d_initial.set_xticklabels(x_tick_labels)
ax2d_initial.set_yticklabels(y_tick_labels)

ax2d_initial.grid(True, which='both', linestyle='-', linewidth=1.5)
ax2d_initial.set_aspect('equal')
plt.show()

# Prompt User for Coordinate
print("Please examine the initial plots to choose a coordinate.")
coord_input = input(f"Enter an axis coordinate (e.g., '{y_tick_labels[0]}1' to '{y_tick_labels[-1]}{n_ticks}'): ")

# Parse the input coordinate
if len(coord_input) < 2:
    print("Invalid coordinate label. Expected format like 'a1'.")
else:
    letter_part = coord_input[0].lower()
    number_part = coord_input[1:]
    if letter_part not in y_tick_labels:
        print(f"Invalid letter. Please use a letter between {y_tick_labels[0]} and {y_tick_labels[-1]}.")
    else:
        try:
            tick_number = int(number_part)
            if tick_number < 1 or tick_number > n_ticks:
                print(f"Tick number out of range. Please enter a number between 1 and {n_ticks}.")
            else:
                x_idx = tick_number - 1
                y_idx = y_tick_labels.index(letter_part)
                x_val = x_ticks[x_idx]
                y_val = y_ticks[y_idx]
                pca_coord = np.array([x_val, y_val])
                original_centered = pca.inverse_transform(pca_coord)
                original_coord = original_centered + mean

                # Updated 2D Plot
                fig_2d_updated = plt.figure(figsize=(8, 6))
                ax2d_updated = fig_2d_updated.add_subplot(111)
                ax2d_updated.scatter(transformed_points[:, 0], transformed_points[:, 1], alpha=0.5, s=20, color='b')
                ax2d_updated.scatter([x_val], [y_val], color='k', marker='*', s=200, label=f'Input {coord_input}')
                ax2d_updated.set_title('Updated 2D Projection with Selected Point')
                ax2d_updated.set_xticks(x_ticks)
                ax2d_updated.set_yticks(y_ticks)
                ax2d_updated.set_xticklabels(x_tick_labels)
                ax2d_updated.set_yticklabels(y_tick_labels)
                ax2d_updated.grid(True, which='both', linestyle='-', linewidth=1.5)
                ax2d_updated.set_aspect('equal')
                ax2d_updated.legend()
                plt.show()

                # Updated 3D Plot
                fig_3d_updated = plt.figure(figsize=(8, 6))
                ax3d_updated = fig_3d_updated.add_subplot(111, projection='3d')
                ax3d_updated.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.1, s=20, color='b', label='Point Cloud')
                for i in range(2):
                    component = components[i]
                    projections = centered_points.dot(component)
                    t_min = np.min(projections)
                    t_max = np.max(projections)
                    p1 = mean + t_min * component
                    p2 = mean + t_max * component
                    ax3d_updated.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                      color=colors[i], linewidth=2, label=f'PC{i + 1} Axis')
                    ax3d_updated.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                         color=colors[i], marker=markers[i], s=100)
                ax3d_updated.scatter([original_coord[0]], [original_coord[1]], [original_coord[2]],
                                     color='k', marker='*', s=200, label=f'Input {coord_input}')
                ax3d_updated.set_xlabel('X')
                ax3d_updated.set_ylabel('Y')
                ax3d_updated.set_zlabel('Z')
                ax3d_updated.legend()
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                y_min, y_max = points[:, 1].min(), points[:, 1].max()
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = x_range if x_range > y_range else y_range
                max_range = max_range / 2
                ax3d_updated.set_xlim(np.mean(points[:, 0]) - max_range, np.mean(points[:, 0]) + max_range)
                ax3d_updated.set_ylim(np.mean(points[:, 1]) - max_range, np.mean(points[:, 1]) + max_range)
                ax3d_updated.set_zlim(np.mean(points[:, 2]) - max_range, np.mean(points[:, 2]) + max_range)
                plt.show()

                # Print coordinates
                print(f"Input coordinate {coord_input} in PCA space: ({x_val:.4f}, {y_val:.4f})")
                print(f"Approximate original coordinate (3D): ({original_coord[0]:.4f}, {original_coord[1]:.4f}, {original_coord[2]:.4f})")

                # Optional JSON save
                save_flag = input("Save point as JSON? 1=True: ")
                if save_flag == "1":
                    original_coord_savejson = original_coord.tolist()
                    with open(f"{name}_grasp.json", "w") as f:
                        json.dump(original_coord_savejson, f)
                    print("Point saved as JSON.")
        except ValueError:
            print(f"Invalid number. Please use a number between 1 and {n_ticks}.")