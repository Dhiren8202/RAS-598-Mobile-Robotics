# Assignment 1: Semantic Landmark Extraction and Classification

**Dhiren Makwana:** 1233765119  
**Course:** RAS 598 — Mobile Robotics  
**Robot:** OAK-D Camera (Simulation via ROS 2 Bags)  
**ROS Distro:** Jazzy  

---

## Overview

This assignment implements a full perception pipeline that processes raw RGB-D point clouds from an OAK-D camera and detects, localizes, and semantically labels cylinders in the environment. The pipeline runs entirely on recorded ROS 2 bags without a physical robot.

The node subscribes to `/oakd/points` (sensor_msgs/PointCloud2), processes each frame through a multi-stage pipeline, and publishes colored cylinder markers to RViz along with intermediate point clouds for debugging.

---

## Pipeline Summary

```
Raw PointCloud2
      │
      ▼
 [Box Filter]           — Remove points outside XYZ region of interest
      │
      ▼
 [Voxel Downsample]     — Keep one point per 2cm voxel (np.unique)
      │
      ▼
 [Normal Estimation]    — SVD of k-neighbor patch → surface normal per point
      │
      ▼
 [Plane RANSAC × 3]     — Remove floor, table, ceiling iteratively
      │
      ▼
 [Euclidean Clustering] — BFS + cKDTree radius search → separate objects
      │
      ▼
 [Cylinder RANSAC]      — cross(n1, n2) → axis → perpendicular distance inliers
      │
      ▼
 [HSV Classification]   — mean inlier RGB → HSV → hue range → Red/Green/Blue
      │
      ▼
 [RViz MarkerArray]     — Colored cylinder markers at detected positions
```

---

## File Structure

```
assignment1/
├── cylinder_pipeline.py   — Main ROS 2 node (full pipeline implementation)
├── cylinders.rviz         — RViz configuration file
└── README.md              — This file
```

---

## Dependencies

- ROS 2 Jazzy
- Python 3
- NumPy
- SciPy (`cKDTree` only — for neighbor search)
- `sensor_msgs`, `visualization_msgs`, `geometry_msgs` (standard ROS 2 packages)

---

## Algorithm Details

### Task 0 — Preprocessing

**Box Filter:** A single boolean mask keeps only points within the configured XYZ bounds. This is O(N) with no loops — one vectorized NumPy operation.

**Voxel Downsampling:** Each point's coordinates are divided by `voxel_size` and floored to produce integer voxel indices. `np.unique(axis=0)` with `return_index=True` finds the first point in each unique voxel, reducing the cloud from ~100k points to ~5k.

### Task 1 — Plane Segmentation (RANSAC)

Each iteration samples 3 random points, computes the plane normal via cross product, checks alignment with the expected floor normal using a dot product threshold, and counts inliers as points within `floor_dist` of the plane. The best plane is removed and the process repeats up to `num_plane_removals` times to strip the floor, table, and ceiling.

**Normal Estimation:** For each point, the k nearest neighbors are found with cKDTree. SVD is applied to the centered (k × 3) neighborhood matrix. The last row of Vt — the direction of minimum variance — is the surface normal.

### Task 2 — Euclidean Clustering

BFS with a cKDTree radius query groups spatially connected points into clusters. Each unvisited point starts a new cluster. All neighbors within `cluster_radius` are added to the queue and marked visited. Clusters outside `[cluster_min_pts, cluster_max_pts]` are discarded.

### Task 3 — Cylinder Detection (RANSAC)

Two points and their surface normals are sampled. On a cylinder surface, normals point radially outward — perpendicular to the axis. The cross product of two radial normals gives the axis direction. The axis is verified to be approximately vertical. Inliers are counted as points whose perpendicular distance to the axis is within `cyl_inlier_tol` of `cyl_radius`. All inlier distance calculations are vectorized with NumPy.

### Task 4 — Semantic Labeling

The average RGB color of cylinder inlier points is converted to HSV. Hue ranges identify the color:

| Color | Hue Range | Notes |
|---|---|---|
| Red | h < 15° or h > 325° | Wraps around 0° on the hue circle |
| Green | 90° ≤ h ≤ 150° | Centered on 120° |
| Blue | 200° ≤ h ≤ 260° | Centered on 240° |

A saturation guard (`s < 0.10`) and value guard (`v < 0.15`) prevent misclassifying grey or dark surfaces.

---

## Configuration Parameters

All parameters are in `PipelineConfig` at the top of `cylinder_pipeline.py`:

| Parameter | Value | Description |
|---|---|---|
| `voxel_size` | 0.02 | Voxel grid cell size in meters |
| `floor_dist` | 0.02 | Plane inlier threshold in meters |
| `normal_thresh` | 0.85 | Min dot product with vertical for floor planes |
| `num_plane_removals` | 3 | Number of planes to strip per frame |
| `cluster_radius` | 0.06 | BFS neighbor search radius in meters |
| `cluster_min_pts` | 50 | Minimum points per valid cluster |
| `cyl_radius` | 0.055 | Expected cylinder radius in meters |
| `cyl_inlier_tol` | 0.015 | Tolerance around expected radius |
| `cyl_axis_thresh` | 0.80 | Min dot product with vertical for cylinder axis |
| `cyl_min_inliers` | 20 | Minimum inliers to accept a detection |
| `max_cylinders` | 3 | Maximum cylinders to detect per frame |

---

## Library Compliance

- **Allowed and used:** NumPy, `scipy.spatial.cKDTree` (neighbor search only), standard ROS 2 packages
- **Not used:** Open3D, PCL, scikit-learn, SciPy RANSAC, SciPy clustering
- All geometric logic (RANSAC, normal estimation, clustering, distance calculations) is implemented with pure NumPy

---
## Debugging and Tuning
 
Getting the pipeline working correctly required several rounds of debugging and parameter tuning. Each issue was diagnosed by printing intermediate values and inspecting intermediate point clouds in RViz at each pipeline stage.
 
 
### 1. Cylinder Marker Orientation
 
The initial marker orientation used a 90° rotation around the X axis (`orientation.x = 0.7071`) which laid the cylinder on its side in RViz. After testing all quaternion combinations, the identity quaternion (`orientation.w = 1.0`, all others 0) produced the correct upright orientation in the `oakd_rgb_camera_optical_frame`. This is because the RViz cylinder marker's default Z axis aligns correctly with vertical in this camera frame without any rotation.
 
---
 
### 2. Cluster Radius Tuning for rgbd_bag_2
 
With three cylinders in the scene, the cluster radius needed careful tuning:
 
| Value | Problem |
|---|---|
| `0.08` | Too large — merged nearby cylinders into one blob (sizes: [959, 191]) |
| `0.05` | Too small — split each cylinder into fragments (6 clusters instead of 3) |
| `0.06` | Correct — consistently produced 3–4 distinct clusters |
 
The right value was found by printing cluster sizes and centers at each frame and narrowing in between the two failure modes.
 
---
 
### 3. HSV Threshold Tuning for Red Cylinder
 
The red cylinder in `rgbd_bag_2` was being classified as "unknown" instead of red. Printing the actual HSV values revealed two problems:
 
- **Saturation was 0.19** — just below the original guard threshold of `s < 0.20`, so the cylinder was immediately rejected before even checking hue
- **Hue was 337°** — outside the original red range of `h > 345°`
 
Two fixes were applied:
- Lowered saturation guard from `s < 0.20` to `s < 0.10` to allow washed-out colors through
- Widened the red upper boundary from `h > 345°` to `h > 325°` to capture hues in the magenta-red range
 
After these changes the red cylinder was correctly detected and labeled across all frames.
 

 
---
## Results

### rgbd_bag_0 — Single Green Cylinder

Static scene with one green cylinder. The pipeline correctly detects and labels it across all frames.



<img width="1194" height="883" alt="Screenshot 2026-03-20 011700" src="https://github.com/user-attachments/assets/f629b218-3216-436b-a2d8-3e3e3750a4af" />

<img width="1916" height="992" alt="Screenshot 2026-03-20 011810" src="https://github.com/user-attachments/assets/55b69269-1f99-46a0-a07b-5cbb8672e49e" />


---

### rgbd_bag_1 — Robot Moving Around Cylinders

The robot moves around the scene, changing viewpoint continuously. The pipeline detects and tracks the cylinder reliably across all frames despite changing depth and viewing angle.

<img width="1200" height="873" alt="Screenshot 2026-03-20 011944" src="https://github.com/user-attachments/assets/2e3993ab-dcaa-49dc-8b3d-bbe45bb6e536" />

---<img width="1919" height="997" alt="Screenshot 2026-03-20 012009" src="https://github.com/user-attachments/assets/aba7229c-e9c9-4ff6-8b84-801392e32c29" />


### rgbd_bag_2 — Three Cylinders (Red, Green, Blue)

Three cylinders at different positions in the scene. All three are detected and correctly labeled by color in the same frame.
<img width="1199" height="890" alt="Screenshot 2026-03-20 012049" src="https://github.com/user-attachments/assets/0e41b69b-1619-40ff-af72-df8d430b5260" />

<img width="1917" height="1002" alt="Screenshot 2026-03-20 012121" src="https://github.com/user-attachments/assets/1f81ded4-c1d5-41fb-9c4c-dc7a12ecbcc6" />


---

## Submission

- **Repository:** https://github.com/Dhiren8202/RAS-598-Mobile-Robotics
- **Commit hash:** 6d9a974
