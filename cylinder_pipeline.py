"""
Cylinder Detection Pipeline — ROS 2 Node
Assignment 1: Semantic Landmark Extraction and Classification

Pipeline:
  0. Box filter + voxel downsample
  1. Normal estimation (SVD) + RANSAC plane removal
  2. Euclidean clustering (BFS + cKDTree)
  3. Cylinder RANSAC per cluster
  4. HSV color classification (Red, Green, Blue)
"""

import collections
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray


# ===========================================================================
# CONFIGURATION
# ===========================================================================

class PipelineConfig:
    """All tunable parameters in one place."""

    def __init__(self):
        self.topic = '/oakd/points'

        # Box filter — keep only points in this XYZ region (meters)
        self.box_min = np.array([-1.0, -0.6,  0.2])
        self.box_max = np.array([ 1.0,  0.6,  2.0])

        # Voxel downsampling — 2 cm cubes
        self.voxel_size = 0.02

        # Plane RANSAC — floor/ceiling removal
        self.floor_dist    = 0.02
        self.target_normal = np.array([0.0, 1.0, 0.0])
        self.normal_thresh = 0.85
        self.num_plane_removals = 3

        # Euclidean clustering
        self.cluster_radius  = 0.06
        self.cluster_min_pts = 50
        self.cluster_max_pts = 5000

        # Cylinder RANSAC
        self.cyl_radius      = 0.055
        self.cyl_inlier_tol  = 0.015
        self.cyl_axis_thresh = 0.80
        self.cyl_min_inliers = 20
        self.max_cylinders   = 3


# ===========================================================================
# VISUALIZER
# ===========================================================================

class CylinderVisualizer:
    """Publishes detected cylinders as RViz MarkerArray."""

    def __init__(self, publisher):
        self.pub_markers = publisher

    def create_cylinder_marker(self, center, radius, rgb, marker_id, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD

        m.pose.position.x = float(center[0])
        m.pose.position.y = float(0.0)
        m.pose.position.z = float(center[2])

        # Identity quaternion — correct upright orientation for oakd optical frame
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = float(radius * 2.0)
        m.scale.y = float(radius * 2.0)
        m.scale.z = 0.4

        m.color.r = float(rgb[0])
        m.color.g = float(rgb[1])
        m.color.b = float(rgb[2])
        m.color.a = 0.8
        return m

    def publish_viz(self, cylinders, frame_id):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for i, (model, rgb, name) in enumerate(cylinders):
            center, _, radius = model
            marker = self.create_cylinder_marker(
                center, radius, rgb, 2000 + i, frame_id)
            ma.markers.append(marker)

        self.pub_markers.publish(ma)


# ===========================================================================
# PIPELINE
# ===========================================================================

class CylinderPipeline:
    """All geometric processing. No ROS dependencies — pure NumPy."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Color
    # ------------------------------------------------------------------

    def rgb_to_hsv(self, r, g, b):
        """Convert RGB in [0,1] to HSV. H in [0,360], S and V in [0,1]."""
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        if mx == mn:
            h = 0.0
        elif mx == r:
            h = (60.0 * ((g - b) / df) + 360.0) % 360.0
        elif mx == g:
            h = (60.0 * ((b - r) / df) + 120.0) % 360.0
        else:
            h = (60.0 * ((r - g) / df) + 240.0) % 360.0

        s = 0.0 if mx == 0.0 else (df / mx)
        v = mx
        return h, s, v

    def classify_color(self, h, s, v):
        """
        Map HSV to a semantic label.
        Guard against dark or unsaturated surfaces first.
        Red wraps around 0/360 so it needs two hue checks.
        """
        if s < 0.10 or v < 0.15:
            return "unknown", [0.5, 0.5, 0.5]

        if h < 15.0 or h > 325.0:
            return "red",   [1.0, 0.0, 0.0]
        elif 90.0 <= h <= 150.0:
            return "green", [0.0, 1.0, 0.0]
        elif 200.0 <= h <= 260.0:
            return "blue",  [0.0, 0.0, 1.0]
        else:
            return "unknown", [0.5, 0.5, 0.5]

    # ------------------------------------------------------------------
    # Neighbor search
    # ------------------------------------------------------------------

    def get_neighbors(self, pts, queries, k=15):
        """Return indices of k nearest neighbors for each query point."""
        if len(pts) < k:
            return None
        tree = cKDTree(pts)
        _, idxs = tree.query(queries, k=k)
        return idxs

    # ------------------------------------------------------------------
    # Task 0a — Box filter
    # ------------------------------------------------------------------

    def box_filter(self, pts, colors):
        """
        Keep points inside the configured XYZ bounding box.
        Single boolean mask — O(N), no loops.
        """
        cfg = self.cfg
        mask = (
            (pts[:, 0] >= cfg.box_min[0]) & (pts[:, 0] <= cfg.box_max[0]) &
            (pts[:, 1] >= cfg.box_min[1]) & (pts[:, 1] <= cfg.box_max[1]) &
            (pts[:, 2] >= cfg.box_min[2]) & (pts[:, 2] <= cfg.box_max[2])
        )
        return pts[mask], colors[mask]

    # ------------------------------------------------------------------
    # Task 0b — Voxel downsample
    # ------------------------------------------------------------------

    def downsample(self, pts, colors):
        """
        Keep one point per voxel_size cube.
        floor(p / voxel_size) maps each point to an integer voxel index.
        np.unique finds the first point in each unique voxel.
        """
        voxel_indices = np.floor(pts / self.cfg.voxel_size).astype(np.int32)
        _, first_idx = np.unique(voxel_indices, axis=0, return_index=True)
        return pts[first_idx], colors[first_idx]

    # ------------------------------------------------------------------
    # Task 1a — Normal estimation
    # ------------------------------------------------------------------

    def estimate_normals(self, pts, k=15):
        """
        Estimate surface normal at each point using SVD on its k-neighborhood.
        The last row of Vt from SVD is the direction of minimum variance
        which is perpendicular to the local surface — the normal.
        """
        n_pts = len(pts)
        normals = np.zeros((n_pts, 3), dtype=np.float64)

        neighbor_indices = self.get_neighbors(pts, pts, k=k)
        if neighbor_indices is None:
            return normals

        for i in range(n_pts):
            neighbors = pts[neighbor_indices[i]]
            centered  = neighbors - neighbors.mean(axis=0)
            _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
            normals[i] = Vt[-1]

        return normals

    # ------------------------------------------------------------------
    # Task 1b — Plane RANSAC
    # ------------------------------------------------------------------

    def find_plane_ransac(self, pts, iters=100):
        """
        Fit a horizontal plane using RANSAC.
        Sample 3 points → compute normal via cross product →
        check vertical alignment → count inliers by distance to plane.
        Returns the plane with the most inliers.
        """
        cfg   = self.cfg
        n_pts = len(pts)
        best_count = 0
        best_normal, best_d, best_mask = None, None, None

        for _ in range(iters):
            idx = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-6:
                continue

            normal = normal / norm_len

            # Reject planes not aligned with expected floor direction
            if abs(np.dot(normal, cfg.target_normal)) < cfg.normal_thresh:
                continue

            d     = -np.dot(normal, p1)
            dists = np.abs(pts @ normal + d)
            mask  = dists < cfg.floor_dist
            count = int(np.sum(mask))

            if count > best_count:
                best_count  = count
                best_normal = normal
                best_d      = d
                best_mask   = mask

        return best_normal, best_d, best_mask

    # ------------------------------------------------------------------
    # Task 2 — Euclidean clustering
    # ------------------------------------------------------------------

    def euclidean_clustering(self, pts):
        """
        Group points by proximity using BFS and a cKDTree radius search.
        Points within cluster_radius of each other belong to the same cluster.
        Filters clusters by min/max size.
        """
        cfg     = self.cfg
        n_pts   = len(pts)
        visited = np.zeros(n_pts, dtype=bool)
        tree    = cKDTree(pts)
        clusters = []

        for i in range(n_pts):
            if visited[i]:
                continue

            cluster_indices = []
            queue = collections.deque([i])
            visited[i] = True

            while queue:
                current = queue.popleft()
                cluster_indices.append(current)
                neighbors = tree.query_ball_point(
                    pts[current], r=cfg.cluster_radius)
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            size = len(cluster_indices)
            if cfg.cluster_min_pts <= size <= cfg.cluster_max_pts:
                clusters.append(np.array(cluster_indices, dtype=np.int32))

        return clusters

    # ------------------------------------------------------------------
    # Task 3 — Cylinder RANSAC
    # ------------------------------------------------------------------

    def find_single_cylinder(self, pts, normals, iters=300):
        """
        Fit a cylinder using RANSAC.
        Sample 2 points and their normals → axis = cross(n1, n2) →
        check vertical alignment → count inliers by perpendicular
        distance to axis. Returns best (center, axis, radius, mask).
        """
        cfg   = self.cfg
        n_pts = len(pts)
        if n_pts < 20:
            return None

        best_count  = 0
        best_result = None
        vertical    = cfg.target_normal

        for _ in range(iters):
            idx = np.random.choice(n_pts, 2, replace=False)
            p1, p2 = pts[idx[0]], pts[idx[1]]
            n1, n2 = normals[idx[0]], normals[idx[1]]

            axis     = np.cross(n1, n2)
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-6:
                continue

            axis = axis / axis_len
            if np.dot(axis, vertical) < 0:
                axis = -axis

            if abs(np.dot(axis, vertical)) < cfg.cyl_axis_thresh:
                continue

            # Perpendicular distance from every point to the axis through p1
            V          = pts - p1
            proj_len   = V @ axis
            proj_vecs  = np.outer(proj_len, axis)
            perp_vecs  = V - proj_vecs
            perp_dists = np.linalg.norm(perp_vecs, axis=1)

            inlier_mask = np.abs(perp_dists - cfg.cyl_radius) < cfg.cyl_inlier_tol
            count = int(np.sum(inlier_mask))

            if count > best_count:
                best_count    = count
                inlier_pts    = pts[inlier_mask]
                center        = inlier_pts.mean(axis=0)
                actual_radius = float(perp_dists[inlier_mask].mean())
                best_result   = (center, axis, actual_radius, inlier_mask)

        if best_count < cfg.cyl_min_inliers:
            return None

        return best_result


# ===========================================================================
# ROS 2 NODE
# ===========================================================================

class CylinderProcessorNode(Node):

    def __init__(self):
        super().__init__('cylinder_processor_node')
        self.cfg      = PipelineConfig()
        self.pipeline = CylinderPipeline(self.cfg)

        # Debug publishers — add these in RViz to inspect each stage
        self.pub_stage0 = self.create_publisher(
            PointCloud2, 'pipeline/stage0_box', 10)
        self.pub_stage1 = self.create_publisher(
            PointCloud2, 'pipeline/stage1_no_planes', 10)
        self.pub_stage3 = self.create_publisher(
            PointCloud2, 'pipeline/stage3_candidates', 10)

        marker_pub = self.create_publisher(MarkerArray, 'viz/detections', 10)
        self.visualizer = CylinderVisualizer(marker_pub)

        self.sub = self.create_subscription(
            PointCloud2, self.cfg.topic, self.listener_callback, 10)

        self.get_logger().info("CylinderProcessorNode ready.")

    def _decode_colors(self, packed_col):
        """
        Unpack float32-encoded RGB from the point cloud.
        The OAK-D packs r, g, b as 3 bytes inside a uint32 cast to float32.
        RGB is at byte offset 16 = float32 column 4 for this camera.
        """
        rgb_u32 = packed_col.view(np.uint32)
        r = ((rgb_u32 >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_u32 >>  8) & 0xFF).astype(np.float32) / 255.0
        b = ( rgb_u32        & 0xFF).astype(np.float32) / 255.0
        return np.stack([r, g, b], axis=1)

    def numpy_to_pc2_rgb(self, pts, colors, frame_id):
        """Pack (N,3) XYZ and (N,3) RGB into a PointCloud2 message."""
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width  = len(pts)
        msg.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 16
        msg.row_step     = 16 * len(pts)
        msg.is_dense     = True

        c = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint32)
        rgb_packed = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)
        data = np.hstack([pts.astype(np.float32), rgb_packed.reshape(-1, 1)])
        msg.data = data.tobytes()
        return msg

    def _publish_colored_clusters(self, clusters, work_pts, frame_id):
        """Color each cluster differently and publish for RViz inspection."""
        palette = np.array([
            [1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0],
            [1.0, 1.0, 0.2], [1.0, 0.2, 1.0], [0.2, 1.0, 1.0],
        ])
        all_pts, all_colors = [], []
        for i, idx_arr in enumerate(clusters):
            color = palette[i % len(palette)]
            all_pts.append(work_pts[idx_arr])
            all_colors.append(np.tile(color, (len(idx_arr), 1)))
        if all_pts:
            self.pub_stage3.publish(self.numpy_to_pc2_rgb(
                np.vstack(all_pts), np.vstack(all_colors), frame_id))

    def listener_callback(self, msg):
        """
        Main pipeline: PointCloud2 → cylinder detections with color labels.
        """
        frame_id = msg.header.frame_id

        # Parse raw data
        stride   = msg.point_step // 4
        raw_data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, stride)
        pts      = raw_data[:, :3].copy()

        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite_mask]

        # RGB is at byte offset 16 = column 4 (confirmed for this camera)
        raw_colors = self._decode_colors(raw_data[finite_mask, 4].copy())

        if len(pts) < 100:
            return

        # Stage 0a — box filter
        pts_box, colors_box = self.pipeline.box_filter(pts, raw_colors)
        if len(pts_box) < 100:
            self.get_logger().warn(
                f"Box filter left {len(pts_box)} points. Check box bounds.")
            return

        # Stage 0b — voxel downsample
        pts_v, colors_v = self.pipeline.downsample(pts_box, colors_box)
        if len(pts_v) < 50:
            self.get_logger().warn("Downsample left < 50 points.")
            return

        self.pub_stage0.publish(
            self.numpy_to_pc2_rgb(pts_v, colors_v, frame_id))

        # Stage 1a — normal estimation
        normals = self.pipeline.estimate_normals(pts_v, k=15)

        # Stage 1b — remove dominant planes (floor, table, ceiling)
        work_pts     = pts_v.copy()
        work_colors  = colors_v.copy()
        work_normals = normals.copy()

        for _ in range(self.cfg.num_plane_removals):
            if len(work_pts) < 50:
                break
            _, _, inlier_mask = self.pipeline.find_plane_ransac(work_pts)
            if inlier_mask is None or int(np.sum(inlier_mask)) < 20:
                break
            keep         = ~inlier_mask
            work_pts     = work_pts[keep]
            work_colors  = work_colors[keep]
            work_normals = work_normals[keep]

        self.pub_stage1.publish(
            self.numpy_to_pc2_rgb(work_pts, work_colors, frame_id))

        if len(work_pts) < 20:
            return

        # Stage 2 — euclidean clustering
        clusters = self.pipeline.euclidean_clustering(work_pts)
        if not clusters:
            self.visualizer.publish_viz([], frame_id)
            return

        self._publish_colored_clusters(clusters, work_pts, frame_id)

        # Stage 3 + 4 — cylinder RANSAC and color classification
        detected_cylinders = []

        for cluster_idx_array in clusters:
            if len(detected_cylinders) >= self.cfg.max_cylinders:
                break

            cluster_pts     = work_pts[cluster_idx_array]
            cluster_colors  = work_colors[cluster_idx_array]
            cluster_normals = work_normals[cluster_idx_array]

            result = self.pipeline.find_single_cylinder(
                cluster_pts, cluster_normals)
            if result is None:
                continue

            center, axis, radius, inlier_mask = result

            # Classify color using only the cylinder surface inlier points
            avg_rgb = work_colors[cluster_idx_array][inlier_mask].mean(axis=0)
            h, s, v = self.pipeline.rgb_to_hsv(
                float(avg_rgb[0]), float(avg_rgb[1]), float(avg_rgb[2]))
            label, display_rgb = self.pipeline.classify_color(h, s, v)

            model = (center, axis, radius)
            detected_cylinders.append((model, display_rgb, label))

            self.get_logger().info(
                f"Cylinder: label={label}  "
                f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})  "
                f"r={radius:.3f}m  inliers={int(np.sum(inlier_mask))}"
            )

        self.visualizer.publish_viz(detected_cylinders, frame_id)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    rclpy.init()
    node = CylinderProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()