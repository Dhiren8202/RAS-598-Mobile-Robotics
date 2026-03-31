import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray, Marker
from example_interfaces.srv import Trigger
import math
import threading
from scipy.ndimage import binary_dilation
import heapq
import numpy as np


class PathPlanner(Node):

    def __init__(self):
        super().__init__("path_planner")

        # -- map data storage --
        self.map_data = None
        self.map_info = None
        self.grid = None

        # -- robot current position --
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # -- mission coordinates --
        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0

        # -- path storage --
        self.raw_path = []
        self.pruned_path = []
        self.current_target_idx = 0
        self.mission_active = False

        # -- subscribers --
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, map_qos)
        self.gt_sub = self.create_subscription(Odometry, "/ground_truth", self.pose_callback, 10)

        # -- publishers --
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/planner_markers", 10)

        # -- service client for getting the task --
        self.task_client = self.create_client(Trigger, "get_task")

        self.get_logger().info("PathPlanner node started, waiting for map...")

    def map_callback(self, msg):
        if self.map_data is not None:
            return
        self.map_info = msg.info
        self.map_data = msg
        self.get_logger().info("Map received! Requesting task from grading scout...")
        self.request_task()

    def request_task(self):
        while not self.task_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_task service...")
        req = Trigger.Request()
        future = self.task_client.call_async(req)
        future.add_done_callback(self.task_response_callback)

    def task_response_callback(self, future):
        response = future.result()
        if response.success:
            coords = response.message.split(",")
            self.start_x = float(coords[0])
            self.start_y = float(coords[1])
            self.goal_x  = float(coords[2])
            self.goal_y  = float(coords[3])
            self.get_logger().info(f"Task received! Start({self.start_x},{self.start_y}) Goal({self.goal_x},{self.goal_y})")
            thread = threading.Thread(target=self.run_planner)
            thread.daemon = True
            thread.start()
        else:
            self.get_logger().error("Failed to get task from grading scout!")

    def pose_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def world_to_grid(self, wx, wy):
        col = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        row = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        return row, col

    def grid_to_world(self, row, col):
        wx = self.map_info.origin.position.x + col * self.map_info.resolution
        wy = self.map_info.origin.position.y + row * self.map_info.resolution
        return wx, wy

    def build_grid(self):
        width  = self.map_info.width
        height = self.map_info.height
        raw = np.array(self.map_data.data).reshape((height, width))

        # cells with value > 65 are obstacles, rest is free
        binary = np.zeros((height, width), dtype=np.uint8)
        binary[raw > 65] = 1

        # inflate obstacles so robot keeps a safe distance from walls
        inflation_radius = 8
        struct = np.ones((2*inflation_radius+1, 2*inflation_radius+1), dtype=bool)
        inflated = binary_dilation(binary, structure=struct).astype(np.uint8)

        self.grid = inflated
        self.get_logger().info(f"Grid built: {width}x{height}, inflation radius={inflation_radius}")

    def run_astar(self, start_world, goal_world):
        sr, sc = self.world_to_grid(start_world[0], start_world[1])
        gr, gc = self.world_to_grid(goal_world[0],  goal_world[1])

        self.get_logger().info(f"A* from grid({sr},{sc}) to grid({gr},{gc})")

        height, width = self.grid.shape

        def heuristic(r, c):
            return math.sqrt((r - gr)**2 + (c - gc)**2)

        open_heap = []
        heapq.heappush(open_heap, (heuristic(sr, sc), 0.0, sr, sc))

        came_from = {}
        g_scores  = {(sr, sc): 0.0}

        # 8 directional moves
        directions = [
            (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)

            if (r, c) == (gr, gc):
                path = []
                node = (gr, gc)
                while node in came_from:
                    wr, wc = self.grid_to_world(node[0], node[1])
                    path.append((wr, wc))
                    node = came_from[node]
                wr, wc = self.grid_to_world(sr, sc)
                path.append((wr, wc))
                path.reverse()
                self.get_logger().info(f"A* found path with {len(path)} waypoints")
                return path

            for dr, dc, cost in directions:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                if self.grid[nr][nc] == 1:
                    continue
                new_g = g + cost
                if (nr, nc) not in g_scores or new_g < g_scores[(nr, nc)]:
                    g_scores[(nr, nc)] = new_g
                    f_new = new_g + heuristic(nr, nc)
                    heapq.heappush(open_heap, (f_new, new_g, nr, nc))
                    came_from[(nr, nc)] = (r, c)

        self.get_logger().error("A* could not find a path!")
        return []

    def check_los(self, wx1, wy1, wx2, wy2):
        r1, c1 = self.world_to_grid(wx1, wy1)
        r2, c2 = self.world_to_grid(wx2, wy2)

        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        r, c = r1, c1
        step_r = 1 if r2 > r1 else -1
        step_c = 1 if c2 > c1 else -1
        error = dr - dc

        height, width = self.grid.shape

        while True:
            if r < 0 or r >= height or c < 0 or c >= width:
                return False
            if self.grid[r][c] == 1:
                return False
            if r == r2 and c == c2:
                break
            double_error = 2 * error
            if double_error > -dc:
                error -= dc
                r += step_r
            if double_error < dr:
                error += dr
                c += step_c

        return True

    def prune_path(self, path):
        if len(path) < 3:
            return path

        pruned = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            furthest = current_idx + 1
            for look_ahead in range(current_idx + 2, len(path)):
                wx1, wy1 = pruned[-1]
                wx2, wy2 = path[look_ahead]
                if self.check_los(wx1, wy1, wx2, wy2):
                    furthest = look_ahead
                else:
                    break
            pruned.append(path[furthest])
            current_idx = furthest

        self.get_logger().info(f"Path pruned: {len(path)} -> {len(pruned)} waypoints")
        return pruned

    def publish_paths(self):
        marker_array = MarkerArray()
        marker_id = 0

        # green markers for raw A* path
        for (wx, wy) in self.raw_path:
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "raw_path"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            marker_array.markers.append(m)
            marker_id += 1

        # blue markers for pruned path
        for (wx, wy) in self.pruned_path:
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "pruned_path"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 1.0
            marker_array.markers.append(m)
            marker_id += 1

        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(self.raw_path)} green and {len(self.pruned_path)} blue markers")

    def start_controller(self):
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.mission_active = True
        self.current_target_idx = 0
        self.get_logger().info("Controller started!")

    def control_loop(self):
        if not self.mission_active:
            return
        if self.current_target_idx >= len(self.pruned_path):
            self.stop_robot()
            self.mission_active = False
            self.get_logger().info("Mission complete! Robot reached the goal.")
            return

        target_x, target_y = self.pruned_path[self.current_target_idx]

        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)

        angle_error = angle_to_target - self.robot_yaw
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd = Twist()

        if distance < 0.3:
            self.current_target_idx += 1
            self.get_logger().info(f"Waypoint {self.current_target_idx} reached, {len(self.pruned_path) - self.current_target_idx} remaining")
            return

        if abs(angle_error) > 0.3:
            cmd.angular.z = 1.2 * angle_error
            cmd.angular.z = max(-1.5, min(1.5, cmd.angular.z))
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = min(0.5, 0.4 * distance)
            cmd.angular.z = 1.0 * angle_error

        self.vel_pub.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.vel_pub.publish(cmd)

    def run_planner(self):
        self.build_grid()
        self.raw_path = self.run_astar(
            (self.start_x, self.start_y),
            (self.goal_x,  self.goal_y)
        )
        if self.raw_path:
            self.pruned_path = self.prune_path(self.raw_path)
            self.get_logger().info(f"Planning complete! Raw={len(self.raw_path)} Pruned={len(self.pruned_path)} waypoints.")
            self.publish_paths()
            self.start_controller()
        else:
            self.get_logger().error("Planning failed!")


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
