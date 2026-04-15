import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray, Marker
from example_interfaces.srv import Trigger
from scipy.ndimage import binary_dilation
import math
import heapq
import numpy as np
import threading


class PathPlanner(Node):

    def __init__(self):
        super().__init__("path_planner")

        # --- map storage ---
        self.map_data = None        # raw map message, None until map arrives
        self.map_info = None        # map metadata: resolution, origin, width, height
        self.grid     = None        # processed 2D binary grid (0=free, 1=obstacle)

        # --- robot position updated every time /ground_truth publishes ---
        self.robot_x   = 0.0       # robot x position in meters
        self.robot_y   = 0.0       # robot y position in meters
        self.robot_yaw = 0.0       # robot heading angle in radians

        # --- mission start and goal coordinates (received from grading scout) ---
        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_x  = 0.0
        self.goal_y  = 0.0

        # --- path storage ---
        self.raw_path    = []       # full A* path as list of (x,y) world coords (shown as green in RViz)
        self.pruned_path = []       # simplified path after line-of-sight pruning (shown as blue in RViz)

        # --- controller state ---
        self.current_target_idx = 0     # index of the waypoint we are currently heading towards
        self.mission_active     = False # becomes True once controller starts, False when goal is reached

        # --- map subscriber uses TRANSIENT_LOCAL so we don't miss the map if it was published before we started ---
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, map_qos)

        # --- ground truth gives us the exact robot position from the simulator ---
        self.gt_sub = self.create_subscription(Odometry, "/ground_truth", self.pose_callback, 10)

        # --- velocity commands sent to the simulator to move the robot ---
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # --- path markers published so RViz can draw the green and blue paths ---
        self.marker_pub = self.create_publisher(MarkerArray, "/planner_markers", 10)

        # --- service client to call get_task on the grading scout ---
        # this gives us the start/goal coordinates and resets the energy counter
        self.task_client = self.create_client(Trigger, "get_task")

        self.get_logger().info("PathPlanner node started, waiting for map...")


    # -----------------------------------------------------------------------
    # MAP CALLBACK
    # called once when the map arrives on /map topic
    # -----------------------------------------------------------------------
    def map_callback(self, msg):
        # only store the map once, ignore any future map updates
        if self.map_data is not None:
            return
        self.map_info = msg.info    # save map metadata (resolution, origin, size)
        self.map_data = msg         # save the full map message
        self.get_logger().info("Map received! Requesting task from grading scout...")
        self.request_task()


    # -----------------------------------------------------------------------
    # REQUEST TASK
    # calls the get_task service on the grading scout to get start/goal coords
    # also resets the energy counter to zero so grading starts fresh
    # -----------------------------------------------------------------------
    def request_task(self):
        # wait until the grading scout service is available
        while not self.task_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_task service...")

        req = Trigger.Request()
        # call the service asynchronously so we don't block ROS
        future = self.task_client.call_async(req)
        # when the response arrives, call task_response_callback
        future.add_done_callback(self.task_response_callback)


    # -----------------------------------------------------------------------
    # TASK RESPONSE CALLBACK
    # called when the grading scout responds to our get_task request
    # response.message contains "start_x,start_y,goal_x,goal_y"
    # -----------------------------------------------------------------------
    def task_response_callback(self, future):
        response = future.result()
        if response.success:
            # split the comma-separated string into 4 float coordinates
            coords = response.message.split(",")
            self.start_x = float(coords[0])
            self.start_y = float(coords[1])
            self.goal_x  = float(coords[2])
            self.goal_y  = float(coords[3])
            self.get_logger().info(f"Task received! Start({self.start_x},{self.start_y}) Goal({self.goal_x},{self.goal_y})")

            # run A* in a background thread so ROS keeps spinning
            # without threading, A* would freeze ROS for 2-3 seconds
            thread = threading.Thread(target=self.run_planner)
            thread.daemon = True  # thread dies automatically when main program exits
            thread.start()
        else:
            self.get_logger().error("Failed to get task from grading scout!")


    # -----------------------------------------------------------------------
    # POSE CALLBACK
    # called every time /ground_truth publishes a new robot position
    # extracts x, y, and yaw (heading angle) from the odometry message
    # -----------------------------------------------------------------------
    def pose_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # the orientation comes as a quaternion (q.x, q.y, q.z, q.w)
        # we need to convert it to yaw (the rotation angle around the vertical axis)
        # this is standard math for extracting yaw from a quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)


    # -----------------------------------------------------------------------
    # WORLD TO GRID
    # converts real world coordinates (meters) to grid pixel coordinates
    # formula: col = (world_x - origin_x) / resolution
    # example: world(-7.0, -7.0) --> grid(31, 31)
    # -----------------------------------------------------------------------
    def world_to_grid(self, wx, wy):
        col = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        row = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        return row, col


    # -----------------------------------------------------------------------
    # GRID TO WORLD
    # converts grid pixel coordinates back to real world coordinates (meters)
    # formula: world_x = origin_x + col * resolution
    # this is used after A* to convert the pixel path back to meter coordinates
    # -----------------------------------------------------------------------
    def grid_to_world(self, row, col):
        wx = self.map_info.origin.position.x + col * self.map_info.resolution
        wy = self.map_info.origin.position.y + row * self.map_info.resolution
        return wx, wy


    # -----------------------------------------------------------------------
    # BUILD GRID
    # converts the raw occupancy grid map into a clean binary 2D array
    # then inflates obstacles so the robot stays safely away from walls
    # -----------------------------------------------------------------------
    def build_grid(self):
        width  = self.map_info.width
        height = self.map_info.height

        # reshape the flat map data list into a 2D array of height x width
        raw = np.array(self.map_data.data).reshape((height, width))

        # cells with occupancy value > 65 are walls/obstacles, everything else is free space
        # this threshold comes from occupied_thresh: 0.65 in map.yaml
        binary = np.zeros((height, width), dtype=np.uint8)
        binary[raw > 65] = 1  # mark obstacles as 1, free space stays 0

        # inflate obstacles by 8 pixels (8 * 0.032m = 0.256m safety margin)
        # this prevents A* from planning paths that go too close to walls
        # binary_dilation expands every obstacle cell by the given radius in all directions
        inflation_radius = 8
        struct   = np.ones((2*inflation_radius+1, 2*inflation_radius+1), dtype=bool)
        inflated = binary_dilation(binary, structure=struct).astype(np.uint8)

        self.grid = inflated
        self.get_logger().info(f"Grid built: {width}x{height}, inflation radius={inflation_radius}")


    # -----------------------------------------------------------------------
    # RUN A*
    # finds the shortest collision-free path from start to goal on the grid
    # uses a priority queue (heapq) to always explore the cheapest cell first
    # -----------------------------------------------------------------------
    def run_astar(self, start_world, goal_world):
        # convert start and goal from world coords (meters) to grid coords (pixels)
        sr, sc = self.world_to_grid(start_world[0], start_world[1])
        gr, gc = self.world_to_grid(goal_world[0],  goal_world[1])

        self.get_logger().info(f"A* from grid({sr},{sc}) to grid({gr},{gc})")

        height, width = self.grid.shape

        # heuristic: straight line distance from current cell to goal
        # this guides A* towards the goal instead of searching blindly
        def heuristic(r, c):
            return math.sqrt((r - gr)**2 + (c - gc)**2)

        # heap stores (f_cost, g_cost, row, col)
        # f = g + h, we always pop the cell with the lowest f first
        open_heap = []
        heapq.heappush(open_heap, (heuristic(sr, sc), 0.0, sr, sc))

        # came_from stores breadcrumbs: "I reached this cell FROM that cell"
        came_from = {}

        # g_scores stores the actual travel cost from start to each visited cell
        g_scores = {(sr, sc): 0.0}

        # 8 possible moves: up, down, left, right (cost 1.0) and 4 diagonals (cost 1.414)
        # diagonal costs more because the actual distance is sqrt(2) = 1.414 by Pythagoras
        directions = [
            (-1,  0, 1.0),   # up
            ( 1,  0, 1.0),   # down
            ( 0, -1, 1.0),   # left
            ( 0,  1, 1.0),   # right
            (-1, -1, 1.414), # up-left diagonal
            (-1,  1, 1.414), # up-right diagonal
            ( 1, -1, 1.414), # down-left diagonal
            ( 1,  1, 1.414), # down-right diagonal
        ]

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)  # always get the cheapest cell

            # goal reached - trace back the breadcrumbs to build the path
            if (r, c) == (gr, gc):
                path = []
                node = (gr, gc)
                # follow came_from backwards from goal to start
                while node in came_from:
                    wr, wc = self.grid_to_world(node[0], node[1])
                    path.append((wr, wc))
                    node = came_from[node]
                # add the start position
                wr, wc = self.grid_to_world(sr, sc)
                path.append((wr, wc))
                # reverse because we traced from goal to start, we need start to goal
                path.reverse()
                self.get_logger().info(f"A* found path with {len(path)} waypoints")
                return path

            # check all 8 neighbors of current cell
            for dr, dc, move_cost in directions:
                nr, nc = r + dr, c + dc

                # skip if neighbor is outside the grid boundaries
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                # skip if neighbor is an obstacle (inflated wall)
                if self.grid[nr][nc] == 1:
                    continue

                # calculate cost to reach this neighbor through current cell
                new_g = g + move_cost

                # only update if this is a cheaper way to reach this neighbor
                if (nr, nc) not in g_scores or new_g < g_scores[(nr, nc)]:
                    g_scores[(nr, nc)] = new_g
                    f_new = new_g + heuristic(nr, nc)
                    heapq.heappush(open_heap, (f_new, new_g, nr, nc))
                    came_from[(nr, nc)] = (r, c)  # store breadcrumb

        self.get_logger().error("A* could not find a path!")
        return []


    # -----------------------------------------------------------------------
    # CHECK LINE OF SIGHT
    # checks if a straight line between two world points is free of obstacles
    # uses Bresenham's line algorithm which traces all grid cells the line passes through
    # returns True if all cells are free, False if any cell is an obstacle
    # -----------------------------------------------------------------------
    def check_los(self, wx1, wy1, wx2, wy2):
        # convert both world points to grid cells
        r1, c1 = self.world_to_grid(wx1, wy1)
        r2, c2 = self.world_to_grid(wx2, wy2)

        # Bresenham's line algorithm - integer only, no floating point
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        r, c   = r1, c1
        step_r = 1 if r2 > r1 else -1  # which direction to step in rows
        step_c = 1 if c2 > c1 else -1  # which direction to step in cols
        error  = dr - dc               # error accumulator to decide which direction to step

        height, width = self.grid.shape

        while True:
            # if current cell is out of bounds, line of sight is blocked
            if r < 0 or r >= height or c < 0 or c >= width:
                return False

            # if current cell is an obstacle, line of sight is blocked
            if self.grid[r][c] == 1:
                return False

            # reached the destination cell - line of sight is clear
            if r == r2 and c == c2:
                break

            # decide whether to step in row direction, col direction, or both
            double_error = 2 * error
            if double_error > -dc:
                error -= dc
                r += step_r
            if double_error < dr:
                error += dr
                c += step_c

        return True  # all cells along the line were free


    # -----------------------------------------------------------------------
    # PRUNE PATH
    # simplifies the A* path using line-of-sight skipping
    # instead of following every tiny step A* took, we jump as far ahead
    # as possible in a straight line, only adding a waypoint when we must turn
    # this reduces 618 waypoints down to ~22 waypoints
    # -----------------------------------------------------------------------
    def prune_path(self, path):
        # no pruning needed if path is very short
        if len(path) < 3:
            return path

        pruned      = [path[0]]  # always keep the start point
        current_idx = 0

        while current_idx < len(path) - 1:
            furthest = current_idx + 1  # at minimum we must go to the next waypoint

            # try to jump as far ahead as possible with a clear line of sight
            for look_ahead in range(current_idx + 2, len(path)):
                wx1, wy1 = pruned[-1]           # where we currently are
                wx2, wy2 = path[look_ahead]     # where we are trying to jump to
                if self.check_los(wx1, wy1, wx2, wy2):
                    furthest = look_ahead  # line is clear, we can jump this far
                else:
                    break  # line is blocked, stop looking further ahead

            pruned.append(path[furthest])  # add the furthest reachable waypoint
            current_idx = furthest         # move our position forward

        self.get_logger().info(f"Path pruned: {len(path)} -> {len(pruned)} waypoints")
        return pruned


    # -----------------------------------------------------------------------
    # PUBLISH PATHS
    # sends the green and blue path markers to RViz for visualization
    # green = raw A* path (all 618 waypoints)
    # blue  = pruned path (only ~22 waypoints)
    # -----------------------------------------------------------------------
    def publish_paths(self):
        marker_array = MarkerArray()
        marker_id    = 0  # each marker needs a unique ID, we just count up

        # green spheres for the raw A* path
        for (wx, wy) in self.raw_path:
            m = Marker()
            m.header.frame_id   = "map"        # markers are in the map coordinate frame
            m.ns                = "raw_path"   # namespace groups related markers
            m.id                = marker_id
            m.type              = Marker.SPHERE
            m.action            = Marker.ADD
            m.pose.position.x   = wx
            m.pose.position.y   = wy
            m.pose.position.z   = 0.1          # slightly above ground so it's visible
            m.pose.orientation.w = 1.0         # no rotation
            m.scale.x = m.scale.y = m.scale.z = 0.1   # sphere size in meters
            m.color.r = 0.0
            m.color.g = 1.0   # green
            m.color.b = 0.0
            m.color.a = 1.0   # fully opaque
            marker_array.markers.append(m)
            marker_id += 1

        # blue spheres for the pruned path (slightly bigger so they stand out)
        for (wx, wy) in self.pruned_path:
            m = Marker()
            m.header.frame_id   = "map"
            m.ns                = "pruned_path"
            m.id                = marker_id
            m.type              = Marker.SPHERE
            m.action            = Marker.ADD
            m.pose.position.x   = wx
            m.pose.position.y   = wy
            m.pose.position.z   = 0.2          # slightly higher than green path
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15  # bigger than green dots
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0   # blue
            m.color.a = 1.0
            marker_array.markers.append(m)
            marker_id += 1

        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(self.raw_path)} green and {len(self.pruned_path)} blue markers")


    # -----------------------------------------------------------------------
    # START CONTROLLER
    # creates a timer that calls control_loop every 0.1 seconds
    # the timer keeps running until the mission is complete
    # -----------------------------------------------------------------------
    def start_controller(self):
        self.current_target_idx = 0      # start from the first waypoint
        self.mission_active     = True   # tell control_loop it should run
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Controller started!")


    # -----------------------------------------------------------------------
    # CONTROL LOOP
    # runs every 0.1 seconds to drive the robot towards the next waypoint
    # uses smooth blended motion - robot keeps moving forward while turning
    # this avoids stop/start which wastes energy (startup tax = 0.6 per stop)
    # -----------------------------------------------------------------------
    def control_loop(self):
        # do nothing if mission is not active
        if not self.mission_active:
            return

        # all waypoints reached - stop the robot and end mission
        if self.current_target_idx >= len(self.pruned_path):
            self.stop_robot()
            self.mission_active = False
            self.get_logger().info("Mission complete! Robot reached the goal.")
            return

        # get the current target waypoint coordinates
        target_x, target_y = self.pruned_path[self.current_target_idx]

        # calculate how far and in what direction the target is
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        distance        = math.sqrt(dx**2 + dy**2)   # straight line distance to target
        angle_to_target = math.atan2(dy, dx)          # angle we need to face

        # angle_error = how much we need to rotate to face the target
        angle_error = angle_to_target - self.robot_yaw

        # normalize angle to [-pi, pi] to always take the shortest rotation
        # without this, robot might rotate 350 degrees instead of -10 degrees
        while angle_error >  math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # waypoint reached - move to the next one
        if distance < 0.3:
            self.current_target_idx += 1
            self.get_logger().info(f"Waypoint {self.current_target_idx} reached, {len(self.pruned_path) - self.current_target_idx} remaining")
            return

        cmd = Twist()  # velocity command message (linear.x = forward, angular.z = rotation)

        # SMOOTH CONTROLLER: always blend forward motion with turning
        # the more we need to turn, the slower we go forward - but we never fully stop
        # this avoids the startup tax that charges 0.6 energy every time we stop then move

        # forward speed: reduce speed proportionally when angle error is large
        # when facing directly at target (angle_error=0): full speed 0.5 m/s
        # when facing 90 degrees away (angle_error=pi/2): half speed
        cmd.linear.x = 0.5 * (1.0 - min(abs(angle_error) / math.pi, 1.0))
        cmd.linear.x = max(0.1, cmd.linear.x)  # never go below 0.1 m/s to avoid full stop

        # turning speed: proportional to how much we need to turn
        cmd.angular.z = 1.5 * angle_error
        # clamp angular velocity to [-2.0, 2.0] to prevent spinning too fast
        cmd.angular.z = max(-2.0, min(2.0, cmd.angular.z))

        self.vel_pub.publish(cmd)


    # -----------------------------------------------------------------------
    # STOP ROBOT
    # sends zero velocity to bring the robot to a complete stop
    # only called when mission is fully complete
    # -----------------------------------------------------------------------
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x  = 0.0  # no forward motion
        cmd.angular.z = 0.0  # no rotation
        self.vel_pub.publish(cmd)


    # -----------------------------------------------------------------------
    # RUN PLANNER
    # main planning sequence called in background thread
    # 1. build the inflated obstacle grid
    # 2. run A* to find raw path
    # 3. prune the path using line of sight
    # 4. visualize both paths in RViz
    # 5. start the controller to drive the robot
    # -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# MAIN
# entry point of the program
# initializes ROS, creates the PathPlanner node, and keeps it running
# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)        # initialize ROS 2
    node = PathPlanner()         # create our planner node
    rclpy.spin(node)             # keep the node alive and processing callbacks
    rclpy.shutdown()             # clean up when done


if __name__ == "__main__":
    main()
