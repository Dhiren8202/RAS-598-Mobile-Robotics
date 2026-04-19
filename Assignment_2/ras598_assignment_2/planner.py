import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import MarkerArray, Marker
from example_interfaces.srv import Trigger
from std_msgs.msg import Float32
from scipy.ndimage import binary_dilation
from PIL import Image
import math
import heapq
import numpy as np
import threading
import os


class PathPlanner(Node):
    """
    This is the main planner node for Assignment 2.
    It loads the cave map, builds a grid, runs A* to find a path,
    prunes the path using Line of Sight, and drives the robot to the goal.
    """

    def __init__(self):
        super().__init__("path_planner")

        # --- GRID SETTINGS (as required by assignment spec) ---
        self.cell_size  = 0.2   # each grid cell represents 0.2m x 0.2m in the real world
        self.world_size = 16.0  # the cave world is 16m x 16m total
        self.origin     = -8.0  # the bottom-left corner of the world is at (-8, -8) in meters
        self.grid_size  = 80    # 16m / 0.2m = 80 cells per side, so we have an 80x80 grid
        self.grid       = None  # this will hold our obstacle map after build_grid() runs

        # --- ROBOT POSITION (updated continuously from /ground_truth topic) ---
        self.robot_x   = 0.0   # robot x position in meters (east-west)
        self.robot_y   = 0.0   # robot y position in meters (north-south)
        self.robot_yaw = 0.0   # robot heading angle in radians (which direction it faces)

        # --- MISSION COORDINATES (received from grading scout via get_task service) ---
        self.start_x = 0.0  # x coordinate of starting position
        self.start_y = 0.0  # y coordinate of starting position
        self.goal_x  = 0.0  # x coordinate of goal position
        self.goal_y  = 0.0  # y coordinate of goal position

        # --- PATH STORAGE ---
        self.raw_path    = []  # the full A* path before simplification (shown as green line in RViz)
        self.pruned_path = []  # the simplified path after line-of-sight pruning (shown as blue line in RViz)

        # --- CONTROLLER STATE ---
        self.current_target_idx = 0      # which waypoint in pruned_path we are currently heading to
        self.mission_active     = False  # True when robot is actively driving, False when done

        # --- ROS SUBSCRIBERS ---
        # /ground_truth gives us the exact robot position from the simulator
        self.gt_sub = self.create_subscription(
            Odometry, "/ground_truth", self.pose_callback, 10
        )

        # /energy_consumed gives us real-time energy feedback from the grading scout
        self.energy_sub = self.create_subscription(
            Float32, "/energy_consumed", self.energy_callback, 10
        )
        self.current_energy = 0.0  # stores the latest energy reading

        # --- ROS PUBLISHERS ---
        # /cmd_vel sends velocity commands to move the robot (linear and angular speed)
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # /planner_markers sends the path visualization to RViz (green, blue, red markers)
        self.marker_pub = self.create_publisher(MarkerArray, "/planner_markers", 10)

        # --- SERVICE CLIENT ---
        # get_task is called once at startup to get start/goal coordinates
        # it also resets the energy counter to zero so grading starts fresh
        self.task_client = self.create_client(Trigger, "get_task")

        # --- START THE PROCESS ---
        # Step 1: load the cave image and build the obstacle grid
        self.build_grid()
        # Step 2: call the grading scout to get our mission coordinates
        self.request_task()

        self.get_logger().info("PathPlanner node started...")


    def energy_callback(self, msg):
        # called every time the grading scout publishes a new energy reading
        # we store it so we can monitor our score in real time
        self.current_energy = msg.data

    def build_grid(self):
        """
        Loads the cave_filled.png image directly and converts it into an 80x80 binary grid.
        
        White pixels in the image = free space (robot can go here)
        Black pixels in the image = walls/obstacles (robot cannot go here)
        
        We also inflate (expand) the obstacles by 3 cells = 0.6m so the robot
        always keeps a safe distance from walls and doesnt clip them.
        """

        # find and load the cave image file
        home     = os.path.expanduser("~")
        img_path = os.path.join(home, "ros_ws/src/ras598_assignment_2/cave_filled.png")
        img      = Image.open(img_path).convert("L")  # "L" means grayscale (0=black, 255=white)
        img_arr  = np.array(img)                       # convert to a numpy array of numbers

        # the image has its origin at the TOP-LEFT corner
        # but ROS world coordinates have origin at BOTTOM-LEFT
        # so we flip the image upside down to match the world coordinate system
        img_arr = np.flipud(img_arr)

        # mark dark pixels as obstacles (value < 128 means dark = wall)
        # and light pixels as free space (value >= 128 means light = walkable)
        binary_full = np.zeros(img_arr.shape, dtype=np.uint8)
        binary_full[img_arr < 128] = 1  # 1 = obstacle, 0 = free

        # the original image is 500x500 pixels but we need an 80x80 grid
        # each cell in our 80x80 grid covers 500/80 = 6.25 original pixels
        # we check each 6.25x6.25 pixel region - if ANY pixel is a wall, the whole cell is a wall
        scale        = img_arr.shape[0] / self.grid_size  # how many pixels per grid cell
        binary_small = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # find which pixels in the original image this cell covers
                r0 = int(r * scale)
                r1 = min(int((r+1) * scale), img_arr.shape[0])
                c0 = int(c * scale)
                c1 = min(int((c+1) * scale), img_arr.shape[1])
                # if any pixel in this region is an obstacle, mark the whole cell as obstacle
                if np.any(binary_full[r0:r1, c0:c1] == 1):
                    binary_small[r][c] = 1

        # inflate (expand) obstacles by 3 cells in every direction
        # this creates a 0.6m safety buffer around every wall
        # so A* never plans a path that goes too close to a wall
        inflation_radius = 3  # 3 cells x 0.2m per cell = 0.6m safety margin
        struct   = np.ones((2*inflation_radius+1, 2*inflation_radius+1), dtype=bool)  # 7x7 block of True
        inflated = binary_dilation(binary_small, structure=struct).astype(np.uint8)   # expand all obstacles

        self.grid = inflated
        self.get_logger().info(
            f"Grid built: {self.grid_size}x{self.grid_size} at {self.cell_size}m/cell, "
            f"inflation={inflation_radius} cells ({inflation_radius*self.cell_size}m)"
        )


    def world_to_grid(self, wx, wy):
        """
        Converts real world coordinates (in meters) to grid cell coordinates.
        
        The world goes from -8m to +8m in both x and y.
        The grid goes from 0 to 79 in both row and col.
        
        Formula: col = (world_x - origin) / cell_size
                 row = (world_y - origin) / cell_size
        
        Example: world(-7.0, -7.0) --> grid(5, 5)   [bottom-left area]
                 world( 7.0,  2.5) --> grid(52, 75)  [right-middle area]
                 world( 0.0,  0.0) --> grid(40, 40)  [center of map]
        """
        col = int((wx - self.origin) / self.cell_size)
        row = int((wy - self.origin) / self.cell_size)
        # clamp to grid boundaries so we never go out of bounds
        col = max(0, min(self.grid_size - 1, col))
        row = max(0, min(self.grid_size - 1, row))
        return row, col


    def grid_to_world(self, row, col):
        """
        Converts grid cell coordinates back to real world coordinates (in meters).
        We add half a cell size (0.1m) to return the CENTER of the cell, not its corner.
        
        Formula: world_x = origin + col * cell_size + half_cell
                 world_y = origin + row * cell_size + half_cell
        """
        wx = self.origin + col * self.cell_size + self.cell_size / 2.0
        wy = self.origin + row * self.cell_size + self.cell_size / 2.0
        return wx, wy


    def request_task(self):
        """
        Calls the get_task service on the grading scout.
        This does two things:
        1. Gives us the start and goal coordinates for the mission
        2. Resets the energy counter to zero so our score starts fresh
        
        We wait until the service is available before calling it.
        The response comes back asynchronously via task_response_callback.
        """
        while not self.task_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_task service...")

        req    = Trigger.Request()
        future = self.task_client.call_async(req)  # call the service without blocking
        future.add_done_callback(self.task_response_callback)  # call this when response arrives


    def task_response_callback(self, future):
        """
        Called when the grading scout responds to our get_task request.
        
        The response message is a string like: "-7.0,-7.0,7.0,2.5"
        We split it by comma to get: start_x, start_y, goal_x, goal_y
        
        After storing the coordinates, we launch A* in a background thread
        so ROS can keep running while we compute the path.
        """
        response = future.result()
        if response.success:
            coords       = response.message.split(",")  # split "-7.0,-7.0,7.0,2.5" into 4 parts
            self.start_x = float(coords[0])  # -7.0
            self.start_y = float(coords[1])  # -7.0
            self.goal_x  = float(coords[2])  #  7.0
            self.goal_y  = float(coords[3])  #  2.5
            self.get_logger().info(
                f"Task received! Start({self.start_x},{self.start_y}) Goal({self.goal_x},{self.goal_y})"
            )
            # run A* in a background thread so ROS doesnt freeze
            # A* takes 2-3 seconds and would block all ROS callbacks if run directly
            thread        = threading.Thread(target=self.run_planner)
            thread.daemon = True  # thread dies automatically when the program exits
            thread.start()
        else:
            self.get_logger().error("Failed to get task from grading scout!")


    def pose_callback(self, msg):
        """
        Called every time the /ground_truth topic publishes a new robot position.
        
        Extracts x, y position and yaw (heading angle) from the Odometry message.
        
        The orientation comes as a quaternion (4 numbers: x, y, z, w).
        We need to convert it to yaw (just one angle around the vertical axis).
        This is standard math for extracting yaw from a quaternion.
        """
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # convert quaternion to yaw angle using standard formula
        q         = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)  # result is in radians (-pi to +pi)


    def run_astar(self, start_world, goal_world):
        """
        Runs the A* pathfinding algorithm from start to goal on our 80x80 grid.
        
        A* works like a smart explorer - it always explores the cell that looks
        cheapest to reach AND closest to the goal. It uses a priority queue (heap)
        so the cheapest cell is always at the front.
        
        Cost of each move:
        - Straight moves (up/down/left/right): cost = 1.0
        - Diagonal moves: cost = 1.414 (square root of 2, from Pythagoras theorem)
        
        Returns a list of (x, y) world coordinates from start to goal.
        """
        # convert start and goal from world meters to grid cell numbers
        sr, sc = self.world_to_grid(start_world[0], start_world[1])
        gr, gc = self.world_to_grid(goal_world[0],  goal_world[1])

        self.get_logger().info(f"A* from grid({sr},{sc}) to grid({gr},{gc})")
        self.get_logger().info(f"Start cell obstacle={self.grid[sr][sc]}, Goal cell obstacle={self.grid[gr][gc]}")

        height, width = self.grid.shape

        # heuristic = straight line distance to goal
        # this guides A* to explore cells closer to the goal first
        def heuristic(r, c):
            return math.sqrt((r - gr)**2 + (c - gc)**2)

        # the open heap is our "to-do list" of cells to explore
        # each entry is (total_cost, travel_cost, row, col)
        # heapq always gives us the lowest total_cost cell first
        open_heap = []
        heapq.heappush(open_heap, (heuristic(sr, sc), 0.0, sr, sc))

        # came_from stores breadcrumbs: "I reached this cell FROM that cell"
        # when we reach the goal, we follow these backwards to rebuild the path
        came_from = {}

        # g_scores stores the actual travel cost from start to each visited cell
        g_scores = {(sr, sc): 0.0}

        # 8 possible moves from any cell
        # (row_change, col_change, movement_cost)
        directions = [
            (-1,  0, 1.0),    # move up
            ( 1,  0, 1.0),    # move down
            ( 0, -1, 1.0),    # move left
            ( 0,  1, 1.0),    # move right
            (-1, -1, 1.414),  # move up-left diagonal
            (-1,  1, 1.414),  # move up-right diagonal
            ( 1, -1, 1.414),  # move down-left diagonal
            ( 1,  1, 1.414),  # move down-right diagonal
        ]

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)  # always get the cheapest cell

            # if we reached the goal, trace back the breadcrumbs to get the path
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
                # reverse because we traced goal->start, but we need start->goal
                path.reverse()
                self.get_logger().info(f"A* found path with {len(path)} waypoints")
                return path

            # check all 8 neighbors of the current cell
            for dr, dc, move_cost in directions:
                nr, nc = r + dr, c + dc

                # skip cells outside the grid boundaries
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                # skip cells that are obstacles (inflated walls)
                if self.grid[nr][nc] == 1:
                    continue

                # calculate the cost to reach this neighbor through the current cell
                new_g = g + move_cost

                # only update if this is a cheaper way to reach this neighbor
                if (nr, nc) not in g_scores or new_g < g_scores[(nr, nc)]:
                    g_scores[(nr, nc)] = new_g
                    total_cost = new_g + heuristic(nr, nc)
                    heapq.heappush(open_heap, (total_cost, new_g, nr, nc))
                    came_from[(nr, nc)] = (r, c)  # drop a breadcrumb

        self.get_logger().error("A* could not find a path!")
        return []


    def check_los(self, wx1, wy1, wx2, wy2):
        """
        Checks if there is a clear Line of Sight between two world points.
        Uses Bresenham line algorithm to trace every grid cell the straight line passes through.
        
        Returns True if ALL cells along the line are free (no obstacles).
        Returns False if ANY cell along the line is an obstacle.
        
        This is used by prune_path to decide if we can skip intermediate waypoints.
        """
        # convert both world points to grid cells
        r1, c1 = self.world_to_grid(wx1, wy1)
        r2, c2 = self.world_to_grid(wx2, wy2)

        # Bresenham line algorithm variables
        dr     = abs(r2 - r1)      # total row distance
        dc     = abs(c2 - c1)      # total col distance
        r, c   = r1, c1            # start at the first point
        step_r = 1 if r2 > r1 else -1   # which direction to step in rows (+1 or -1)
        step_c = 1 if c2 > c1 else -1   # which direction to step in cols (+1 or -1)
        error  = dr - dc           # error accumulator to decide which direction to step next

        h, w = self.grid.shape

        while True:
            # if current cell is out of bounds, line of sight is blocked
            if r < 0 or r >= h or c < 0 or c >= w:
                return False

            # if current cell is an obstacle, line of sight is blocked
            if self.grid[r][c] == 1:
                return False

            # reached the destination cell - all cells were clear
            if r == r2 and c == c2:
                break

            # decide next step using Bresenham error accumulator
            de = 2 * error
            if de > -dc:
                error -= dc
                r     += step_r
            if de < dr:
                error += dr
                c     += step_c

        return True  # all cells along the line are free


    def in_tight_corridor(self, wx, wy):
        """
        Returns True if the given world point is in a tight/dangerous area of the cave.
        
        We identified three tight areas through testing:
        1. The middle corridor (around x=-3 to 0.5, y=-2 to 3.5) - narrow passage
        2. The top area (y >= 5.5) - close to the top boundary wall
        3. The right side (x >= 5.0) - close to the right boundary wall near goal
        
        In these areas we limit how far ahead we can jump in path pruning
        to avoid creating diagonal shortcuts that clip walls.
        """
        middle = (-3.0 <= wx <= 0.5) and (-2.0 <= wy <= 3.5)  # narrow middle corridor
        top    = (wy >= 5.5)                                    # near top wall
        right  = (wx >= 5.0)                                    # near right wall and goal
        return middle or top or right


    def prune_path(self, path):
        """
        Simplifies the A* path using Line of Sight (LOS) pruning.
        
        A* produces many small steps that hug walls. We dont need to follow every step.
        Instead, we look ahead and ask: "can I jump directly to a further waypoint
        in a straight line without hitting any wall?"
        
        If yes, we skip all the waypoints in between and jump directly.
        This reduces hundreds of waypoints down to just 10-15 waypoints.
        
        Fewer waypoints = fewer turns = fewer stops = less energy consumed.
        
        We use different maximum jump distances for different areas:
        - Tight corridors: max 6m jump (careful in narrow areas)
        - Open areas: max 50m jump (basically no limit in open space)
        """
        if len(path) < 3:
            return path  # too short to prune, return as-is

        pruned      = [path[0]]  # always keep the starting point
        current_idx = 0

        while current_idx < len(path) - 1:
            furthest = current_idx + 1  # at minimum, go to the very next waypoint

            # try to jump as far ahead as possible
            for look_ahead in range(current_idx + 2, len(path)):
                wx1, wy1 = pruned[-1]          # where we currently are
                wx2, wy2 = path[look_ahead]    # the waypoint we are trying to jump to

                # decide maximum allowed jump distance based on location
                if self.in_tight_corridor(wx1, wy1) or self.in_tight_corridor(wx2, wy2):
                    max_jump = 7.0   # tight corridors - shorter jumps to avoid wall clipping
                else:
                    max_jump = 50.0  # open areas - no practical limit needed

                # calculate straight line distance of this potential jump
                jump_dist = math.sqrt((wx2 - wx1)**2 + (wy2 - wy1)**2)

                # if this jump is too long, stop looking further
                if jump_dist > max_jump:
                    break

                # if line of sight is clear, we can jump this far
                if self.check_los(wx1, wy1, wx2, wy2):
                    furthest = look_ahead  # update how far we can jump
                else:
                    break  # line of sight is blocked, stop looking further

            # add the furthest reachable waypoint to our pruned path
            pruned.append(path[furthest])
            current_idx = furthest  # move our position forward

        self.get_logger().info(f"Path pruned: {len(path)} -> {len(pruned)} waypoints")
        return pruned


    def publish_paths(self):
        """
        Publishes the green and blue path lines and red goal sphere to RViz.
        
        Green LINE_STRIP = the raw A* path (all original waypoints)
        Blue LINE_STRIP  = the pruned path (simplified waypoints the robot follows)
        Red SPHERE       = the current target waypoint the robot is heading to
        
        LINE_STRIP draws connected lines between points (not individual dots).
        """
        marker_array = MarkerArray()

        # --- GREEN LINE for raw A* path ---
        green                    = Marker()
        green.header.frame_id    = "map"       # markers are in the map coordinate frame
        green.ns                 = "raw_path"  # namespace to group related markers
        green.id                 = 0           # unique ID within this namespace
        green.type               = Marker.LINE_STRIP  # connected line between points
        green.action             = Marker.ADD
        green.scale.x            = 0.05        # line width = 5cm
        green.color.r            = 0.0
        green.color.g            = 1.0         # green color
        green.color.b            = 0.0
        green.color.a            = 1.0         # fully opaque
        green.pose.orientation.w = 1.0         # no rotation needed for line markers
        for (wx, wy) in self.raw_path:
            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.1  # slightly above ground so its visible
            green.points.append(p)
        marker_array.markers.append(green)

        # --- BLUE LINE for pruned path ---
        blue                    = Marker()
        blue.header.frame_id    = "map"
        blue.ns                 = "pruned_path"
        blue.id                 = 1
        blue.type               = Marker.LINE_STRIP
        blue.action             = Marker.ADD
        blue.scale.x            = 0.08         # slightly thicker than green line
        blue.color.r            = 0.0
        blue.color.g            = 0.0
        blue.color.b            = 1.0          # blue color
        blue.color.a            = 1.0
        blue.pose.orientation.w = 1.0
        for (wx, wy) in self.pruned_path:
            p = Point()
            p.x = wx
            p.y = wy
            p.z = 0.2  # slightly higher than green line
            blue.points.append(p)
        marker_array.markers.append(blue)

        # --- RED SPHERE for current target waypoint ---
        self.add_goal_marker(marker_array)

        self.marker_pub.publish(marker_array)
        self.get_logger().info(
            f"Published green({len(self.raw_path)}pts) blue({len(self.pruned_path)}pts)"
        )


    def add_goal_marker(self, marker_array):
        """
        Adds a red sphere marker at the current target waypoint position.
        This shows in RViz which waypoint the robot is currently heading towards.
        The sphere moves forward as each waypoint is reached.
        """
        if not self.pruned_path:
            return

        # get the current target waypoint (dont go out of bounds at the end)
        idx   = min(self.current_target_idx, len(self.pruned_path) - 1)
        tx, ty = self.pruned_path[idx]

        red                    = Marker()
        red.header.frame_id    = "map"
        red.ns                 = "current_goal"
        red.id                 = 2
        red.type               = Marker.SPHERE  # sphere shape
        red.action             = Marker.ADD
        red.pose.position.x    = tx
        red.pose.position.y    = ty
        red.pose.position.z    = 0.3            # above the path lines
        red.pose.orientation.w = 1.0
        red.scale.x = red.scale.y = red.scale.z = 0.3  # 30cm diameter sphere
        red.color.r = 1.0   # red color
        red.color.g = 0.0
        red.color.b = 0.0
        red.color.a = 1.0
        marker_array.markers.append(red)


    def update_goal_marker(self):
        """
        Publishes just the red sphere marker update during the control loop.
        Called every 0.1 seconds to keep the red sphere at the correct waypoint.
        """
        if not self.pruned_path:
            return
        ma = MarkerArray()
        self.add_goal_marker(ma)
        self.marker_pub.publish(ma)


    def start_controller(self):
        """
        Initializes and starts the control timer.
        The control_loop runs every 0.1 seconds until the mission is complete.
        We always start in rotating mode so the robot faces the first waypoint
        before it starts moving.
        """
        self.current_target_idx = 0
        self.mission_active     = True
        self.rotating_to_next   = True   # always rotate first before driving
        self.segment_start_x    = self.robot_x  # records where each straight segment starts
        self.segment_start_y    = self.robot_y
        self.control_timer      = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Controller started!")


    def calc_lateral_drift(self, target_x, target_y):
        """
        Calculates how far sideways (laterally) the robot has drifted
        from the intended straight line between segment_start and target.

        Uses the cross product formula:
        lateral_drift = |(target - start) x (robot - start)| / |target - start|

        Returns the perpendicular distance from the line in meters.
        """
        # vector from segment start to target
        line_dx = target_x - self.segment_start_x
        line_dy = target_y - self.segment_start_y
        line_len = math.sqrt(line_dx**2 + line_dy**2)

        if line_len < 0.001:
            return 0.0

        # vector from segment start to robot current position
        robot_dx = self.robot_x - self.segment_start_x
        robot_dy = self.robot_y - self.segment_start_y

        # cross product magnitude = lateral distance from the line
        cross = abs(line_dx * robot_dy - line_dy * robot_dx)
        return cross / line_len


    def control_loop(self):
        """
        Pure Turn-Go-Turn controller with lateral drift detection.

        STATE 1 - ROTATE: Stop completely and rotate precisely to face the waypoint.
                          Stay in this state until angle error < 0.03 rad (~2 degrees).

        STATE 2 - DRIVE:  Drive straight forward with angular.z = exactly 0.0.
                          Monitor lateral drift from the intended straight line.
                          If drift exceeds 0.3m, stop and go back to STATE 1 to re-align.
                          This gives ONE clean correction instead of many micro-corrections.
        """
        if not self.mission_active:
            return

        if self.current_target_idx >= len(self.pruned_path):
            self.stop_robot()
            self.mission_active = False
            self.get_logger().info("Mission complete!")
            return

        self.update_goal_marker()

        target_x, target_y = self.pruned_path[self.current_target_idx]

        dx              = target_x - self.robot_x
        dy              = target_y - self.robot_y
        distance        = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)

        angle_error = angle_to_target - self.robot_yaw
        while angle_error >  math.pi: angle_error -= 2 * math.pi
        while angle_error < -math.pi: angle_error += 2 * math.pi

        # waypoint reached
        if distance < 0.45:
            self.current_target_idx += 1
            self.rotating_to_next = True  # must re-align for next waypoint
            self.get_logger().info(
                f"Waypoint {self.current_target_idx} reached, "
                f"{len(self.pruned_path) - self.current_target_idx} remaining"
            )
            return

        cmd = Twist()

        # STATE 1: ROTATE - align precisely with next waypoint
        if self.rotating_to_next:
            if abs(angle_error) > 0.03:
                # rotate slowly and precisely
                cmd.angular.z = 1.0 * angle_error
                cmd.angular.z = max(-1.5, min(1.5, cmd.angular.z))
                cmd.linear.x  = 0.0
            else:
                # perfectly aligned - switch to drive and record segment start
                self.rotating_to_next  = False
                self.segment_start_x   = self.robot_x
                self.segment_start_y   = self.robot_y
                self.get_logger().info(f"Aligned! Driving straight to waypoint {self.current_target_idx}")

        # STATE 2: DRIVE - go straight with angular.z = 0.0
        else:
            # check lateral drift from intended straight line
            drift = self.calc_lateral_drift(target_x, target_y)

            if drift > 0.4:
                # drifted too far off the line - stop and re-rotate
                self.rotating_to_next = True
                self.get_logger().info(f"Drift={drift:.2f}m - stopping to re-align")
                cmd.linear.x  = 0.0
                cmd.angular.z = 0.0
            else:
                # still on track - drive straight
                if distance > 1.0:
                    cmd.linear.x = 0.3
                elif distance > 0.5:
                    cmd.linear.x = 0.2
                else:
                    cmd.linear.x = 0.1
                cmd.angular.z = 0.0  # exactly zero - pure straight line

        self.vel_pub.publish(cmd)

    def stop_robot(self):
        """
        Sends zero velocity to the robot to bring it to a complete stop.
        Only called when the mission is fully complete (all waypoints reached).
        """
        cmd           = Twist()
        cmd.linear.x  = 0.0   # no forward motion
        cmd.angular.z = 0.0   # no rotation
        self.vel_pub.publish(cmd)


    def run_planner(self):
        """
        Main planning sequence - runs in a background thread.
        
        Steps:
        1. Run A* from start to goal -> get raw path (green line)
        2. Prune the path using Line of Sight -> get simplified path (blue line)
        3. Publish both paths to RViz for visualization
        4. Start the controller to drive the robot along the pruned path
        """
        # Step 1: find path using A*
        self.raw_path = self.run_astar(
            (self.start_x, self.start_y),
            (self.goal_x,  self.goal_y)
        )

        if self.raw_path:
            # Step 2: simplify the path
            self.pruned_path = self.prune_path(self.raw_path)
            self.get_logger().info(
                f"Planning complete! Raw={len(self.raw_path)} Pruned={len(self.pruned_path)} waypoints."
            )
            # Step 3: show both paths in RViz
            self.publish_paths()
            # Step 4: start driving the robot
            self.start_controller()
        else:
            self.get_logger().error("Planning failed!")


def main(args=None):
    rclpy.init(args=args)       # initialize the ROS 2 system
    node = PathPlanner()        # create our planner node (this runs __init__)
    rclpy.spin(node)            # keep the node alive, processing callbacks continuously
    rclpy.shutdown()            # clean up when the node is stopped


if __name__ == "__main__":
    main()
