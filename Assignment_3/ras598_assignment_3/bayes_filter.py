import rclpy
# rclpy is the ROS 2 Python library - without this we cannot create nodes or use ROS at all

from rclpy.node import Node
# Node is the base class for all ROS 2 nodes - our BayesFilter3D class inherits from it

from nav_msgs.msg import Odometry, Path, OccupancyGrid
# Odometry - message type for robot position and velocity from /odom and /ground_truth
# Path - message type for drawing lines in RViz (green GT path and red odom path)
# OccupancyGrid - message type for the 2D probability heatmap shown in RViz

from marker_msgs.msg import MarkerDetection
# MarkerDetection - message type from /fiducials topic, tells us when robot sees a landmark

from geometry_msgs.msg import PoseStamped
# PoseStamped - a single position + orientation + timestamp, used to build up Path messages

from visualization_msgs.msg import Marker, MarkerArray
# Marker - a single visual object in RViz (cylinder, text, sphere etc)
# MarkerArray - a collection of Markers published together (all landmarks at once)

import numpy as np
# numpy is our main math library - we use it for the 3D belief array and all calculations

from scipy.ndimage import gaussian_filter
# gaussian_filter blurs our belief array after each motion step
# this spreads uncertainty because odometry is noisy - we become less certain over time

from tf_transformations import euler_from_quaternion
# ROS stores orientation as a quaternion (4 numbers: x, y, z, w)
# euler_from_quaternion converts it to roll, pitch, yaw - we only need yaw (heading angle)

import re
# re is the regular expression library - we use it to parse the cave.world text file
# to extract landmark positions

import os
# os lets us check if files exist and expand paths like ~/ros_ws to full paths


class BayesFilter3D(Node):
    """
    This is the main Bayes Filter node for Assignment 3.
    
    The core idea: instead of knowing exactly where the robot is,
    we maintain a 3D grid where every cell holds a probability.
    - The first two dimensions (x, y) represent position in the world
    - The third dimension (theta) represents which direction the robot is facing
    - Each cell value = probability that the robot is at that position facing that direction
    
    We update this grid in two ways:
    1. PREDICT step: when robot moves, shift the probability cloud to match
    2. UPDATE step: when robot sees a landmark, sharpen the cloud using sensor data
    
    Over time the cloud converges to show the robots true location.
    This is called a Histogram Filter or Discrete Bayes Filter.
    """

    def __init__(self, world_file_path):
        super().__init__("bayes_filter_3d_node")
        # super().__init__ calls the Node class constructor
        # "bayes_filter_3d_node" is the name this node shows up as in ROS

        # -----------------------------------------------------------------------
        # GRID CONFIGURATION
        # -----------------------------------------------------------------------

        self.world_size = 16.0
        # the cave world spans 16 meters in both x and y directions
        # it goes from -8m to +8m (centered at origin)

        self.resolution = 0.2
        # each grid cell represents a 0.2m x 0.2m square of real world space
        # finer resolution = more accurate but more memory and computation

        self.theta_res = 10
        # each angle bin covers 10 degrees of orientation
        # so we have 360/10 = 36 bins to cover all possible headings

        self.grid_dim = int(self.world_size / self.resolution)
        # grid_dim = 16.0 / 0.2 = 80
        # we have an 80x80 grid of spatial cells

        self.theta_dim = int(360 / self.theta_res)
        # theta_dim = 360 / 10 = 36
        # we have 36 orientation bins

        # so our full belief array shape is (80, 80, 36) = 230,400 cells total

        # -----------------------------------------------------------------------
        # ROS PUBLISHERS - these send data out to RViz and other nodes
        # -----------------------------------------------------------------------

        self.costmap_pub = self.create_publisher(OccupancyGrid, "viz/belief_costmap", 10)
        # publishes the 2D probability heatmap to RViz
        # topic name: viz/belief_costmap
        # queue size 10 means ROS keeps up to 10 messages buffered

        self.landmark_pub = self.create_publisher(MarkerArray, "viz/landmarks", 10)
        # publishes red cylinder markers showing where each landmark is in the world
        # topic name: viz/landmarks

        self.gt_path_pub = self.create_publisher(Path, "viz/gt_path", 10)
        # publishes the green line showing the robots TRUE path from ground truth
        # topic name: viz/gt_path

        self.odom_path_pub = self.create_publisher(Path, "viz/odom_path", 10)
        # publishes the red line showing where ODOMETRY thinks the robot went
        # this drifts over time showing why we need the Bayes filter
        # topic name: viz/odom_path

        # -----------------------------------------------------------------------
        # PATH MESSAGE INITIALIZATION
        # -----------------------------------------------------------------------

        self.gt_path_msg = Path()
        # create an empty Path message for ground truth
        # we will keep appending poses to this as the robot moves

        self.gt_path_msg.header.frame_id = "map"
        # tell RViz these coordinates are in the "map" frame (world coordinates)

        self.odom_path_msg = Path()
        # create an empty Path message for odometry path

        self.odom_path_msg.header.frame_id = "map"
        # same frame as above

        # -----------------------------------------------------------------------
        # LANDMARK MAP LOADING
        # -----------------------------------------------------------------------

        self.landmarks = self._parse_world_file(world_file_path)
        # reads the cave.world file and extracts all landmark positions
        # result is a dictionary: {landmark_id: (x, y)}
        # example: {10: (-5.0, -5.0), 20: (-5.0, 1.0), ...}

        # -----------------------------------------------------------------------
        # ROBOT INITIAL STATE
        # -----------------------------------------------------------------------

        self.initial_pose = [-7.0, -7.0, 90.0]
        # the robot always starts at x=-7, y=-7, facing 90 degrees (north/up)
        # this is given by the assignment - DO NOT CHANGE

        self.odom_x = self.initial_pose[0]
        # initialize odometry tracking x position to the starting x = -7.0

        self.odom_y = self.initial_pose[1]
        # initialize odometry tracking y position to the starting y = -7.0
        # Stage simulator odometry starts at (0,0) internally
        # we add the initial pose offset in odom_callback to get world coordinates

        # -----------------------------------------------------------------------
        # BELIEF INITIALIZATION
        # -----------------------------------------------------------------------

        self.initialize_belief(pose=self.initial_pose)
        # create the 3D belief array and set the starting probability
        # since we know the starting pose, we put all probability in that one cell

        self.last_odom_pose = None
        # stores the previous odometry message so we can calculate how much we moved
        # starts as None because we have no previous message yet

        # -----------------------------------------------------------------------
        # ROS SUBSCRIPTIONS - these receive data from the simulator
        # -----------------------------------------------------------------------

        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        # subscribe to /odom topic
        # every time a new odometry message arrives, call self.odom_callback
        # /odom gives NOISY position data - this is what a real robot would have

        self.create_subscription(Odometry, "/ground_truth", self.gt_callback, 10)
        # subscribe to /ground_truth topic
        # gives PERFECT position data from the simulator
        # used ONLY for visualization and performance comparison - never for filter calculations

        self.create_subscription(MarkerDetection, "/fiducials", self.fiducial_callback, 10)
        # subscribe to /fiducials topic
        # called when the robot sees one or more landmarks
        # this triggers the measurement update step

        self.create_timer(1.0, self._publish_landmarks)
        # call _publish_landmarks every 1.0 seconds
        # this refreshes the landmark cylinders in RViz so they stay visible

        # -----------------------------------------------------------------------
        # STARTUP LOG - print landmark positions to verify they loaded correctly
        # -----------------------------------------------------------------------

        self.get_logger().info("--- Landmark Locations ---")
        for tid, pos in self.landmarks.items():
            lx, ly = pos
            self.get_logger().info(f"ID {tid}: x={lx:.2f}, y={ly:.2f}")
        self.get_logger().info("---------------------------")

    # ---------------------------------------------------------------------------
    # UTILITY AND VISUALIZATION FUNCTIONS
    # ---------------------------------------------------------------------------

    def _parse_world_file(self, path):
        """
        Reads the Stage cave.world file and extracts the position of every landmark.
        
        The world file is a text file that describes the simulation environment.
        Each landmark appears as a block like this:
        
            my_block (
                pose [ -5.0 -5.0 0 0 ]
                fiducial_return 10
            )
        
        We use regular expressions (regex) to find these blocks and extract
        the x,y position and the ID number from each one.
        
        Returns a dictionary mapping ID to (x, y) position.
        """
        found = {}
        # start with empty dictionary - will be filled as we find landmarks

        if not os.path.exists(path):
            return found
        # if the world file doesnt exist just return empty - avoids a crash

        with open(path, "r") as f:
            content = f.read()
        # read the entire world file as one big string

        block_pattern = re.compile(r"my_block\s*\((.*?)\)", re.DOTALL)
        # regex pattern to find each my_block(...) section in the file
        # \s* means zero or more whitespace characters
        # (.*?) captures everything inside the parentheses (non-greedy)
        # re.DOTALL means . matches newlines too (blocks span multiple lines)

        pose_pattern = re.compile(r"pose\s*\[\s*([-\d.]+)\s+([-\d.]+)")
        # regex to extract x and y from pose [ x y 0 0 ]
        # [-\d.]+ matches numbers including negative sign and decimal point

        id_pattern = re.compile(r"fiducial_return\s+(\d+)")
        # regex to extract the landmark ID number from fiducial_return 10

        for block_content in block_pattern.findall(content):
            # loop through each my_block section found in the file
            p_match = pose_pattern.search(block_content)
            id_match = id_pattern.search(block_content)

            if p_match and id_match:
                # only add this landmark if we found both a position and an ID
                found[int(id_match.group(1))] = (float(p_match.group(1)), float(p_match.group(2)))
                # group(1) = first captured group = the ID number or x coordinate
                # group(2) = second captured group = the y coordinate

        return found

    def _publish_landmarks(self):
        """
        Sends red cylinder markers to RViz so we can see where each landmark is.
        Also adds a white text label above each cylinder showing the ID number.
        This is called every second by the timer to keep the markers visible in RViz.
        """
        ma = MarkerArray()
        # create an empty MarkerArray - we will add one marker per landmark

        for tid, (tx, ty) in self.landmarks.items():
            # loop through each landmark in our map database

            # --- RED CYLINDER MARKER ---
            c = Marker()
            # create a new marker object

            c.header.frame_id = "map"
            # this marker lives in the map coordinate frame (world coordinates)

            c.id = int(tid)
            # unique ID for this marker - must be unique across all markers
            # we use the landmark ID directly (10, 20, 30, 40, 50)

            c.type = Marker.CYLINDER
            # shape of this marker = cylinder (like a pillar)

            c.action = Marker.ADD
            # ADD means create or update this marker in RViz

            c.pose.position.x = float(tx)
            # x position of the cylinder center in world coordinates

            c.pose.position.y = float(ty)
            # y position of the cylinder center in world coordinates

            c.pose.position.z = 0.5
            # z position = 0.5m above the ground (center of a 1m tall cylinder)

            c.pose.orientation.w = 1.0
            # no rotation - w=1.0 means identity quaternion (upright cylinder)

            c.scale.x = 0.3
            # cylinder diameter in x direction = 0.3m

            c.scale.y = 0.3
            # cylinder diameter in y direction = 0.3m (same as x = round cylinder)

            c.scale.z = 1.0
            # cylinder height = 1.0m tall

            c.color.r = 1.0
            # red color component = maximum (1.0)

            c.color.g = 0.0
            # green color component = zero

            c.color.b = 0.0
            # blue color component = zero
            # so the cylinder is pure red

            c.color.a = 1.0
            # alpha (transparency) = 1.0 means fully opaque (not see-through)

            ma.markers.append(c)
            # add this cylinder to the marker array

            # --- WHITE TEXT LABEL MARKER ---
            t = Marker()
            # create another marker for the text label

            t.header.frame_id = "map"
            # same coordinate frame as the cylinder

            t.id = int(tid) + 1000
            # text marker ID must be different from cylinder ID
            # adding 1000 ensures no clash (cylinder=10, text=1010 etc)

            t.type = Marker.TEXT_VIEW_FACING
            # this marker always faces the camera - useful for text labels

            t.action = Marker.ADD
            # create or update this marker

            t.text = f"ID: {tid}"
            # the text to display above the cylinder

            t.pose.position.x = float(tx)
            # same x as the cylinder

            t.pose.position.y = float(ty) + 0.5
            # slightly offset in y so text doesnt overlap the cylinder

            t.pose.position.z = 1.2
            # 1.2m above ground - just above the top of the cylinder

            t.pose.orientation.w = 1.0
            # no rotation needed for text markers

            t.scale.z = 0.4
            # text height = 0.4m (controls font size in RViz)

            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            # white text (all color components = 1.0)

            t.color.a = 1.0
            # fully opaque

            ma.markers.append(t)
            # add the text label to the marker array

        self.landmark_pub.publish(ma)
        # publish the entire marker array to RViz in one message

    def _publish_costmap(self):
        """
        Projects the 3D belief grid down to 2D and publishes it to RViz as a heatmap.
        
        We cannot display a 3D array directly in RViz.
        So we collapse it to 2D by summing over the angle dimension.
        This gives us a bird-eye view where bright/dark cells show where the robot probably is.
        
        The OccupancyGrid message expects values from 0 to 100.
        We normalize our belief values to this range.
        """
        if not hasattr(self, "belief"):
            return
        # safety check - dont try to publish if belief array doesnt exist yet

        grid = OccupancyGrid()
        # create a new OccupancyGrid message

        grid.header.frame_id = "map"
        # this grid lives in the map coordinate frame

        grid.header.stamp = self.get_clock().now().to_msg()
        # timestamp = current ROS time - tells RViz when this data was created

        grid.info.resolution = self.resolution
        # each cell in the grid = 0.2m x 0.2m in real world

        grid.info.width = self.grid_dim
        # grid is 80 cells wide

        grid.info.height = self.grid_dim
        # grid is 80 cells tall

        grid.info.origin.position.x = -8.0
        # the bottom-left corner of the grid is at x=-8.0 in world coordinates

        grid.info.origin.position.y = -8.0
        # the bottom-left corner of the grid is at y=-8.0 in world coordinates

        grid.info.origin.orientation.w = 1.0
        # no rotation - grid is aligned with world axes

        belief_2d = np.sum(self.belief, axis=2)
        # sum over the angle axis (axis=2) to collapse 3D to 2D
        # result shape: (80, 80) - for each x,y cell, total probability across all angles
        # bright spots = robot is probably somewhere near here (regardless of heading)

        belief_t = belief_2d.T
        # transpose the array from [x, y] to [y, x]
        # our belief array uses [ix, iy] = [x_index, y_index]
        # but ROS OccupancyGrid expects [row, col] = [y_index, x_index]
        # transposing swaps the axes to match ROS convention

        max_val = np.max(belief_t)
        # find the maximum probability value in the 2D map

        if max_val > 0:
            data = (belief_t / max_val * 100).astype(np.int8)
            # scale all values so the maximum becomes 100
            # OccupancyGrid expects integers from 0 to 100
            # 100 = most likely location, 0 = least likely
        else:
            data = np.zeros_like(belief_t, dtype=np.int8)
            # if everything is zero just publish a blank map

        grid.data = data.flatten().tolist()
        # flatten 2D array to 1D list - OccupancyGrid stores data as a flat list
        # row by row, left to right

        self.costmap_pub.publish(grid)
        # send the costmap to RViz

    # ---------------------------------------------------------------------------
    # ASSIGNMENT TASKS
    # ---------------------------------------------------------------------------

    def gt_callback(self, msg):
        """
        Called every time /ground_truth publishes a new robot position.
        We ONLY use this to draw the green path line in RViz.
        The ground truth position is NEVER used in the filter calculations.
        Using ground truth in the filter would be cheating - a real robot doesnt have it.
        """
        p = PoseStamped()
        # create a new pose message for this position

        p.header.frame_id = "map"
        # this pose is in the map (world) coordinate frame

        p.header.stamp = msg.header.stamp
        # copy the timestamp from the ground truth message

        p.pose.position.x = msg.pose.pose.position.x
        # copy the x position from ground truth

        p.pose.position.y = msg.pose.pose.position.y
        # copy the y position from ground truth

        p.pose.orientation.w = 1.0
        # no rotation needed - we only care about position for the path line

        self.gt_path_msg.poses.append(p)
        # add this pose to our growing list of ground truth positions

        self.gt_path_pub.publish(self.gt_path_msg)
        # publish the entire path so RViz draws the green line up to this point

    def initialize_belief(self, pose=None):
        """
        Creates the 3D belief array and sets the initial probability distribution.
        
        If we know the starting pose (like we do here):
            Put ALL probability (1.0) in the single cell matching that pose.
            This means we are 100% certain where the robot starts.
        
        If we do NOT know the starting pose (kidnapped robot scenario):
            Spread probability equally across ALL cells.
            Every cell gets 1 / total_cells.
            This means we have no idea where the robot is - completely uncertain.
        
        The belief must always sum to 1.0 - it is a probability distribution.
        """
        self.belief = np.zeros((self.grid_dim, self.grid_dim, self.theta_dim))
        # create a 3D array of zeros with shape (80, 80, 36)
        # all zeros means zero probability everywhere to start

        if pose is not None:
            # we know the starting pose - put all probability there
            ix, iy, ith = self.real_to_grid(pose[0], pose[1], pose[2])
            # convert the starting pose from world coordinates to grid indices

            self.belief[ix, iy, ith] = 1.0
            # set that single cell to probability 1.0 (100% certain)
            # all other cells remain 0.0

        else:
            # we dont know where the robot is - uniform distribution
            self.belief[:] = 1.0 / (self.grid_dim * self.grid_dim * self.theta_dim)
            # every cell gets equal probability
            # 1 / (80 * 80 * 36) = 1 / 230400 = very small number
            # but they all sum to 1.0

    def real_to_grid(self, x, y, theta_deg):
        """
        Converts real world coordinates (in meters and degrees) to grid indices.
        
        The world coordinate system:
            x goes from -8.0m (left) to +8.0m (right)
            y goes from -8.0m (bottom) to +8.0m (top)
            theta goes from 0 to 360 degrees
        
        The grid index system:
            ix goes from 0 (left) to 79 (right)
            iy goes from 0 (bottom) to 79 (top)
            ith goes from 0 to 35 (36 angle bins)
        
        Formula: index = (world_coordinate + 8.0) / resolution
        
        Examples:
            x=-7.0 -> ix = (-7.0 + 8.0) / 0.2 = 1.0 / 0.2 = 5
            x= 0.0 -> ix = ( 0.0 + 8.0) / 0.2 = 8.0 / 0.2 = 40
            x=+7.0 -> ix = ( 7.0 + 8.0) / 0.2 = 15.0 / 0.2 = 75
            theta=90 -> ith = 90 / 10 = 9
        """
        ix = int((x + 8.0) / self.resolution)
        # shift x from (-8 to +8) range to (0 to 16) range, then divide by cell size
        # int() truncates to integer grid index

        iy = int((y + 8.0) / self.resolution)
        # same conversion for y coordinate

        ith = int(theta_deg % 360 / self.theta_res)
        # theta_deg % 360 wraps the angle to 0-360 range first
        # then divide by 10 to get bin index (0 to 35)

        ix = int(np.clip(ix, 0, self.grid_dim - 1))
        # clamp ix to valid range [0, 79] to prevent going out of array bounds
        # np.clip(value, min, max) keeps value within [min, max]

        iy = int(np.clip(iy, 0, self.grid_dim - 1))
        # clamp iy to valid range [0, 79]

        ith = int(np.clip(ith, 0, self.theta_dim - 1))
        # clamp ith to valid range [0, 35]

        return ix, iy, ith
        # return all three grid indices

    def grid_to_real(self):
        """
        The reverse of real_to_grid - converts grid indices to real world coordinates.
        
        Instead of converting a single point, this function converts the ENTIRE grid at once.
        It returns three 3D numpy arrays (rx, ry, rth), each the same shape as self.belief (80,80,36).
        
        rx[i,j,k] = real world x coordinate of grid cell (i,j,k)
        ry[i,j,k] = real world y coordinate of grid cell (i,j,k)
        rth[i,j,k] = real world angle of grid cell (i,j,k)
        
        We use this in the measurement model so we can calculate the expected
        range and bearing for ALL cells simultaneously using numpy operations.
        This is much faster than looping through all 230,400 cells one by one.
        """
        ix = np.arange(self.grid_dim)
        # create array [0, 1, 2, ..., 79] for x indices

        iy = np.arange(self.grid_dim)
        # create array [0, 1, 2, ..., 79] for y indices

        ith = np.arange(self.theta_dim)
        # create array [0, 1, 2, ..., 35] for angle indices

        gx, gy, gth = np.meshgrid(ix, iy, ith, indexing="ij")
        # meshgrid creates 3D arrays covering all combinations of (ix, iy, ith)
        # indexing="ij" means first axis=x, second=y, third=theta (matrix-style)
        # gx[i,j,k] = i (the x index)
        # gy[i,j,k] = j (the y index)
        # gth[i,j,k] = k (the angle index)

        rx = gx * self.resolution - 8.0
        # convert x index to real world x: multiply by 0.2 then shift by -8
        # example: index 5 -> 5 * 0.2 - 8.0 = 1.0 - 8.0 = -7.0 meters

        ry = gy * self.resolution - 8.0
        # same conversion for y coordinate

        rth = gth * self.theta_res
        # convert angle index to degrees: multiply by 10
        # example: index 9 -> 9 * 10 = 90 degrees

        return rx, ry, rth
        # return three 3D arrays each shaped (80, 80, 36)

    def predict(self, curr_msg, last_msg):
        """
        Motion model - shifts the probability cloud when the robot moves.
        
        This implements the Turn-Go-Turn odometry model:
        Step 1: Apply first rotation (d_rot1) - shift angle bins
        Step 2: Apply translation (d_trans) - shift x,y position for each angle bin
        Step 3: Apply second rotation (d_rot2) - shift angle bins again
        Step 4: Apply Gaussian blur to spread uncertainty (odometry is noisy)
        Step 5: Normalize so belief still sums to 1
        
        Why Turn-Go-Turn?
        Any movement can be broken into: rotate to face direction, drive straight, rotate to final heading.
        This decomposition makes it easy to update the grid - rotations shift angle bins,
        translation shifts x,y position based on which direction each angle bin is pointing.
        """
        # --- EXTRACT YAW ANGLES ---
        q = curr_msg.pose.pose.orientation
        # get the orientation quaternion from current odometry message

        curr_yaw = np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2]) % 360
        # euler_from_quaternion returns (roll, pitch, yaw) - we take index [2] = yaw
        # np.degrees converts from radians to degrees
        # % 360 wraps to 0-360 range

        q_old = last_msg.pose.pose.orientation
        # get orientation from the previous odometry message

        old_yaw = np.degrees(euler_from_quaternion([q_old.x, q_old.y, q_old.z, q_old.w])[2]) % 360
        # same conversion for the old yaw

        # --- CALCULATE MOVEMENT ---
        dx = curr_msg.pose.pose.position.x - last_msg.pose.pose.position.x
        # how much did x change between last message and current message

        dy = curr_msg.pose.pose.position.y - last_msg.pose.pose.position.y
        # how much did y change

        d_trans = np.sqrt(dx**2 + dy**2)
        # Pythagoras theorem - straight line distance moved
        # sqrt(dx^2 + dy^2) = hypotenuse of the triangle

        d_rot_total = (curr_yaw - old_yaw + 180) % 360 - 180
        # total rotation change between messages
        # the +180 % 360 - 180 trick wraps the result to -180 to +180 range
        # this handles the wraparound case (e.g. going from 350 to 10 degrees = +20 not -340)

        # --- TURN-GO-TURN DECOMPOSITION ---
        d_rot1 = d_rot_total / 2.0
        # split total rotation evenly - first half happens before translation

        d_rot2 = d_rot_total / 2.0
        # second half happens after translation

        # --- STEP 1: FIRST ROTATION ---
        angle_shift = int(round(d_rot1 / self.theta_res))
        # convert rotation in degrees to number of angle bins to shift
        # example: d_rot1 = 15 degrees, theta_res = 10 -> shift 2 bins (rounds 1.5 to 2)

        self.belief = np.roll(self.belief, angle_shift, axis=2)
        # np.roll shifts the array along axis=2 (the angle dimension)
        # positive shift = rotate counter-clockwise (higher angle bins)
        # negative shift = rotate clockwise (lower angle bins)
        # elements that roll off one end reappear at the other end (wraps around 360 degrees)

        # --- STEP 2: TRANSLATION ---
        trans_cells = d_trans / self.resolution
        # convert distance in meters to number of grid cells
        # example: d_trans = 0.4m, resolution = 0.2m -> 2 cells

        new_belief = np.zeros_like(self.belief)
        # create a blank array to hold the shifted belief
        # we need a separate array because we cant shift in-place

        for ith in range(self.theta_dim):
            # loop through each of the 36 angle bins separately
            # each angle bin points in a different direction so each shifts differently

            angle_rad = np.radians(ith * self.theta_res)
            # convert bin index to actual angle in radians
            # example: bin 9 -> 9 * 10 = 90 degrees -> pi/2 radians

            shift_x = int(round(trans_cells * np.cos(angle_rad)))
            # x component of movement for this angle
            # if facing right (0 deg): cos(0) = 1.0 -> shift fully in x
            # if facing up (90 deg): cos(90) = 0.0 -> no x shift

            shift_y = int(round(trans_cells * np.sin(angle_rad)))
            # y component of movement for this angle
            # if facing right (0 deg): sin(0) = 0.0 -> no y shift
            # if facing up (90 deg): sin(90) = 1.0 -> shift fully in y

            new_belief[:, :, ith] = np.roll(
                np.roll(self.belief[:, :, ith], shift_x, axis=0),
                shift_y, axis=1
            )
            # take the 2D slice for this angle bin (shape 80x80)
            # shift it shift_x cells along x axis (axis=0)
            # then shift it shift_y cells along y axis (axis=1)
            # store result in new_belief at same angle bin

        self.belief = new_belief
        # replace old belief with the translated version

        # --- STEP 3: SECOND ROTATION ---
        angle_shift2 = int(round(d_rot2 / self.theta_res))
        # same as before - convert rotation to bin shift

        self.belief = np.roll(self.belief, angle_shift2, axis=2)
        # apply second rotation shift to angle dimension

        # --- STEP 4: GAUSSIAN BLUR ---
        self.belief = gaussian_filter(self.belief, sigma=[0.8, 0.8, 0.5])
        # blur the belief to model odometry noise and uncertainty
        # sigma=[0.8, 0.8, 0.5] controls how much blurring in each dimension
        # sigma=0.8 in x and y means the uncertainty spreads about 0.8 cells = 0.16m
        # sigma=0.5 in theta means angle uncertainty spreads about 0.5 bins = 5 degrees
        # larger sigma = more uncertain = more spread out cloud
        # these values were tuned to match the observed odometry noise in Stage

        # --- STEP 5: NORMALIZE ---
        total = np.sum(self.belief) + 1e-300
        # sum all values in the belief array
        # add 1e-300 (a tiny number close to zero) to prevent division by zero
        # this handles the edge case where the entire belief collapses to zero

        self.belief /= total
        # divide every cell by the total so all cells sum to 1.0 again
        # this is required because a probability distribution must always sum to 1

    def update_measurement(self, landmark_x, landmark_y, measured_range, measured_bearing):
        """
        Measurement model - sharpens the belief when the robot sees a landmark.
        
        For each of the 230,400 cells in the belief grid, we ask:
        IF the robot were at this cell (this x, y, facing this theta),
        WHAT range and bearing would it measure to this landmark?
        
        We then compare this EXPECTED measurement to the ACTUAL sensor reading.
        Using a Gaussian (bell curve) scoring function:
        - Perfect match (expected = measured): score = 1.0 (maximum)
        - Small error: score close to 1.0
        - Large error: score close to 0.0
        
        We multiply the current belief by these scores (Bayes Rule):
        new_belief = old_belief * likelihood
        
        Cells that match the sensor reading keep their probability.
        Cells that dont match get their probability reduced toward zero.
        
        Finally we normalize so everything still sums to 1.
        
        This causes the probability cloud to sharpen and concentrate
        around cells that are consistent with what the sensor reported.
        """
        rx, ry, rth = self.grid_to_real()
        # get real world x, y, theta for every cell in the grid
        # these are 3D arrays each shaped (80, 80, 36)
        # rx[i,j,k] = x coordinate of cell (i,j,k) in meters

        # --- CALCULATE EXPECTED RANGE ---
        exp_range = np.sqrt((landmark_x - rx)**2 + (landmark_y - ry)**2)
        # for every cell, calculate Euclidean distance to the landmark
        # if robot were at (rx, ry), this is how far the landmark would be
        # numpy does this calculation for all 230,400 cells simultaneously

        # --- CALCULATE EXPECTED BEARING ---
        angle_to_landmark = np.degrees(np.arctan2(landmark_y - ry, landmark_x - rx))
        # arctan2(dy, dx) gives the absolute angle from cell to landmark in world frame
        # np.degrees converts from radians to degrees

        exp_bearing = (angle_to_landmark - rth + 180) % 360 - 180
        # subtract the cells heading angle (rth) to get RELATIVE bearing
        # a robot facing north (90 deg) looking at a landmark directly ahead
        # would report bearing = 0 (not 90)
        # the +180 % 360 - 180 wraps result to -180 to +180 range

        # --- GAUSSIAN SCORING ---
        sigma_range = 1.0
        # range tolerance = 1.0 meter
        # cells within 1m of the correct range get a high score
        # cells more than 2-3m off get a very low score

        sigma_bearing = 20.0
        # bearing tolerance = 20 degrees
        # cells within 20 degrees of the correct bearing get a high score

        range_score = np.exp(-0.5 * ((measured_range - exp_range) / sigma_range) ** 2)
        # Gaussian bell curve formula: exp(-0.5 * (error/sigma)^2)
        # when error=0: exp(0) = 1.0 (perfect match, maximum score)
        # when error=sigma: exp(-0.5) = 0.607 (one sigma off, decent score)
        # when error=2*sigma: exp(-2) = 0.135 (two sigma off, low score)
        # when error is very large: exp(-huge) ≈ 0.0 (terrible match, zero score)

        bearing_score = np.exp(-0.5 * ((measured_bearing - exp_bearing) / sigma_bearing) ** 2)
        # same Gaussian formula applied to bearing error

        likelihood = range_score * bearing_score
        # combined score = range score * bearing score
        # BOTH range AND bearing must match to get a high likelihood
        # if either one is wrong the combined score drops toward zero
        # this is what collapses the ring into a point - range alone gives a ring,
        # but range AND bearing together give only one (or a few) possible locations

        # --- BAYES UPDATE ---
        self.belief = self.belief * likelihood
        # multiply every cell in belief by its likelihood score
        # this is the Bayes rule: posterior = prior * likelihood (unnormalized)
        # cells that match the measurement keep their probability
        # cells that dont match get multiplied by a small number (suppressed)

        # --- NORMALIZE ---
        total = np.sum(self.belief) + 1e-300
        # sum all updated belief values
        # add epsilon to prevent division by zero

        if total > 1e-10:
            self.belief /= total
            # divide by total to make everything sum to 1 again
        else:
            self.belief[:] = 1.0 / self.belief.size
            # if belief completely collapsed to zero (shouldnt happen but just in case)
            # reset to uniform distribution - like a kidnapped robot starting over

    def odom_callback(self, msg):
        """
        Called every time /odom publishes a new odometry message.
        
        Does two things:
        1. Updates and publishes the red odometry path for RViz visualization
        2. Runs the prediction step if the robot moved enough
        
        Stage simulator note:
        - /odom gives position relative to where the robot started (0,0)
        - /ground_truth gives absolute world coordinates
        - Stage odom.x corresponds to world Y displacement
        - Stage odom.y corresponds to world X displacement
        - We add initial_pose to convert odom to world coordinates for visualization
        
        We only run predict if robot moved > 1mm or rotated > 0.1 degrees
        to avoid wasting computation when robot is stationary.
        """
        if self.last_odom_pose is None:
            self.last_odom_pose = msg
            return
        # first time this runs we have no previous message to compare to
        # so just save this message and return - we need two messages to calculate movement

        # --- EXTRACT YAW ANGLES ---
        q = msg.pose.pose.orientation
        curr_yaw_deg = np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2]) % 360
        # current heading angle in degrees

        q_old = self.last_odom_pose.pose.pose.orientation
        old_yaw_deg = np.degrees(euler_from_quaternion([q_old.x, q_old.y, q_old.z, q_old.w])[2]) % 360
        # previous heading angle in degrees

        # --- CALCULATE MOVEMENT SINCE LAST MESSAGE ---
        dx = msg.pose.pose.position.x - self.last_odom_pose.pose.pose.position.x
        # change in odometry x position

        dy = msg.pose.pose.position.y - self.last_odom_pose.pose.pose.position.y
        # change in odometry y position

        dth = (curr_yaw_deg - old_yaw_deg + 180) % 360 - 180
        # change in heading angle, wrapped to -180 to +180 range

        # --- UPDATE ODOMETRY PATH VISUALIZATION ---
        self.odom_x = self.initial_pose[0] + msg.pose.pose.position.y
        # Stage odom.x = world Y displacement, so map odom.x to world y
        # add initial_pose[0] (x=-7) to get absolute world x coordinate

        self.odom_y = self.initial_pose[1] + msg.pose.pose.position.x
        # Stage odom.y = world X displacement, so map odom.y to world x
        # add initial_pose[1] (y=-7) to get absolute world y coordinate

        p = PoseStamped()
        # create a new pose for this odometry position

        p.header.frame_id = "map"
        # in the map coordinate frame

        p.header.stamp = msg.header.stamp
        # use the timestamp from the odometry message

        p.pose.position.x = float(self.odom_x)
        # x position in world coordinates

        p.pose.position.y = float(self.odom_y)
        # y position in world coordinates

        p.pose.orientation.w = 1.0
        # no rotation needed for path visualization

        self.odom_path_msg.poses.append(p)
        # add this position to the growing red path

        self.odom_path_pub.publish(self.odom_path_msg)
        # publish the updated path to RViz

        # --- RUN PREDICTION STEP IF ROBOT MOVED ---
        if np.sqrt(dx**2 + dy**2) > 0.001 or abs(dth) > 0.1:
            # only predict if robot moved more than 1mm OR rotated more than 0.1 degrees
            # this avoids running predict when robot is standing still (saves computation)

            self.predict(msg, self.last_odom_pose)
            # run the motion model to shift the belief cloud

            self.last_odom_pose = msg
            # update the stored previous message for next time

            self._publish_costmap()
            # publish the updated belief heatmap to RViz

    def fiducial_callback(self, msg):
        """
        Called every time /fiducials publishes a landmark detection message.
        
        The robot can see multiple landmarks at once (multiple pink dotted lines in Stage).
        We loop through all detected landmarks and run a measurement update for each one.
        
        Each update sharpens the belief further:
        - 1 landmark seen: ring of probability (range constrains location)
        - 2 landmarks seen: intersection of two rings (fewer possible locations)  
        - 3 landmarks seen: very sharp peak (nearly certain location)
        
        After processing all landmarks we publish the updated costmap to RViz.
        """
        for marker in msg.markers:
            # loop through each landmark detected in this message

            marker_id = int(marker.ids[0])
            # get the landmark ID - marker.ids is an array so we take index [0]
            # int() converts it from array element to a plain integer

            if marker_id not in self.landmarks:
                continue
            # skip this landmark if we dont have it in our map database
            # this shouldnt happen but is a safety check

            lx, ly = self.landmarks[marker_id]
            # look up the known WORLD position of this landmark from our map
            # this is the ground truth position we loaded from cave.world

            mx = float(marker.pose.position.x)
            # x position of landmark RELATIVE TO THE ROBOT from the sensor
            # this is what the sensor actually measured

            my = float(marker.pose.position.y)
            # y position of landmark relative to robot from sensor

            measured_range = np.sqrt(mx**2 + my**2)
            # calculate actual distance from robot to landmark
            # using Pythagoras: range = sqrt(mx^2 + my^2)

            measured_bearing = np.degrees(np.arctan2(my, mx))
            # calculate actual angle to landmark relative to robot heading
            # arctan2(y, x) gives angle in radians, np.degrees converts to degrees

            self.update_measurement(lx, ly, measured_range, measured_bearing)
            # run the measurement update using:
            # - landmark world position (lx, ly) - from our map
            # - measured range and bearing - from the sensor

        self._publish_costmap()
        # publish the updated belief heatmap after processing all landmarks


def main():
    rclpy.init()
    # initialize the ROS 2 communication system - must be called before anything else

    world_path = os.path.expanduser("~/ros_ws/src/stage_ros2/world/cave.world")
    # build the full path to the cave world file
    # os.path.expanduser replaces ~ with the actual home directory path

    if not os.path.exists(world_path):
        print("ERROR: World file not found at: " + world_path)
        return
    # check the file exists before trying to start
    # if it doesnt exist we print an error and exit gracefully

    node = BayesFilter3D(world_path)
    # create our Bayes filter node - this runs __init__ which sets everything up

    try:
        rclpy.spin(node)
        # keep the node alive and processing incoming messages
        # spin() blocks here and calls our callbacks whenever messages arrive

    except KeyboardInterrupt:
        pass
        # when user presses Ctrl+C, catch the interrupt and exit cleanly

    finally:
        rclpy.shutdown()
        # always clean up ROS 2 resources when done, even if there was an error


if __name__ == "__main__":
    main()
    # only run main() if this script is run directly (not imported as a module)
