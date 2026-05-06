import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from marker_msgs.msg import MarkerDetection
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy.ndimage import gaussian_filter
from tf_transformations import euler_from_quaternion
import re
import os


class BayesFilter3D(Node):
    """
    This is the main Bayes Filter node for Assignment 3.
    Instead of knowing exactly where the robot is, we maintain a 3D grid
    where every cell holds a probability - how likely is the robot HERE, facing THIS direction.
    We update this grid every time the robot moves (predict step)
    and every time it sees a landmark (update step).
    Over time the grid converges to show the robots true location.
    """

    def __init__(self, world_file_path):
        super().__init__("bayes_filter_3d_node")

        # --- GRID SETTINGS ---
        # the cave world is 16m x 16m, going from -8m to +8m in both x and y
        self.world_size = 16.0

        # each cell in the grid covers 0.2m x 0.2m of real world space
        self.resolution = 0.2

        # each angle bin covers 10 degrees, so we have 36 bins to cover all 360 degrees
        self.theta_res = 10

        # how many cells in each spatial direction: 16.0 / 0.2 = 80 cells
        self.grid_dim = int(self.world_size / self.resolution)

        # how many angle bins: 360 / 10 = 36 bins
        self.theta_dim = int(360 / self.theta_res)

        # --- ROS PUBLISHERS ---
        # publishes the probability map to RViz as a 2D heatmap
        self.costmap_pub = self.create_publisher(OccupancyGrid, "viz/belief_costmap", 10)

        # publishes red cylinders showing landmark positions in RViz
        self.landmark_pub = self.create_publisher(MarkerArray, "viz/landmarks", 10)

        # publishes the green ground truth path line in RViz
        self.gt_path_pub = self.create_publisher(Path, "viz/gt_path", 10)

        # publishes the red raw odometry path line in RViz
        self.odom_path_pub = self.create_publisher(Path, "viz/odom_path", 10)

        # --- PATH MESSAGE SETUP ---
        # these store all the poses as the robot moves so RViz draws a continuous line
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = "map"
        self.odom_path_msg = Path()
        self.odom_path_msg.header.frame_id = "map"

        # --- LOAD LANDMARKS FROM WORLD FILE ---
        # reads the cave.world file to find where each landmark (fiducial) is located
        self.landmarks = self._parse_world_file(world_file_path)

        # --- ROBOT STARTING POSITION ---
        # the robot always starts at (-7, -7) facing 90 degrees (north) - DO NOT CHANGE
        self.initial_pose = [-7.0, -7.0, 90.0]

        # --- ODOMETRY TRACKING ---
        # we track where the odometry thinks the robot is so we can draw the red path
        # Stage odometry starts at (0,0) so we initialize at the robot starting position
        self.odom_x = self.initial_pose[0]
        self.odom_y = self.initial_pose[1]

        # --- INITIALIZE THE BELIEF GRID ---
        # since we know the starting pose, we put all probability in that one cell
        self.initialize_belief(pose=self.initial_pose)

        # we need to remember the last odometry message to calculate how much we moved
        self.last_odom_pose = None

        # --- ROS SUBSCRIPTIONS ---
        # /odom gives us noisy odometry - used for the prediction step
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        # /ground_truth gives us the real robot position - used ONLY for visualization
        self.create_subscription(Odometry, "/ground_truth", self.gt_callback, 10)

        # /fiducials tells us when the robot sees a landmark - used for the update step
        self.create_subscription(MarkerDetection, "/fiducials", self.fiducial_callback, 10)

        # refresh landmark markers in RViz every second
        self.create_timer(1.0, self._publish_landmarks)

        # print all landmark positions at startup so we can verify they loaded correctly
        self.get_logger().info("--- Landmark Locations ---")
        for tid, pos in self.landmarks.items():
            lx, ly = pos
            self.get_logger().info(f"ID {tid}: x={lx:.2f}, y={ly:.2f}")
        self.get_logger().info("---------------------------")

    # -------------------------------------------------------------------------
    # UTILITY AND VISUALIZATION FUNCTIONS
    # -------------------------------------------------------------------------

    def _parse_world_file(self, path):
        """
        Reads the cave.world file and extracts the x,y position of every landmark.
        Each landmark block in the file has a pose and a fiducial_return ID.
        We store them in a dictionary: {landmark_id: (x, y)}
        """
        found = {}

        # if the file doesnt exist return an empty dictionary
        if not os.path.exists(path):
            return found

        with open(path, "r") as f:
            content = f.read()

        # use regex to find each my_block section in the world file
        block_pattern = re.compile(r"my_block\s*\((.*?)\)", re.DOTALL)

        # extract the pose (x, y position) from each block
        pose_pattern = re.compile(r"pose\s*\[\s*([-\d.]+)\s+([-\d.]+)")

        # extract the fiducial ID from each block
        id_pattern = re.compile(r"fiducial_return\s+(\d+)")

        for block_content in block_pattern.findall(content):
            p_match = pose_pattern.search(block_content)
            id_match = id_pattern.search(block_content)
            if p_match and id_match:
                found[int(id_match.group(1))] = (float(p_match.group(1)), float(p_match.group(2)))

        return found

    def _publish_landmarks(self):
        """
        Sends red cylinder markers to RViz so we can see where each landmark is on the map.
        Also adds a text label above each cylinder showing its ID number.
        Called every second by the timer.
        """
        ma = MarkerArray()

        for tid, (tx, ty) in self.landmarks.items():
            # create a red cylinder at the landmark position
            c = Marker()
            c.header.frame_id = "map"
            c.id = int(tid)
            c.type = Marker.CYLINDER
            c.action = Marker.ADD
            c.pose.position.x = float(tx)
            c.pose.position.y = float(ty)
            c.pose.position.z = 0.5       # center of cylinder at 0.5m height
            c.pose.orientation.w = 1.0    # no rotation
            c.scale.x = 0.3              # cylinder diameter
            c.scale.y = 0.3
            c.scale.z = 1.0              # cylinder height = 1m
            c.color.r = 1.0              # red color
            c.color.g = 0.0
            c.color.b = 0.0
            c.color.a = 1.0              # fully opaque
            ma.markers.append(c)

            # create a white text label floating above the cylinder
            t = Marker()
            t.header.frame_id = "map"
            t.id = int(tid) + 1000       # offset ID so it doesnt clash with cylinder ID
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.text = f"ID: {tid}"
            t.pose.position.x = float(tx)
            t.pose.position.y = float(ty) + 0.5
            t.pose.position.z = 1.2      # above the cylinder
            t.pose.orientation.w = 1.0
            t.scale.z = 0.4              # text height
            t.color.r = 1.0              # white color
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 1.0
            ma.markers.append(t)

        self.landmark_pub.publish(ma)

    def _publish_costmap(self):
        """
        Projects the 3D belief grid down to 2D and sends it to RViz as a heatmap.
        We cannot display a 3D grid directly so we sum over the angle axis.
        This gives us a 2D map where bright areas = robot is probably here.
        Called after every predict and update step.
        """
        # dont publish if belief doesnt exist yet
        if not hasattr(self, "belief"):
            return

        # set up the OccupancyGrid message
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.resolution   # 0.2m per cell
        grid.info.width = self.grid_dim          # 80 cells wide
        grid.info.height = self.grid_dim         # 80 cells tall
        grid.info.origin.position.x = -8.0      # bottom-left corner of the map
        grid.info.origin.position.y = -8.0
        grid.info.origin.orientation.w = 1.0

        # sum over the angle axis (axis=2) to collapse 3D to 2D
        belief_2d = np.sum(self.belief, axis=2)

        # transpose because our array is [x, y] but ROS expects [row, col] = [y, x]
        belief_t = belief_2d.T

        # scale values to 0-100 range as required by OccupancyGrid
        max_val = np.max(belief_t)
        if max_val > 0:
            data = (belief_t / max_val * 100).astype(np.int8)
        else:
            data = np.zeros_like(belief_t, dtype=np.int8)

        # flatten to a 1D list as required by ROS OccupancyGrid format
        grid.data = data.flatten().tolist()
        self.costmap_pub.publish(grid)

    # -------------------------------------------------------------------------
    # ASSIGNMENT TASKS
    # -------------------------------------------------------------------------

    def gt_callback(self, msg):
        """
        Called every time /ground_truth publishes a new robot position.
        We use this ONLY to draw the green path in RViz.
        We never feed ground truth into the filter calculations - that would be cheating.
        """
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = msg.header.stamp
        p.pose.position.x = msg.pose.pose.position.x
        p.pose.position.y = msg.pose.pose.position.y
        p.pose.orientation.w = 1.0
        self.gt_path_msg.poses.append(p)
        self.gt_path_pub.publish(self.gt_path_msg)

    def initialize_belief(self, pose=None):
        """
        Sets up the starting probability distribution across the entire 3D grid.
        If we know the starting pose, put ALL probability in that one cell (certain start).
        If we dont know, spread probability equally everywhere (uniform / kidnapped robot).
        The belief must always sum to 1.0 - this is a probability distribution.
        """
        # create an empty 3D grid: 80 x-cells, 80 y-cells, 36 angle bins
        self.belief = np.zeros((self.grid_dim, self.grid_dim, self.theta_dim))

        if pose is not None:
            # we know where we start - put all probability in that one cell
            ix, iy, ith = self.real_to_grid(pose[0], pose[1], pose[2])
            self.belief[ix, iy, ith] = 1.0
        else:
            # we have no idea where the robot is - spread equally everywhere
            # every cell gets 1 / total_number_of_cells
            self.belief[:] = 1.0 / (self.grid_dim * self.grid_dim * self.theta_dim)

    def real_to_grid(self, x, y, theta_deg):
        """
        Converts real world coordinates (meters) to grid array indices.
        The world goes from -8m to +8m. The grid goes from index 0 to 79.
        Formula: index = (world_coordinate + 8.0) / 0.2
        Example: x=-7.0 -> ix = (-7+8)/0.2 = 5
                 x= 0.0 -> ix = (0+8)/0.2  = 40
                 x=+7.0 -> ix = (7+8)/0.2  = 75
        Also converts angle in degrees to a bin index (0 to 35).
        """
        # shift from world coords (-8 to +8) to positive range (0 to 16) then divide by cell size
        ix = int((x + 8.0) / self.resolution)
        iy = int((y + 8.0) / self.resolution)

        # convert angle to bin index: 0 deg = bin 0, 10 deg = bin 1, 350 deg = bin 35
        ith = int(theta_deg % 360 / self.theta_res)

        # clamp all indices to valid range so we never go out of bounds
        ix = int(np.clip(ix, 0, self.grid_dim - 1))
        iy = int(np.clip(iy, 0, self.grid_dim - 1))
        ith = int(np.clip(ith, 0, self.theta_dim - 1))

        return ix, iy, ith

    def grid_to_real(self):
        """
        For every cell in the 3D grid, calculates its real world x, y, theta values.
        Returns three 3D arrays (rx, ry, rth) all the same shape as self.belief.
        We use this in the measurement model to check every cell at once efficiently.
        This avoids a slow triple for-loop over all 80x80x36 = 230,400 cells.
        """
        # create index arrays for each dimension
        ix = np.arange(self.grid_dim)   # [0, 1, 2, ..., 79]
        iy = np.arange(self.grid_dim)   # [0, 1, 2, ..., 79]
        ith = np.arange(self.theta_dim) # [0, 1, 2, ..., 35]

        # meshgrid creates 3D arrays where every combination of (ix, iy, ith) exists
        # indexing="ij" means first axis = x, second = y, third = theta
        gx, gy, gth = np.meshgrid(ix, iy, ith, indexing="ij")

        # convert indices back to real world values
        rx = gx * self.resolution - 8.0   # x in meters
        ry = gy * self.resolution - 8.0   # y in meters
        rth = gth * self.theta_res         # angle in degrees

        return rx, ry, rth

    def predict(self, curr_msg, last_msg):
        """
        Motion model - updates the belief when the robot moves.
        Uses the Turn-Go-Turn model: first rotate, then translate, then rotate again.
        After shifting the belief cloud, we blur it with a Gaussian filter
        to model the fact that odometry is noisy and we become less certain over time.
        Finally we normalize so the belief still sums to 1.
        """
        # extract yaw angle from the quaternion orientation in both messages
        q = curr_msg.pose.pose.orientation
        curr_yaw = np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2]) % 360

        q_old = last_msg.pose.pose.orientation
        old_yaw = np.degrees(euler_from_quaternion([q_old.x, q_old.y, q_old.z, q_old.w])[2]) % 360

        # calculate how much the robot moved since the last message
        dx = curr_msg.pose.pose.position.x - last_msg.pose.pose.position.x
        dy = curr_msg.pose.pose.position.y - last_msg.pose.pose.position.y
        d_trans = np.sqrt(dx**2 + dy**2)    # straight line distance moved

        # total rotation change, wrapped to -180 to +180 range
        d_rot_total = (curr_yaw - old_yaw + 180) % 360 - 180

        # split total rotation evenly into before and after the translation (Turn-Go-Turn)
        d_rot1 = d_rot_total / 2.0
        d_rot2 = d_rot_total / 2.0

        # --- STEP 1: apply first rotation ---
        # shift all probability in the angle dimension using np.roll
        # positive shift = rotate counter-clockwise, negative = clockwise
        angle_shift = int(round(d_rot1 / self.theta_res))
        self.belief = np.roll(self.belief, angle_shift, axis=2)

        # --- STEP 2: apply translation ---
        # for each angle bin, shift the x-y probability in the direction that angle points
        trans_cells = d_trans / self.resolution   # how many cells to move
        new_belief = np.zeros_like(self.belief)

        for ith in range(self.theta_dim):
            # figure out what direction this angle slice is pointing in radians
            angle_rad = np.radians(ith * self.theta_res)

            # how many cells to shift in x and y for this direction
            shift_x = int(round(trans_cells * np.cos(angle_rad)))
            shift_y = int(round(trans_cells * np.sin(angle_rad)))

            # shift this angle slice using np.roll
            new_belief[:, :, ith] = np.roll(
                np.roll(self.belief[:, :, ith], shift_x, axis=0),
                shift_y, axis=1
            )

        self.belief = new_belief

        # --- STEP 3: apply second rotation ---
        angle_shift2 = int(round(d_rot2 / self.theta_res))
        self.belief = np.roll(self.belief, angle_shift2, axis=2)

        # --- STEP 4: apply Gaussian blur to spread uncertainty ---
        # sigma controls how much we blur: [x_blur, y_blur, angle_blur]
        # larger sigma = more uncertainty = more spread out cloud
        self.belief = gaussian_filter(self.belief, sigma=[0.8, 0.8, 0.5])

        # --- STEP 5: normalize so everything still sums to 1 ---
        # add epsilon (tiny number) to avoid dividing by zero
        total = np.sum(self.belief) + 1e-300
        self.belief /= total

    def update_measurement(self, landmark_x, landmark_y, measured_range, measured_bearing):
        """
        Measurement model - sharpens the belief when the robot sees a landmark.
        For every cell in the grid, we ask: if the robot were HERE facing THIS direction,
        what range and bearing would it measure to this landmark?
        We compare that expected measurement to what the sensor actually reported.
        Cells that match get a high score (boosted). Cells that dont match get a low score (suppressed).
        We use a Gaussian bell curve for scoring - perfect match = score of 1.0, big error = score near 0.
        Then we multiply the entire belief by these scores (Bayes rule) and normalize.
        """
        # get real world x, y, theta for every cell in the grid at once
        rx, ry, rth = self.grid_to_real()

        # calculate expected range from every cell to this landmark
        # this is just Euclidean distance: sqrt((lx-rx)^2 + (ly-ry)^2)
        exp_range = np.sqrt((landmark_x - rx)**2 + (landmark_y - ry)**2)

        # calculate expected bearing from every cell to this landmark
        # bearing is the angle to the landmark RELATIVE to the robots heading
        angle_to_landmark = np.degrees(np.arctan2(landmark_y - ry, landmark_x - rx))
        exp_bearing = (angle_to_landmark - rth + 180) % 360 - 180

        # sigma values control how forgiving we are about measurement errors
        # larger sigma = more forgiving = wider Gaussian bell curve
        sigma_range = 1.0    # 1 meter tolerance for range errors
        sigma_bearing = 20.0 # 20 degree tolerance for bearing errors

        # Gaussian scoring: how well does each cell match the actual sensor reading?
        # score = 1.0 when expected = measured, drops off as error increases
        range_score = np.exp(-0.5 * ((measured_range - exp_range) / sigma_range) ** 2)
        bearing_score = np.exp(-0.5 * ((measured_bearing - exp_bearing) / sigma_bearing) ** 2)

        # combine range and bearing scores - both must match to get a high score
        likelihood = range_score * bearing_score

        # Bayes update: multiply current belief by the likelihood
        # cells that match the measurement keep their probability
        # cells that dont match get their probability reduced
        self.belief = self.belief * likelihood

        # normalize so everything sums to 1 again
        # add epsilon to avoid dividing by zero if everything collapsed
        total = np.sum(self.belief) + 1e-300
        if total > 1e-10:
            self.belief /= total
        else:
            # if belief collapsed completely reset to uniform - kidnapped robot scenario
            self.belief[:] = 1.0 / self.belief.size

    def odom_callback(self, msg):
        """
        Called every time /odom publishes a new odometry message.
        Does two things:
        1. Updates and publishes the red odometry path for visualization in RViz
        2. Runs the prediction step if the robot moved enough
        We only run predict if the robot moved more than 1mm or rotated more than 0.1 degrees
        to avoid unnecessary computation when the robot is standing still.
        """
        # first message - just store it as reference and wait for the next one
        if self.last_odom_pose is None:
            self.last_odom_pose = msg
            return

        # extract current and previous yaw angles from quaternion
        q = msg.pose.pose.orientation
        curr_yaw_deg = np.degrees(euler_from_quaternion([q.x, q.y, q.z, q.w])[2]) % 360

        q_old = self.last_odom_pose.pose.pose.orientation
        old_yaw_deg = np.degrees(euler_from_quaternion([q_old.x, q_old.y, q_old.z, q_old.w])[2]) % 360

        # calculate how much we moved since last message
        dx = msg.pose.pose.position.x - self.last_odom_pose.pose.pose.position.x
        dy = msg.pose.pose.position.y - self.last_odom_pose.pose.pose.position.y
        dth = (curr_yaw_deg - old_yaw_deg + 180) % 360 - 180

        # Stage odometry starts at (0,0) and gives displacement from start
        # Stage odom.x = world Y displacement, Stage odom.y = world X displacement
        # so we add initial pose to get absolute world coordinates
        self.odom_x = self.initial_pose[0] + msg.pose.pose.position.y
        self.odom_y = self.initial_pose[1] + msg.pose.pose.position.x

        # add this position to the red odom path and publish it
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = msg.header.stamp
        p.pose.position.x = float(self.odom_x)
        p.pose.position.y = float(self.odom_y)
        p.pose.orientation.w = 1.0
        self.odom_path_msg.poses.append(p)
        self.odom_path_pub.publish(self.odom_path_msg)

        # only run the prediction step if the robot actually moved enough
        if np.sqrt(dx**2 + dy**2) > 0.001 or abs(dth) > 0.1:
            self.predict(msg, self.last_odom_pose)
            self.last_odom_pose = msg
            self._publish_costmap()

    def fiducial_callback(self, msg):
        """
        Called every time /fiducials publishes a new landmark detection.
        The robot can see multiple landmarks at once so we loop through all of them.
        For each landmark seen, we run the measurement update to sharpen our belief.
        After processing all landmarks we publish the updated costmap to RViz.
        """
        for marker in msg.markers:
            # get the landmark ID - it comes as an array so we take the first element
            marker_id = int(marker.ids[0])

            # skip if we dont have this landmark in our map database
            if marker_id not in self.landmarks:
                continue

            # get the known world position of this landmark from our map
            lx, ly = self.landmarks[marker_id]

            # get the landmark position relative to the robot from the sensor message
            mx = float(marker.pose.position.x)
            my = float(marker.pose.position.y)

            # calculate range (distance) and bearing (angle) from robot to landmark
            measured_range = np.sqrt(mx**2 + my**2)
            measured_bearing = np.degrees(np.arctan2(my, mx))

            # run the measurement update for this landmark
            self.update_measurement(lx, ly, measured_range, measured_bearing)

        # publish the updated belief as a costmap after processing all landmarks
        self._publish_costmap()


def main():
    rclpy.init()

    # path to the cave world file - we need this to read landmark positions
    world_path = os.path.expanduser("~/ros_ws/src/stage_ros2/world/cave.world")

    # check the file exists before trying to start
    if not os.path.exists(world_path):
        print("ERROR: World file not found at: " + world_path)
        return

    node = BayesFilter3D(world_path)

    try:
        rclpy.spin(node)   # keep the node alive and processing callbacks
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()   # clean up when stopped


if __name__ == "__main__":
    main()
