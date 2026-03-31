import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from example_interfaces.srv import Trigger
import math

class GradingScout(Node):
    def __init__(self):
        super().__init__('grading_scout')
        
        # --- 1. THE TASK SERVICE ---
        self.srv = self.create_service(Trigger, 'get_task', self.get_task_callback)
        
        # --- 2. MONITORING ---
        self.gt_sub = self.create_subscription(Odometry, '/ground_truth', self.gt_cb, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        
        # --- 3. OUTPUT TOPICS ---
        self.status_pub = self.create_publisher(String, '/grading_status', 10)
        self.energy_pub = self.create_publisher(Float32, '/energy_consumed', 10)
        
        # --- 4. STATE ---
        self.total_energy_used = 0.0
        self.current_gt_pos = (0.0, 0.0)
        self.goal_pos = (7.0, 2.5) 
        self.mission_completed = False
        self.last_v = 0.0
        self.last_cmd = Twist()
        
        # --- 5. STATIC ENERGY PARAMETERS ---
        # With no speed limits, base_drain is the penalty for a slow/inefficient path.
        # self.base_drain = 0.05      # Cost per tick (0.1s) just for being active
        # self.linear_coeff = 0.06    # Cost of moving forward
        # self.angular_coeff = 0.15   # Heavier penalty for turning/oscillating
        # self.startup_tax = 0.3     # Penalty for jerky start/stop motion
            
        self.efficiency_dampener = 0.5 

        self.timer = self.create_timer(0.1, self.update_energy)
        self.get_logger().info("Scout Ready: Static Obstacle Energy Model (No Speed Limits).")

        self.debug_start_up_tax_counter = 0

    def get_task_callback(self, request, response):
        self.total_energy_used = 0.0
        self.mission_completed = False
        
        # Define the mission coordinates
        start_x, start_y, goal_x, goal_y = -7.0, -7.0, 7.0, 2.5
        self.goal_pos = (goal_x, goal_y)
        
        response.success = True
        response.message = f"{start_x},{start_y},{goal_x},{goal_y}"
        self.get_logger().warn(f"NEW TASK: Start({start_x},{start_y}) Goal({goal_x},{goal_y})")
        return response

    def gt_cb(self, msg):
        self.current_gt_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # Check if robot reached the goal (within 0.5m)
        if math.dist(self.current_gt_pos, self.goal_pos) < 0.5 and not self.mission_completed:
            self.finalize_mission()
    
    def cmd_cb(self, msg):
        self.last_cmd = msg 
    
    def update_energy(self):
        if self.mission_completed: return

        v = abs(self.last_cmd.linear.x)
        w = abs(self.last_cmd.angular.z)

        # --- MODIFIED PARAMETERS ---
        # Higher time penalty rewards faster (pruned) paths
        time_penalty = 0.05 
        
        # Lower angular penalty so turns don't 'bankrupt' the robot
        motion_cost = (v * 0.06) + (w * 0.15) 

        # Startup tax remains to penalize jerky movement
        if (v > 0.04 and self.last_v <= 0.04):
            self.debug_start_up_tax_counter = self.debug_start_up_tax_counter + 1
            startup_tax = 0.6 
        else:
            startup_tax = 0.0
        
        
        # Calculate tick cost
        tick_cost = time_penalty + motion_cost + startup_tax
        
        self.total_energy_used += (tick_cost * self.efficiency_dampener)
        self.last_v = v 
        self.energy_pub.publish(Float32(data=self.total_energy_used))

    def finalize_mission(self):
        self.mission_completed = True
        summary = f"GOAL REACHED! \nTotal Energy Consumed: {self.total_energy_used:.2f} units"
        self.status_pub.publish(String(data=summary))
        self.get_logger().info(summary)
        self.get_logger().info(f"Startup Tax Counter: {self.debug_start_up_tax_counter}")

def main():
    rclpy.init()
    node = GradingScout()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()