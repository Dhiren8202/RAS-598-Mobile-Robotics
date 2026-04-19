# Assignment 2: Motion Planning

## Overview
This package implements a complete navigation stack for a robot to navigate through a cave environment using the Stage simulator. The robot navigates from start (-7.0, -7.0) to goal (7.0, 2.5) while minimizing total energy consumption.

## Results
- **Energy Consumed:** 43.19 units
- **Startup Taxes:** 10 (theoretical minimum — one per waypoint)
- **Raw A* Waypoints:** 120
- **Pruned Waypoints:** 11 (after Line-of-Sight pruning)
- **No collisions**

## How to Run

### Terminal 1 — Launch Everything
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros_ws/install/setup.bash
ros2 launch ras598_assignment_2 planner_launch.py
```

### Terminal 2 — Monitor Energy in Real Time
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros_ws/install/setup.bash
ros2 daemon stop && ros2 daemon start
ros2 topic echo /energy_consumed
```

### Terminal 3 — Monitor Goal Status
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros_ws/install/setup.bash
ros2 daemon stop && ros2 daemon start
ros2 topic echo /grading_status
```

> **Note:** Start Terminal 2 and Terminal 3 before Terminal 1. Wait until you see "Controller started!" in Terminal 1 before checking other terminals.

## Implementation Details

### 1. Occupancy Grid (80x80 at 0.2m/cell)
- Loads `cave_filled.png` directly using PIL library
- Flips image vertically to match ROS world coordinates — image origin is top-left, world origin is bottom-left
- Downsamples from 500x500 pixels to 80x80 grid where each cell = 0.2m x 0.2m
- Inflates obstacles by 3 cells = 0.6m safety margin using binary dilation
- Note: 4 cell inflation (0.8m) was tested but completely blocks all corridors in this specific cave map. 3 cells is the maximum viable inflation as verified through testing.

### 2. Coordinate Mapping
- **World to Grid:** `col = (world_x + 8.0) / 0.2`, `row = (world_y + 8.0) / 0.2`
- **Grid to World:** `world_x = -8.0 + col * 0.2 + 0.1`, `world_y = -8.0 + row * 0.2 + 0.1`
- The +0.1 offset returns the center of each cell not the corner
- Example: Start(-7.0, -7.0) → Grid(5, 5) | Goal(7.0, 2.5) → Grid(52, 75)

### 3. A* Global Planner
- Implemented completely from scratch using a priority queue (heapq)
- 8-directional movement: straight moves cost 1.0, diagonal moves cost 1.414 (Pythagoras)
- Heuristic: straight line distance to goal guides the search toward the goal efficiently
- Breadcrumb dictionary (came_from) traces the path back from goal to start then reverses it

### 4. Line-of-Sight Path Pruning
- Reduces raw A* path from 120 waypoints down to 11 waypoints
- Uses Bresenham line algorithm to check if a straight line between two waypoints is obstacle-free
- Adaptive jump distance: 7m max in tight corridors, 50m in open areas
- Fewer waypoints = fewer turns = fewer stops = less energy consumed

### 5. Turn-Go-Turn Controller with Lateral Drift Detection
- **State 1 (Rotate):** Stop completely and rotate in place until heading error < 0.03 radians (~2 degrees)
- **State 2 (Drive):** Drive straight forward with `angular.z = exactly 0.0` as per spec
- **Lateral Drift Detection:** While driving, continuously monitor perpendicular distance from the intended straight line using cross product formula. If drift exceeds 0.4m, stop and re-align in one clean correction
- Speed proportional to distance: 0.3 m/s far away, 0.2 m/s getting close, 0.1 m/s almost there
- 10 startup taxes = one per waypoint = theoretical minimum for 11 waypoints

## Why 10 Startup Taxes is the Minimum
The grading scout charges a startup tax every time the robot goes from stopped to moving. With 11 waypoints and a pure Turn-Go-Turn controller, the robot must stop and rotate at each waypoint before driving. That gives exactly 10 stops (no rotation needed after the final waypoint = goal). This is the absolute minimum possible for this controller design.

## ROS 2 Topics and Services

| Type | Topic | Message Type |
|------|-------|-------------|
| Subscriber | /ground_truth | nav_msgs/Odometry |
| Subscriber | /energy_consumed | std_msgs/Float32 |
| Publisher | /cmd_vel | geometry_msgs/Twist |
| Publisher | /planner_markers | visualization_msgs/MarkerArray |
| Service Client | /get_task | example_interfaces/Trigger |

## Visualization in RViz
- **Green LINE_STRIP** — raw A* path (all 120 original waypoints)
- **Blue LINE_STRIP** — pruned path (11 simplified waypoints the robot follows)
- **Red SPHERE** — current target waypoint the robot is heading towards

## Package Structure
```
ras598_assignment_2/
├── ras598_assignment_2/
│   └── planner.py           # main planner node
├── launch/
│   └── planner_launch.py    # launches all 5 components
├── cave_filled.png           # cave map bitmap
├── map.yaml                  # map configuration
├── planning.rviz             # RViz configuration
└── grading_scout.py          # provided grading node (do not modify)
```
