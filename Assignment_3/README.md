# Assignment 3: Robot Localization using Bayes Filter (Histogram Filter)

## Course: RAS 598 - Mobile Robotics | Arizona State University

---

## What This Assignment Does

This package implements a **3D Discrete Bayes Filter** (also called a Histogram Filter) to localize a robot in a known cave environment.

Instead of knowing exactly where the robot is, we maintain a **3D probability grid** where every cell holds a number representing how likely the robot is at that position facing that direction.

The filter updates in two ways:
1. **Predict Step**: When the robot moves, we shift the probability cloud to match
2. **Update Step**: When the robot sees a landmark, we sharpen the cloud using sensor data

Over time the cloud converges to the robots true location.

---

## How to Run

### Launch Everything at Once
```bash
ros2 launch ras598_assignment_3 bayes_launch.py
```

### Or Step by Step

**Terminal 1 - Stage Simulator:**
```bash
QT_QPA_PLATFORM=wayland ros2 launch stage_ros2 demo.launch.py world:=cave use_stamped_velocity:=false
```

**Terminal 2 - Bayes Filter Node:**
```bash
ros2 run ras598_assignment_3 bayes_filter
```

**Terminal 3 - RViz:**
```bash
rviz2 -d ~/ros_ws/src/ras598_assignment_3/bayes.rviz
```

---

## Package Structure

```
ras598_assignment_3/
├── ras598_assignment_3/
│   └── bayes_filter.py        # Main Bayes Filter node - all filter logic is here
├── launch/
│   └── bayes_launch.py        # Launch file - starts Stage, RViz and filter together
├── bayes.rviz                 # RViz configuration file
├── bayes_boilerplate.py       # Original boilerplate provided by professor
├── package.xml                # ROS 2 package description
├── setup.py                   # Python package setup
└── setup.cfg                  # Script installation configuration
```

---

## Grid Configuration

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| World Size | 16m x 16m | Cave goes from -8m to +8m in x and y |
| Spatial Resolution | 0.2m per cell | Each cell = 20cm x 20cm of real space |
| Grid Dimensions | 80 x 80 cells | 16m / 0.2m = 80 cells per side |
| Angular Resolution | 10 degrees per bin | 360 / 10 = 36 orientation bins |
| Belief Array Shape | (80, 80, 36) | Total of 230,400 cells |

---

## Landmark Map

| ID | X (meters) | Y (meters) | Description |
|----|-----------|-----------|-------------|
| 10 | -5.0 | -5.0 | Near the starting area |
| 20 | -5.0 | 1.0 | Center-left corridor |
| 30 | -2.0 | 7.2 | Top-left alcove |
| 40 | -1.0 | 3.1 | Central passage |
| 50 | -1.0 | -1.0 | Near the central junction |

---

## ROS Topics

| Type | Topic | Message Type | Description |
|------|-------|-------------|-------------|
| Publisher | viz/belief_costmap | OccupancyGrid | 2D probability heatmap |
| Publisher | viz/landmarks | MarkerArray | Red cylinders showing landmark positions |
| Publisher | viz/gt_path | Path | Green line - ground truth trajectory |
| Publisher | viz/odom_path | Path | Red line - raw odometry trajectory |
| Subscriber | /odom | Odometry | Noisy odometry - used for prediction step |
| Subscriber | /ground_truth | Odometry | True position - used ONLY for visualization |
| Subscriber | /fiducials | MarkerDetection | Landmark detections - used for update step |

---

## How the Filter Works

### Initialization
- Robot starts at (-7.0, -7.0) facing 90 degrees (north)
- All probability is placed in that single cell (certain start)
- If starting position is unknown, probability spreads uniformly (kidnapped robot scenario)

### Predict Step (Motion Model)
Uses **Turn-Go-Turn** decomposition:
1. First rotation (d_rot1 = half of total rotation)
2. Translation (shift x,y based on distance moved and heading)
3. Second rotation (d_rot2 = other half of total rotation)
4. Gaussian blur to model odometry noise
5. Normalize so belief sums to 1.0

### Update Step (Measurement Model)
For every landmark detected:
1. Calculate expected range and bearing from every grid cell to the landmark
2. Score each cell using Gaussian PDF
3. Multiply belief by likelihood scores (Bayes Rule)
4. Normalize so belief sums to 1.0

---

## Why is the Probability Distribution Ring Shaped?

When the robot sees a single landmark and only uses **range** (distance), every point that is exactly that distance away from the landmark is equally likely. This forms a **ring (circle)** of high probability centered on the landmark.

When **bearing** (angle) is added, the ring collapses because only cells facing the right direction match both range AND bearing. With 2-3 landmarks the ring collapses to a single sharp peak near the robots true location.

---

## Dependencies

- ROS 2 Jazzy
- stage_ros2 (bayes branch)
- numpy
- scipy
- tf_transformations

---

## Author
**Dhiren Makwana**
RAS 598 Mobile Robotics - Arizona State University
