# Puzzlebot Tracker

Autonomous visual tracking system for the differential-drive **Puzzlebot** robot (Jetson Nano + ROS2). The project implements a complete closed-loop perception and control pipeline that allows the robot to detect and follow a target in real time using computer vision techniques.

---

## General Description

The project combines three interconnected subsystems:

- **Perception:** Visual target detection using ArUco markers and/or HSV color segmentation (OpenCV 4.x).
- **Estimation:** 2D Kalman filtering for smoothing detections and predicting target position during temporary vision loss.
- **Control:** Two interchangeable controllers — classical PID and discrete optimal LQR — that convert visual errors into robot motion commands.

---

## Project Objective

Enable the Puzzlebot to autonomously follow a target (green balloon, cone, or ArUco marker) while maintaining a desired distance and keeping the target centered in the camera field of view.

---

## Technologies Used

| Category | Technology |
|---|---|
| Robotics framework | ROS2 (Humble / Iron / Jazzy) |
| Programming language | Python 3.8+ |
| Computer vision | OpenCV 4.x, cv_bridge |
| Linear algebra / Control | NumPy, SciPy (DARE solver) |
| Camera system | CSI Camera (Jetson Nano, GStreamer + NVMM) |
| Build system | Colcon / ament_python |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    PUZZLEBOT TRACKER                     │
│                                                          │
│  ┌─────────────┐    /detection/error (Point)             │
│  │ vision_     │───────────────────────────────┐        │
│  │ tracker     │    /detection/active (Bool)    │        │
│  │             │───────────────────┐            ▼        │
│  │ CSI Camera  │                   │   ┌──────────────┐  │
│  │ GStreamer   │                   │   │ lqr_ /       │  │
│  └─────────────┘                   │   │ pid_         │  │
│                                    │   │ controller   │  │
│  ┌─────────────┐                   │   │              │  │
│  │ kalman_     │ (auxiliary module)│   └──────┬───────┘  │
│  │ tracker     │ optionally used   │          │          │
│  └─────────────┘                   │     /VelocitySetL   │
│                                    │     /VelocitySetR   │
│  ┌─────────────┐                   │     (Float32)       │
│  │ raw_cam     │ (diagnostics)     │          │          │
│  └─────────────┘                   │          ▼          │
│                                    │   ┌──────────────┐  │
│                                    │   │  Puzzlebot   │  │
│                                    │   │  Hardware    │  │
│                                    │   └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Data Flow

```text
CSI Camera
    │  (GStreamer NVMM pipeline)
    ▼
VisionTracker._image_cb()
    │  Detection: ArUco (OpenCV aruco) or HSV blob
    │  EMA Filter → error smoothing
    ▼
/detection/error  (Point)
    │  x = normalized horizontal error [-1, +1]
    │  y = normalized vertical error   [-1, +1]
    │  z = estimated distance [m]  (-1 = lost target)
    ▼
LQRVisualController / VisualServoController
    │  LQR: u = -K·x  where x = [e_x, e_d, ∫e_x, ∫e_d]
    │  PID: u = kp·e + ki·∫e + kd·ė
    ▼
/VelocitySetL + /VelocitySetR  (Float32, rad/s)
    ▼
Puzzlebot Motors
```

---

## Project Structure

```text
puzzlebot_tracker/
├── config/
│   └── params.yaml
├── launch/
│   └── puzzlebot_tracker.launch.py
├── puzzlebot_tracker/
│   ├── __init__.py
│   ├── vision_tracker.py
│   ├── pid_controller.py
│   ├── lqr_controller.py
│   ├── kalman_tracker.py
│   ├── raw_cam.py
│   └── video.sh
├── test/
├── package.xml
├── setup.py
└── setup.cfg
```

---

## Requirements and Dependencies

### Hardware
- Jetson Nano (2GB or 4GB)
- CSI camera compatible with NVMM (Raspberry Pi Camera v2 or IMX219)
- Puzzlebot differential-drive robot with compatible motor drivers

### Software
- Ubuntu 20.04 (Focal) with JetPack 4.6+
- ROS2 Humble (or Iron / Jazzy with minimal API adjustments)
- Python 3.8+

### Python Dependencies

```bash
sudo apt install python3-opencv python3-numpy python3-scipy
```

### ROS2 Dependencies

```xml
rclpy, sensor_msgs, geometry_msgs, std_msgs, cv_bridge
```

Install with:

```bash
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs
```

---

## Installation

### 1. Clone the repository

```bash
cd ~/ros2_ws/src
git clone https://github.com/Arthwwr420/VisualServoingPuzzlebot puzzlebot_tracker
```

### 2. Install dependencies

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select puzzlebot_tracker
source install/setup.bash
```

### 4. Verify installation

```bash
ros2 pkg list | grep puzzlebot
ros2 run puzzlebot_tracker vision_tracker --ros-args -p detection_mode:=hsv
```

---

## Running the Project

### Full system launch (recommended)

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch puzzlebot_tracker puzzlebot_tracker.launch.py
```

Optional launch arguments:

```bash
ros2 launch puzzlebot_tracker puzzlebot_tracker.launch.py \
    mode:=hsv \
    debug:=true \
    aruco_id:=0
```

### Run individual nodes

```bash
# Perception only
ros2 run puzzlebot_tracker vision_tracker

# LQR controller only
ros2 run puzzlebot_tracker lqr_controller

# PID controller only
ros2 run puzzlebot_tracker pid_controller
```

## Main Nodes / Scripts Explanation

### `vision_tracker.py` — Perception Node

**Main class:** `VisionTracker(Node)`

This node is responsible for acquiring images from the CSI camera through GStreamer and processing them in real time to detect the visual target. It publishes the normalized detection error, which is later used by either the PID or LQR controllers.

The system supports multiple detection strategies depending on the environment and the type of target being tracked.

**Detection modes:**

| Mode | Description |
|---|---|
| `hsv` | Color segmentation using the HSV color space |
| `aruco` | ArUco marker detection using OpenCV |
| `hybrid` | ArUco as primary detection with automatic HSV fallback |

The node performs noise filtering, centroid calculation, and approximate distance estimation based on the apparent size of the detected object in the image.

**Published topics:**
- `/detection/error` (`geometry_msgs/Point`): horizontal error, vertical error, and estimated distance
- `/detection/active` (`std_msgs/Bool`): indicates whether a valid target is currently detected

**Auxiliary class:** `EMAFilter`

Exponential moving average filter used to smooth detections and reduce visual jitter caused by illumination noise or rapid camera movement.

---

### `lqr_controller.py` — LQR Controller (Main Controller)

**Main class:** `LQRVisualController(Node)`

Implements a discrete optimal controller based on LQR (Linear Quadratic Regulator) to transform visual tracking errors into motion commands for the differential-drive robot.

The controller uses a linearized system model with augmented integral states to minimize both tracking error and control effort.

The node receives visual error data from `/detection/error`, computes the optimal control action, and converts it into individual wheel velocities for the Puzzlebot.

It also includes target-loss handling logic and saturation limits to prevent excessive control commands.

**Published topics:**
- `/VelocitySetL`, `/VelocitySetR` (`std_msgs/Float32`): left and right wheel angular velocities [rad/s]
- `/controller/state` (`std_msgs/String`): current controller state
- `/controller/gains` (`std_msgs/String`): computed LQR gain matrix

The controller gain matrix is computed by solving the Discrete Algebraic Riccati Equation (DARE) using SciPy.

---

### `pid_controller.py` — PID Controller (Alternative Controller)

**Main class:** `VisualServoController(Node)`

Implements a classical visual servoing strategy using two decoupled PID controllers:

- Angular control for centering the target horizontally
- Linear control for maintaining a desired distance from the target

The node computes linear and angular velocities from perception errors and publishes motion commands compatible with differential-drive navigation.

The controller includes deadzones, output saturation, and adaptive forward-speed reduction while turning in order to improve stability.

**Auxiliary class:** `PIDController`

Discrete PID implementation with:
- Anti-windup
- Derivative filtering
- Output saturation
- Configurable deadzone

**Published topics:**
- `/cmd_vel` (`geometry_msgs/Twist`): robot linear and angular velocity

---

### `kalman_tracker.py` — 2D Kalman Filter

**Class:** `KalmanVisualTracker`

Auxiliary module implementing a two-dimensional Kalman filter to improve the stability and continuity of visual detections.

It does not operate as an independent ROS2 node, but instead acts as a reusable component that can be integrated into `vision_tracker.py`.

The filter is capable of:
- Smoothing noisy measurements
- Estimating target velocity
- Predicting target position during temporary vision loss
- Reducing oscillatory behavior in the control system

**Model used:**

Constant velocity (CWNA) model with state:

```math
[c_x,\; c_y,\; v_x,\; v_y]
```

where:
- `c_x`, `c_y` represent centroid position
- `v_x`, `v_y` represent estimated image-plane velocity

---

### `raw_cam.py` — Standalone Camera Node

Auxiliary node used for CSI camera testing and diagnostics.

It directly publishes the compressed video stream using `sensor_msgs/CompressedImage`, allowing the camera system, latency, and capture quality to be verified independently from the full perception and control pipeline.

Useful for:
- Validating the GStreamer pipeline
- Adjusting exposure or focus
- Verifying real FPS
- Diagnosing hardware or driver issues

---

## Control: Mathematical Explanation

### PID Control (`pid_controller.py`)

Two independent SISO controllers are implemented.

#### Angular loop

```math
\omega(t) = -\left[K_{p\omega} e_x + K_{i\omega}\int e_x dt + K_{d\omega}\dot{e}_x\right]
```

This loop controls robot orientation to keep the target horizontally centered in the image.

#### Linear loop

```math
e_d = d_{measured} - d_{desired}
```

```math
v(t) = K_{pv} e_d + K_{iv}\int e_d dt + K_{dv}\dot{e}_d
```

Used to maintain a desired distance from the target.

#### Gain scheduling

```math
v_{cmd} = v_{raw} \cdot \max(0,\;1 - c_{gain}|e_x|)
```

Automatically reduces forward velocity while turning to improve trajectory stability.

---

### Discrete LQR Control (`lqr_controller.py`)

#### System model

The system is approximated using a first-order linearized model:

```math
\dot{e}_x = -k_{\omega}\omega
```

```math
\dot{e}_d = -k_v v
```

The augmented state vector is:

```math
x = [e_x,\; e_d,\; \int e_x,\; \int e_d]^T
```

and the control vector is:

```math
u = [\omega,\; v]^T
```

#### Discretization

```math
F = I + A \cdot dt
```

```math
G = (I\cdot dt + A\cdot dt^2/2)\cdot B
```

#### LQR gain computation

The optimal gain matrix is obtained by solving the Discrete Algebraic Riccati Equation:

```math
P = DARE(F,G,Q,R)
```

```math
K = (G^TPG + R)^{-1}G^TPF
```

The resulting control law is:

```math
u^* = -Kx
```

Where:
- `Q` penalizes state error
- `R` penalizes control effort

#### Wheel velocity conversion

```math
v_L = \frac{v - \omega b/2}{r}
```

```math
v_R = \frac{v + \omega b/2}{r}
```

where:
- `b` = wheel separation
- `r` = wheel radius

---

## Computer Vision

### HSV Detection

Processing pipeline:

1. Gaussian Blur for noise reduction
2. BGR → HSV conversion
3. Segmentation using `cv2.inRange`
4. Morphological filtering
5. Contour detection
6. Centroid computation
7. Distance estimation using apparent area

Centroid calculation:

```math
c_x = \frac{M_{10}}{M_{00}}
```

```math
c_y = \frac{M_{01}}{M_{00}}
```

Approximate distance estimation:

```math
dist = d_{ref}\sqrt{\frac{A_{ref}}{area}}
```

---

### ArUco Detection

Processing pipeline:

1. CLAHE equalization
2. Detection using `cv2.aruco`
3. Centroid computation
4. Distance estimation using pinhole geometry

```math
dist = \frac{marker\_size \cdot f_x}{width_{px}}
```

---

### EMA Filter

Exponential smoothing filter used to stabilize measurements:

```math
s_k = \alpha z_k + (1-\alpha)s_{k-1}
```

---

## Configurable Parameters (`params.yaml`)

### `vision_tracker`

| Parameter | Default | Description |
|---|---|---|
| `detection_mode` | `"hsv"` | Detection mode |
| `hsv_lower` | `[35,60,60]` | Lower HSV threshold |
| `hsv_upper` | `[85,255,255]` | Upper HSV threshold |
| `hsv_min_area` | `800` | Minimum contour area |
| `aruco_target_id` | `0` | ArUco target ID |
| `ema_alpha` | `0.35` | EMA smoothing factor |

### `lqr_visual_controller`

| Parameter | Default | Description |
|---|---|---|
| `desired_distance` | `0.50` | Desired distance |
| `v_max` | `0.12` | Maximum linear velocity |
| `omega_max` | `1.20` | Maximum angular velocity |
| `q_ex` | `12.0` | Horizontal error weight |
| `r_omega` | `0.15` | Angular effort penalty |

---

## Usage Examples

### Follow a green object

```bash
ros2 launch puzzlebot_tracker puzzlebot_tracker.launch.py mode:=hsv
```

### Follow an ArUco marker with ID=3

```bash
ros2 param set /vision_cam_tracker detection_mode aruco
ros2 param set /vision_cam_tracker aruco_target_id 3
```

### Real-time PID tuning

```bash
ros2 param set /visual_servo_controller ang_kp 0.80
ros2 param set /visual_servo_controller v_max 0.20
```

### Monitor controller state

```bash
ros2 topic echo /controller/state
ros2 topic echo /detection/error
```

### Visualize with rqt

```bash
rqt_graph
rqt_plot /detection/error/x /detection/error/z
```

---

## Troubleshooting

### Camera does not open / GStreamer error

```bash
ls /dev/video*
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink
```

Verify:
- CSI connection
- JetPack installation
- Camera permissions

### Robot does not move

```bash
ros2 topic echo /VelocitySetL
```

Verify:
- active topics
- motor driver connection
- valid detection

### Unstable HSV detection

- recalibrate HSV thresholds
- increase `hsv_min_area`
- reduce `ema_alpha`

### Excessive oscillations

- reduce `ang_kp`
- increase `r_omega`
- reduce `q_ex`

---

## Future Improvements

- Full Kalman filter integration into `vision_tracker`
- Multi-target tracking
- Automatic camera calibration
- Dynamic ROS2 reconfiguration
- Logging with `rosbag2`
- Adaptive control
- Multi-color detection support

---

## Credits

**Oscar y los de la Rosa** — Autonomous control academic project.

Built using:
- ROS2
- OpenCV
- Jetson Nano CSI Camera

License: **MIT**