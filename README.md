# Development of a 21+-DoF Adult-Sized Humanoid Robot

XR-based upper-body teleoperation system using **Meta Quest 3S** and **OpenTeleVision**.  
The operator's arm, wrist, neck, and finger movements are tracked in real time and mapped to the robot via inverse kinematics.

**Robotics Innovatory Lab, Sungkyunkwan University** — Sungjun Lee

---

## Repository Structure

```
Development-of-the-Adult-Sized-Humanoid-Robot/
├── 21-DoF/
│   └── Wholebody_21_DoF_URDF/
├── 39-DoF/
│   └── Wholebody_39_DoF_URDF/
└── Teleoperation/
    ├── TeleVision/
    │   ├── teleop/TeleVision.py     # Quest streaming server (vuer-based)
    │   ├── cert.pem                 # SSL certificate for Quest HTTPS
    │   └── key.pem                  # SSL private key
    └── control/
        ├── HR_teleop.py             # Main — TeleopController state machine
        ├── config.py                # All parameters and paths
        ├── robot_model.py           # Pinocchio model, IK, coordinate transform
        ├── motion_utils.py          # EMA filter, beep, smooth pose transitions
        ├── ros_interface.py         # ROS publishers/subscribers + camera overlay
        ├── finger_mapping.py        # Hand landmark → finger joint angles
        ├── quest_video.py           # Intro video playback to Quest
        └── calib.json               # Arm calibration data (auto-generated)
```

---

## System Overview

```
[Meta Quest 3S]
  Hand position (25-joint WebXR) ──► OpenTeleVision ──► HR_teleop.py
  Head matrix (4×4)                                          │
                                              ┌──────────────┼──────────────┐
                                              ▼              ▼              ▼
                                        robot_model.py  finger_mapping.py  config.py
                                        (Pinocchio IK)  (FE + AA retarget)
                                              │
                                        ros_interface.py
                                     ┌───────┼──────────────┐
                                     ▼       ▼              ▼
                             /arm_controller  /finger_controller  /neck_controller
                             (12 joints)      (16 joints)         (2 joints)
                                     │
                              [Jetson Orin Nano]
                              RealSense D435i ──► camera stream ──► Quest overlay
```

### Controlled Joints

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | L_shoulder_pitch | 6 | R_shoulder_roll |
| 1 | L_shoulder_roll | 7 | R_shoulder_yaw |
| 2 | L_shoulder_yaw | 8 | R_elbow_pitch |
| 3 | L_elbow_pitch | 9 | R_wrist_yaw |
| 4 | L_wrist_yaw | 10 | Neck_Yaw |
| 5 | R_shoulder_pitch | 11 | Neck_Pitch |

Fingers: 16 channels via `/finger_controller/command` (4 fingers × [AA, FE] × 2 hands)

---

## Installation

### 1. ROS Noetic

```bash
sudo apt install ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Pinocchio

```bash
# Via conda (recommended)
conda install pinocchio -c conda-forge

# Or via apt
sudo apt install -qqy lsb-release curl
sudo mkdir -p /etc/apt/keyrings
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
    | sudo tee /etc/apt/keyrings/robotpkg.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] \
    http://robotpkg.openrobots.org/packages/debian/pub \
    $(lsb_release -cs) robotpkg" \
    | sudo tee /etc/apt/sources.list.d/robotpkg.list
sudo apt update && sudo apt install robotpkg-py38-pinocchio
```

### 3. Python dependencies

```bash
pip install numpy scipy opencv-python vuer==0.0.32rc7
```

> ⚠️ `vuer` must be pinned to `0.0.32rc7`. Newer versions have breaking API changes.

### 4. System audio

```bash
sudo apt install pulseaudio-utils   # provides paplay
```

### 5. SSL certificate (for Quest HTTPS)

```bash
cd Teleoperation/TeleVision
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
    -subj "/CN=localhost"
```

---

## Quick Start

### Network mode (set before anything else)

```bash
rme    # Experiment Mode — Station PC as ROS Master (Gazebo simulation)
rmr    # Real Robot Mode — Gene PC as ROS Master (physical robot)
rms    # Check current mode
```

### Experiment Mode (recommended order)

```bash
xrgazebo        # Terminal 1 — Gazebo simulation
xrcamerastart   # Terminal 2 — RealSense D435i stream via Jetson
xrhand          # Terminal 3 — MuJoCo bridge on Jetson (AmazingHand)
xrhandarduino   # Terminal 4 — Arduino hand control on Jetson
xrteleop        # Terminal 5 — Start teleoperation
```

### Real Robot Mode

```bash
xrcamerastart   # Terminal 1 — Camera stream
xrrviz          # Terminal 2 — RViz visualization
xrteleop        # Terminal 3 — Start teleoperation
```

### Connect from Quest browser

| Service | URL |
|---------|-----|
| VR stream | `https://localhost:8012?ws=wss://localhost:8012` |
| Camera feed | `http://localhost:8080/stream?topic=/camera/color/image_raw&type=mjpeg` |

After connecting, a countdown of `TELEOP_START_DELAY` seconds (default 5s) plays before the robot starts moving.

---

## State Machine

```
WAITING_QUEST → CALIBRATING → SYNCING → TELEOP
                                  ↑         ↓
                               FREEZE ←─────┘
```

| State | Description |
|-------|-------------|
| **WAITING_QUEST** | Waiting for Quest browser connection. Plays intro video (if configured), then counts down `TELEOP_START_DELAY` seconds. |
| **CALIBRATING** | Skipped if `calib.json` exists. Robot moves to arms-forward pose → waits 3s → collects 50 frames → saves `calib.json`. |
| **SYNCING** | Moves robot from its current pose to match the operator's current hand position using smooth-step interpolation. |
| **TELEOP** | Full teleoperation at `CONTROL_HZ`. IK solved every frame. |
| **FREEZE** | Triggered by tracking loss, position jump (`JUMP_THRESHOLD`), or IK divergence. Holds current pose for `FREEZE_DURATION` seconds, then re-syncs. |

---

## Arm Calibration

Runs automatically when `calib.json` is missing.

1. Stand in front of the robot.
2. Extend both arms straight forward, parallel to the ground.
3. Hold for ~1 second — 50 frames are collected automatically.

**What gets saved to `calib.json`:**

| Field | Description |
|-------|-------------|
| `quest_L/R_init` | Quest-space hand position at calibration time |
| `quest_neck_rot_init` | Head rotation matrix at calibration time |
| `quest_L/R_wrist_rot_init` | Wrist rotation matrix at calibration time |

**Reset calibration:**

```bash
xrcalib    # Delete calib.json (triggers re-calibration on next run)
```

---

## Coordinate Transform

Quest and the robot use different coordinate systems. The mapping applied in `robot_model.py`:

| Quest axis | Robot axis | Meaning |
|------------|------------|---------|
| −Z | +X | Forward / backward |
| −X | +Y | Left / right |
| +Y | +Z | Up / down |

Delta motion is computed relative to the calibration pose, so the robot mirrors the operator's relative hand movement regardless of physical position.

---

## Inverse Kinematics

Position-only IK using **Damped Least Squares (DLS) + null-space control** (`robot_model.py`):

```
dq = J†·e + (I − J†J)·(−w·(q − q_ref))
```

- **DLS pseudo-inverse** `J† = Jᵀ(JJᵀ + λI)⁻¹` prevents divergence at singularities (e.g. fully extended arm).
- **λ scales adaptively** with position error — conservative far from target, precise when close.
- **Null-space term** guides the elbow toward a natural reference pose without affecting hand position.
- Left and right arms solved independently using `L_joint_mask` / `R_joint_mask`.

IK target is the palm center, located `PALM_Z_OFFSET = −0.11815 m` from the `wrist_yaw` frame along Z.

**Built-in safety checks (4 levels):**
1. Adaptive λ — larger error → larger λ → prevents divergence
2. Step size clamping — max 0.3 rad per iteration
3. Early exit on divergence — stops if error grows > 1% from previous iteration
4. Joint limit clamping count — exits if joint limits are hit 5 times consecutively

---

## Finger Retargeting

`finger_mapping.py` maps Quest hand landmarks to **16 robot finger joint commands**. No calibration required.

- **Landmark source:** WebXR 25-joint skeleton (Meta Quest 3S + vuer). **Not** MediaPipe 21-point.
- **Robot fingers:** 4 per hand (Thumb, Index, Middle, Ring). Ring finger uses Ring landmark (not Pinky).

**FE (flex/extend) — 3-point angle method:**

For each finger, three landmarks (Wrist → MCP → Tip) define a flex angle at the MCP vertex:
```
flex_angle = 180° − angle_at_vertex(Wrist, MCP, Tip)
```

Linearly mapped to FE joint angle:

| Flex angle | FE command |
|------------|------------|
| ≤ ANGLE_OPEN[f] | 0 rad (fully extended) |
| ≥ ANGLE_CLOSE[f] | −1.501 rad (fully closed) |

Default tuning values:
```python
ANGLE_OPEN  = {1: 10.0, 2: 6.0, 3: 4.0, 4: 10.0}     # [deg] → FE = 0
ANGLE_CLOSE = {1: 50.0, 2: 135.0, 3: 149.0, 4: 150.0} # [deg] → FE_MIN
```

**AA (abduction/adduction):** Computed from MCP→PIP vector projected onto the palm local frame. PIP is used instead of Tip to avoid FE contamination when fingers curl.

**AA offset calibration:** At TELEOP entry, the current hand pose is measured and stored as `aa_offset`. All subsequent AA values have this offset subtracted, so the natural resting hand pose maps to 0°.

**Output array (16 values):**
```
[L_AA_1, L_FE_1, L_AA_2, L_FE_2, L_AA_3, L_FE_3, L_AA_4, L_FE_4,
 R_AA_1, R_FE_1, R_AA_2, R_FE_2, R_AA_3, R_FE_3, R_AA_4, R_FE_4]
```

Even indices (0,2,4,...) = AA, Odd indices (1,3,5,...) = FE.  
Finger numbering: 1=Thumb, 2=Index, 3=Middle, 4=Ring.

**Tuning:**
- Enable `FINGER_DEBUG = True` in `config.py` to print live flex angles to the terminal.
- Robot hand doesn't fully open → decrease `ANGLE_OPEN`
- Robot hand doesn't fully close → decrease `ANGLE_CLOSE`

---

## Neck Control

Three modes, selected via `config.py`:

| Mode | Config | Behavior |
|------|--------|----------|
| Head tracking | `USE_NECK=True` | Neck follows operator's head rotation directly |
| Hand midpoint tracking | `USE_NECK=False`, `USE_NECK_TRACK=True` | Neck points toward midpoint of both hands |
| Fixed | `USE_NECK=False`, `USE_NECK_TRACK=False` | Neck stays at 0° |

Neck commands are published to both `/arm_controller/command` (indices 10, 11) and `/neck_controller/command` (dedicated 2-value topic for Jetson Dynamixel node).

---

## EMA Smoothing

All outputs are smoothed with an Exponential Moving Average filter before publishing:
```
output = α × new + (1 − α) × prev
```

| Filter | α | Applied to |
|--------|---|------------|
| `arm_filter` | 0.6 | IK joint angles (shoulder, elbow) |
| `wrist_filter_l/r` | 0.6 | Wrist yaw |
| `quest_pos_filter_l/r` | 0.7 | Quest hand position input (pre-IK) |
| `neck_filter` | 0.3 | Neck yaw and pitch |
| `finger_filter` | 0.4 | All 16 finger joint commands |

All α values are adjustable in `config.py`. Filters reset to actual robot pose at the start of each SYNCING phase.

---

## ROS Topics

| Topic | Direction | Type | Description |
|-------|-----------|------|-------------|
| `/arm_controller/command` | Publish | Float64MultiArray | 12 arm+neck joint angles [rad] |
| `/finger_controller/command` | Publish | Float64MultiArray | 16 finger joint angles [rad] |
| `/amazing_hand/finger_angles` | Publish | Float32MultiArray | Finger angles (AmazingHand hardware copy) |
| `/neck_controller/command` | Publish | Float64MultiArray | [yaw, pitch] [rad] for Jetson Dynamixel node |
| `/camera/color/image_raw` | Subscribe | sensor_msgs/Image | D435i camera for Quest streaming |
| `/joint_states` | Subscribe | sensor_msgs/JointState | Current robot joint positions |

---

## Configuration (`control/config.py`)

```python
# ── Enable / disable subsystems ─────────────────────────────
USE_ARM        = True    # Arm + neck IK teleoperation
USE_FINGER_FE  = True    # Finger flex/extend tracking
USE_FINGER_AA  = True    # Finger abduction/adduction tracking
USE_NECK       = False   # Direct head rotation tracking
USE_NECK_TRACK = True    # Hand midpoint tracking (active when USE_NECK=False)
USE_SPHERE     = True    # Show hand guide spheres in Quest
FINGER_DEBUG   = False   # Print live flex angles to terminal
USE_INTRO_VIDEO= True    # Play intro video on Quest connection

# ── Timing ──────────────────────────────────────────────────
CONTROL_HZ         = 50     # Main loop frequency [Hz]
TELEOP_START_DELAY = 5.0    # Countdown after Quest connects [s]

# ── Safety ──────────────────────────────────────────────────
JUMP_THRESHOLD       = 0.15   # Hand position jump limit [m]
FREEZE_DURATION      = 2.0    # Hold time after tracking loss [s]
SYNC_DURATION        = 3.0    # Sync interpolation time [s]
SYNC_TIMEOUT         = 10.0   # Max sync wait before forcing teleop [s]
SYNC_JOINT_THRESH    = 0.1    # Sync done: joint error [rad]
SYNC_POSITION_THRESH = 0.05   # Sync done: palm position error [m]

# ── EMA filter alpha values ──────────────────────────────────
EMA_ARM       = 0.6
EMA_WRIST     = 0.6
EMA_QUEST_POS = 0.7
EMA_NECK      = 0.3
EMA_FINGER    = 0.4
```

**Use case presets:**

| Use case | USE_ARM | USE_FINGER_FE | USE_FINGER_AA |
|----------|---------|---------------|---------------|
| Full teleoperation | True | True | True |
| Arm + neck only | True | False | False |
| Fingers only | False | True | True |

---

## Command Reference

### ROS Network Mode

| Command | Description |
|---------|-------------|
| `rme` | Experiment Mode — Station PC as ROS Master |
| `rmr` | Real Robot Mode — Gene PC as ROS Master |
| `rms` | Show current ROS network status |

### Universal Commands

| Command | Description |
|---------|-------------|
| `xrteleop` | Start teleoperation (`HR_teleop.py`) |
| `xrcamerastart` | Start Jetson RealSense D435i stream |
| `xrcamerastop` | Stop camera stream |
| `xrcalib` | Delete `calib.json` (triggers re-calibration) |
| `xrhelp` | Show full command reference |

### Experiment Mode Only

| Command | Description |
|---------|-------------|
| `xrgazebo` | Launch Gazebo simulation |
| `xrhand` | Run MuJoCo bridge on Jetson (AmazingHand) |
| `xrhandarduino` | Run Arduino hand control on Jetson |

### Real Robot Mode Only

| Command | Description |
|---------|-------------|
| `xrrviz` | Launch RViz real-time visualization |

---

## Useful ROS Commands

```bash
# Check joint state topic order
rostopic echo /dual_arm/joint_states/name

# Monitor topic frequency
rostopic hz /arm_controller/command
rostopic hz /camera/color/image_raw

# Check camera bandwidth
rostopic bw /camera/color/image_raw

# Convert xacro to URDF
rosrun xacro xacro dual_arm.xacro --inorder > dual_arm.urdf
```