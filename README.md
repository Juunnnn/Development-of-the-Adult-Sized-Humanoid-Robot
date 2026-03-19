# Development of a 21(+)-DoF Adult-Sized Humanoid Robot

Development of a 21+ Degrees of Freedom Adult Sized Humanoid Robot by Sungjun Lee @Robotics Innovatory Lab, Sungkyunkwan University

![39DOF](https://github.com/user-attachments/assets/fbc37dc6-26eb-4b8e-a24a-afc2e2e71929)

---

## Teleoperation

XR-based upper-body teleoperation system using Meta Quest 3S and [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision).  
The operator's arm, wrist, neck, and finger movements are tracked in real time and mapped to the robot via inverse kinematics (IK).

### Required Files

The following are the **only files needed** to run `HR_teleop.py`.

```
Teleoperation/
├── TeleVision/
│   ├── teleop/
│   │   └── TeleVision.py        # Quest streaming server (vuer-based)
│   ├── cert.pem                 # SSL certificate for Quest HTTPS
│   └── key.pem                  # SSL private key
└── control/
    ├── HR_teleop.py             # Main — state machine entry point
    ├── config.py                # All parameters and paths
    ├── robot_model.py           # Pinocchio model, IK, coordinate transform
    ├── motion_utils.py          # EMA filter, beep, smooth pose transitions
    ├── ros_interface.py         # ROS publishers/subscribers + camera overlay
    ├── finger_mapping.py        # Hand landmark → finger joint angles
    ├── quest_video.py           # Quest video feed handler
    └── calib.json               # Arm calibration data (auto-generated on first run)
```

---

### Installation

#### 1. ROS Noetic

Follow the official guide: http://wiki.ros.org/noetic/Installation/Ubuntu

```bash
sudo apt install ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### 2. Pinocchio (robot kinematics)

```bash
sudo apt install -qqy lsb-release curl
sudo mkdir -p /etc/apt/keyrings
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
    | sudo tee /etc/apt/keyrings/robotpkg.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] \
    http://robotpkg.openrobots.org/packages/debian/pub \
    $(lsb_release -cs) robotpkg" \
    | sudo tee /etc/apt/sources.list.d/robotpkg.list
sudo apt update
sudo apt install robotpkg-py38-pinocchio
```

Or via conda (recommended if using a conda environment):

```bash
conda install pinocchio -c conda-forge
```

#### 3. Python dependencies

All Python packages required across `HR_teleop.py`, `robot_model.py`, `motion_utils.py`, `ros_interface.py`, and `finger_mapping.py`:

```bash
pip install numpy scipy opencv-python vuer==0.0.32rc7
```

| Package | Used in | Purpose |
|---|---|---|
| `numpy` | all files | Array math |
| `scipy` | `robot_model.py`, `HR_teleop.py` | `Rotation` (wrist/neck quaternion) |
| `opencv-python` | `ros_interface.py` | Camera frame resize + overlay drawing |
| `vuer==0.0.32rc7` | `TeleVision.py` | Quest browser streaming server |

> **Note:** `vuer` must be pinned to `0.0.32rc7`. Newer versions may have breaking API changes.

#### 4. System audio (for state-change beeps)

```bash
sudo apt install pulseaudio-utils   # provides paplay
```

#### 5. SSL certificate (for Quest HTTPS connection)

```bash
cd Teleoperation/TeleVision
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
    -subj "/CN=localhost"
```

---

### Quick Start

Run each command in a separate terminal.

**Terminal 1 — Gazebo simulation**
```bash
xrgazebo
```

**Terminal 2 — Camera stream** (For video feed in Quest: https://localhost:8012?ws=wss://localhost:8012)
```bash
xrcamerastart
```

**Terminal 3 — Teleoperation**
```bash
xrteleop
```

Then connect from the Quest browser:
```
http://localhost:8080/stream?topic=/camera/color/image_raw&type=mjpeg
```

After connecting, a countdown of `TELEOP_START_DELAY` seconds (default 5s) is displayed in the Quest headset before the robot begins moving.

---

### State Machine

```
WAITING_QUEST → CALIBRATING → SYNCING → TELEOP
                                  ↑         ↓
                               FREEZE ←────┘
```

| State | Description |
|---|---|
| `WAITING_QUEST` | Waiting for Quest browser connection. Counts down `TELEOP_START_DELAY` s. |
| `CALIBRATING` | Arms-extended pose → collects 50 frames → saves `calib.json`. Skipped if `calib.json` already exists. |
| `SYNCING` | Moves robot to match operator's current hand position using ease-in-out interpolation. |
| `TELEOP` | Full teleoperation — IK solved every frame at `CONTROL_HZ`. |
| `FREEZE` | Tracking lost or jump detected. Holds current pose for `FREEZE_DURATION` s, then re-syncs. |

Calibration data is reloaded automatically on subsequent runs, so **arm calibration only needs to be done once** unless the physical setup changes.

---

### Arm Calibration

Runs automatically if `calib.json` is missing.

1. Stand in front of the robot.
2. Extend both arms straight forward, parallel to the ground.
3. Hold the pose — 50 frames are collected automatically (~1 second).

The following data is saved to `calib.json`:

| Field | Description |
|---|---|
| `quest_L/R_init` | Quest-space hand position at calibration time (translation origin) |
| `quest_neck_rot_init` | Head rotation matrix at calibration time (neck yaw/pitch baseline) |
| `quest_L/R_wrist_rot_init` | Wrist rotation matrix at calibration time (wrist yaw baseline) |

#### Reset calibration

```bash
xrcalib         # Delete calib.json (triggers re-calibration on next run)
```

---

### Coordinate Transform

Quest and the robot use different coordinate systems. The mapping applied in `robot_model.py`:

| Quest axis | Robot axis | Meaning |
|---|---|---|
| −Z | +X | Forward / backward |
| −X | +Y | Left / right |
| +Y | +Z | Up / down |

Delta motion is computed relative to the calibration pose, so the robot mirrors the operator's relative hand movement regardless of physical position.

---

### Inverse Kinematics

Position-only IK using **Damped Least Squares (DLS) + null-space control** (`robot_model.py`):

```
dq = J†·e + (I − J†J)·(−w·(q − q_ref))
```

- **DLS pseudo-inverse** `J† = Jᵀ(JJᵀ + λI)⁻¹` prevents divergence near singularities (e.g. fully extended arm).
- **Null-space term** guides the elbow toward a natural reference pose without affecting hand position.
- `λ` scales adaptively with position error — conservative far from target, precise when close.
- Left and right arms solved independently using `L_joint_mask` / `R_joint_mask`.

IK target is the **palm center**, located `PALM_Z_OFFSET = −0.11815 m` from the wrist_yaw frame along Z.

---

### Finger Retargeting

`finger_mapping.py` maps Quest hand landmarks to 16 robot finger joint commands. **No calibration required.**

**Landmark source:** WebXR 25-joint skeleton (Meta Quest 3S + vuer).  
This is **not** MediaPipe 21-point — the joint indices are different.

For each finger, three landmarks (Wrist → MCP → Tip) define a flex angle at the MCP vertex:

```
flex_angle = 180° − angle_at_vertex(Wrist, MCP, Tip)
```

Linearly mapped to FE joint angle:

| Flex angle | FE command |
|---|---|
| ≤ `ANGLE_OPEN[f]` | 0 rad (fully extended) |
| ≥ `ANGLE_CLOSE[f]` | −1.501 rad (fully closed) |

**Output (16 values):**
```
[L_AA_1, L_FE_1, L_AA_2, L_FE_2, L_AA_3, L_FE_3, L_AA_4, L_FE_4,
 R_AA_1, R_FE_1, R_AA_2, R_FE_2, R_AA_3, R_FE_3, R_AA_4, R_FE_4]
```
Finger numbering: 1=Thumb, 2=Index, 3=Middle, 4=Ring.

**Tuning** (in `finger_mapping.py`):

```python
ANGLE_OPEN  = {1: 14.0, 2: 7.0, 3: 7.0, 4: 9.0}      # [deg] → FE = 0
ANGLE_CLOSE = {1: 53.0, 2: 130.0, 3: 139.0, 4: 147.0}  # [deg] → FE_MIN
```

- Robot hand doesn't fully open → decrease `ANGLE_OPEN`
- Robot hand doesn't fully close → decrease `ANGLE_CLOSE`
- Enable `FINGER_DEBUG = True` in `config.py` to print live flex angles to the terminal.

---

### EMA Smoothing

All outputs are smoothed with an Exponential Moving Average filter before publishing.  
`output = α × new + (1 − α) × prev`

| Filter | α | Applied to |
|---|---|---|
| `arm_filter` | 0.6 | IK joint angles (shoulder, elbow) |
| `wrist_filter_l/r` | 0.6 | Wrist yaw |
| `quest_pos_filter_l/r` | 0.7 | Quest hand position input (pre-IK) |
| `neck_filter` | 0.3 | Neck yaw and pitch |
| `finger_filter` | 0.4 | All 16 finger joint commands |

All α values are adjustable in `config.py`. Filters reset to actual robot pose at the start of each SYNCING phase.

---

### Configuration (`control/config.py`)

```python
# ── Enable / disable subsystems ─────────────────────────
USE_ARM      = True    # Arm + neck IK teleoperation
USE_FINGER   = True    # Finger tracking and teleoperation
USE_SPHERE   = True    # Show hand guide spheres in Quest
FINGER_DEBUG = False   # Print live flex angles to terminal

# ── Timing ──────────────────────────────────────────────
CONTROL_HZ         = 50     # Main loop frequency [Hz]
TELEOP_START_DELAY = 5.0    # Countdown after Quest connects [s]

# ── Safety ──────────────────────────────────────────────
JUMP_THRESHOLD       = 0.15   # Hand position jump limit [m]
FREEZE_DURATION      = 2.0    # Hold time after tracking loss [s]
SYNC_DURATION        = 3.0    # Sync interpolation time [s]
SYNC_TIMEOUT         = 10.0   # Max sync wait before forcing teleop [s]
SYNC_JOINT_THRESH    = 0.1    # Sync done: joint error [rad]
SYNC_POSITION_THRESH = 0.05   # Sync done: palm position error [m]
```

| Use case | `USE_ARM` | `USE_FINGER` |
|---|---|---|
| Full teleoperation | `True` | `True` |
| Arm + neck only | `True` | `False` |
| Fingers only | `False` | `True` |

---

### ROS Topics

| Topic | Direction | Type | Description |
|---|---|---|---|
| `/arm_controller/command` | Publish | `Float64MultiArray` | 12 arm+neck joint angles [rad] |
| `/finger_controller/command` | Publish | `Float64MultiArray` | 16 finger joint angles [rad] |
| `/camera/color/image_raw` | Subscribe | `sensor_msgs/Image` | D435i camera for Quest streaming |
| `/joint_states` | Subscribe | `sensor_msgs/JointState` | Current robot joint positions |

---

### Aliases

| Command | Description |
|---|---|
| `xrgazebo` | Launch Gazebo simulation |
| `xrcamerastart` | Start RealSense D435i stream via Jetson |
| `xrcamerastop` | Stop camera stream |
| `xrteleop` | Start teleoperation (`HR_teleop.py`) |
| `xrcaliball` | Delete all calibration files |
| `xrcalib` | Delete arm calibration only |
| `xrhelp` | Show command list |