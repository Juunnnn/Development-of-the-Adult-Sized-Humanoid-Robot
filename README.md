# Development-of-a-21(+)-DoF-Adult-Sized-Humanoid-Robot
Development of a 21+ Degrees of Freedom Adult Sized Humanoid Robot by Sungjun Lee @Robotics Innovatory Lab, Sungkyunkwan University

![39DOF](https://github.com/user-attachments/assets/fbc37dc6-26eb-4b8e-a24a-afc2e2e71929)

## Teleoperation

XR-based whole-body teleoperation system using Meta Quest and [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision).  
The operator's arm, wrist, neck, and finger movements are tracked in real time and mapped to the robot via inverse kinematics (IK).

### Directory Structure

```
Teleoperation/
├── TeleVision/          # OpenTeleVision (Quest streaming server)
│   ├── teleop/          # TeleVision core (TeleVision.py, etc.)
│   ├── cert.pem
│   └── key.pem
└── control/             # Robot control code
    ├── teleop_ik.py         # Main — state machine entry point
    ├── config.py            # All parameters and paths (edit here)
    ├── robot_model.py       # Pinocchio model, IK, coordinate transform
    ├── motion_utils.py      # EMA filter, beep, smooth move
    ├── ros_interface.py     # ROS publishers/subscribers
    ├── finger_mapping.py    # Hand landmark → finger joint angles
    ├── mimic_fe_follower.py # FE_follower mimic node (run separately)
    ├── calib.json           # Arm calibration (auto-generated)
    └── finger_calib.json    # Finger calibration (auto-generated)
```

### Requirements

- ROS Noetic
- Python 3.8+
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision) (included in `TeleVision/`)
- Meta Quest (3 / 3S) with browser access
- Intel RealSense D435i (for camera streaming)

### Quick Start

Run each command in a separate terminal.

**Terminal 1 — Gazebo simulation**
```bash
xrgazebo
```

**Terminal 2 — Camera stream** (optional, for video feed in Quest)
```bash
xrcamerastart
```

**Terminal 3 — Teleoperation**
```bash
xrteleop
```

Then connect from the Quest browser:
```
https://localhost:8012?ws=wss://localhost:8012
```

### State Machine

```
WAITING_QUEST → CALIBRATING → CALIBRATING_FINGERS → SYNCING → TELEOP
                                                          ↑         ↓
                                                       FREEZE ←────┘
```

| State | Description |
|---|---|
| `WAITING_QUEST` | Waiting for Quest connection |
| `CALIBRATING` | Collecting 50 frames with arms extended → saves `calib.json` |
| `CALIBRATING_FINGERS` | Collecting 50 frames with fingers extended → saves `finger_calib.json` |
| `SYNCING` | Moving robot to match current hand position (ease-in-out interpolation) |
| `TELEOP` | Full teleoperation — IK computed every frame |
| `FREEZE` | Tracking lost or jump detected — holds current pose for 2s, then re-syncs |

Calibration files are saved automatically and reloaded on the next run, so **calibration only needs to be done once** unless the setup changes.

### Configuration

All parameters are in `control/config.py`. Key options:

```python
# Enable/disable subsystems
USE_ARM    = True    # Arm + neck IK teleoperation
USE_FINGER = True    # Finger tracking and teleoperation

# Control rate
CONTROL_HZ = 50      # Main loop frequency [Hz]

# EMA smoothing (0 = max smooth, 1 = no filter)
EMA_ARM       = 0.6
EMA_WRIST     = 0.6
EMA_QUEST_POS = 0.7
EMA_NECK      = 0.3
EMA_FINGER    = 0.4

# Safety thresholds
JUMP_THRESHOLD       = 0.15   # [m] Hand jump detection
FREEZE_DURATION      = 2.0    # [s] Hold time after tracking loss
SYNC_POSITION_THRESH = 0.05   # [m] Sync completion threshold
```

| Use case | `USE_ARM` | `USE_FINGER` |
|---|---|---|
| Full teleoperation | `True` | `True` |
| Arm only (no fingers) | `True` | `False` |
| Fingers only | `False` | `True` |

### Calibration Reset

```bash
xrcaliball      # Reset both arm and finger calibration
xrcalib         # Reset arm calibration only
xrfingercalib   # Reset finger calibration only
```

### Calibration Procedure

**Arm calibration** (runs automatically if `calib.json` is missing):
1. Stand in front of the robot
2. Extend both arms straight forward, parallel to the ground
3. Hold the pose — 50 frames are collected automatically (~1 second)

**Finger calibration** (runs automatically if `finger_calib.json` is missing):
1. Bend elbows to 90°, palms facing the Quest cameras
2. Extend all fingers fully
3. Hold the pose — 50 frames are collected automatically

### Mimic FE Follower

The robot's finger FE_follower joints must be driven separately.  
Run this node alongside `teleop_ik.py`:

```bash
rosrun <your_package> mimic_fe_follower.py
```

This node subscribes to `/joint_states` and publishes to `/hand_controller/command`,  
computing `FE_follower = FE × 0.93` for all 8 fingers.

### Aliases

| Command | Description |
|---|---|
| `xrgazebo` | Launch Gazebo simulation |
| `xrcamerastart` | Start RealSense D435i stream via Jetson |
| `xrcamerastop` | Stop camera stream |
| `xrteleop` | Start teleoperation |
| `xrcaliball` | Delete all calibration files |
| `xrcalib` | Delete arm calibration only |
| `xrfingercalib` | Delete finger calibration only |
| `xrhelp` | Show command list |