# Teleoperation — Reference Notes

XR-based upper-body teleoperation system using Meta Quest 3S and [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision).
Operator arm, wrist, neck, and finger movements are tracked via Quest hand/head tracking and mapped to the robot through position-only inverse kinematics (IK), with optional torso-yaw compensation when shoulder roll hits its limit.

This document is a personal reference, not a setup guide — run commands and aliases are already configured locally.

---

## File Structure

```
control/
├── HR_teleop.py        # Main — state machine entry point
├── config.py            # All parameters, paths, joint order
├── robot_model.py        # Pinocchio model, IK, coordinate transform, torso compensation
├── motion_utils.py       # EMA filter, beep, smooth pose transitions
├── ros_interface.py      # ROS pub/sub, joint state reading, Quest overlay rendering
├── finger_mapping.py     # Quest hand landmarks → 16 finger joint angles
├── quest_video.py        # Intro video playback to Quest headset
└── calib.json            # Arm/wrist/neck calibration data (auto-generated on first run)

TeleVision/
├── TeleVision.py         # Quest streaming server (vuer-based), hand/head data + sphere overlay
├── cert.pem               # SSL certificate for Quest HTTPS
└── key.pem                # SSL private key
```

---

## State Machine

```
WAITING_QUEST → CALIBRATING → SYNCING → TELEOP
                                  ↑         |
                                  └── FREEZE ┘
```

| State | Description |
|---|---|
| `WAITING_QUEST` | Waiting for Quest connection. Plays intro video (if `USE_INTRO_VIDEO`), then counts down `TELEOP_START_DELAY`. |
| `CALIBRATING` | Runs only if `calib.json` doesn't exist yet. 3s pose-alignment wait, then collects `CALIB_COUNT` (50) samples of hand position + wrist/neck rotation, saves `calib.json`. |
| `SYNCING` | Interpolates from current robot pose to the operator's current hand position using quintic smoothstep easing. Exits on convergence (`SYNC_JOINT_THRESH` / `SYNC_POSITION_THRESH`) or `SYNC_TIMEOUT`. |
| `TELEOP` | Full teleoperation. IK solved every frame at `CONTROL_HZ`. AA offset is calibrated once on entry. |
| `FREEZE` | Triggered by a hand-position jump (`JUMP_THRESHOLD`) during `TELEOP`. Holds the last pose for `FREEZE_DURATION`, then returns to `SYNCING`. |

Losing hand tracking entirely (Quest reports a zero position) sends the controller back to `WAITING_QUEST` from any state except `FREEZE` itself; calibration is preserved across this, so re-entry skips straight to `SYNCING`.

On `SYNCING → TELEOP`, a short `SYNC_BLEND_DURATION` blend (also quintic) absorbs any residual error left by the sync threshold, so it isn't dumped into a single 20ms IK step.

---

## Calibration

Runs automatically when `calib.json` is missing.

1. Robot moves to `CALIB_POS` (arms forward).
2. 3-second wait for the operator to align hands with the guide spheres shown in Quest.
3. 50 samples collected (~1s) of: Quest hand position (`quest_L/R_init`), wrist rotation at the last sample (`quest_L/R_wrist_rot_init`), and head rotation (`quest_neck_rot_init`).

These become the zero-reference for every subsequent delta-position and delta-rotation computation. Delete `calib.json` to force recalibration on next launch.

A separate, lightweight calibration happens **every time `TELEOP` is entered** (not just once): `_calibrate_aa_offset()` takes a snapshot of the current finger AA (abduction/adduction) angles and stores it as a per-session offset, so wherever the hand happens to be resting becomes AA-neutral.

---

## Coordinate Transform

Quest and the robot use different axis conventions. Mapping applied in `robot_model.quest_to_robot()`:

| Quest axis | Robot axis | Meaning |
|---|---|---|
| −Z | +X | Forward / backward |
| −X | +Y | Left / right |
| +Y | +Z | Up / down |

```
delta  = quest_hand_pos − quest_init
target = robot_init + [-delta.z, -delta.x, +delta.y]
```

Only the *delta* from the calibration pose is mapped — absolute Quest position is irrelevant, so the operator doesn't need to stand in a fixed spot relative to the headset origin.

---

## Inverse Kinematics

Position-only IK (`robot_model.compute_ik`), DLS pseudo-inverse + null-space, up to 50 iterations per arm per frame:

```
J_dls = Jᵀ(JJᵀ + λI)⁻¹
dq    = J_dls·e + (I − J_dls·J)·(−w·(q − q_ref))
```

- **Adaptive λ**: scales with position error (`0.1 × err`, clipped to `[1e-4, 5e-2]`) — precise near the target, conservative far from it (e.g. near full arm extension).
- **Null-space term**: pulls the elbow back toward `q_ref` (the pose at the start of the current tracking segment) without disturbing palm position.
- **Step clamp**: any single iteration's `dq` is capped at 0.3 rad to prevent overshoot.
- **Divergence guard**: stops early if error grows for two consecutive iterations.
- **Joint-limit guard**: stops if the solution hits a joint limit 5 iterations in a row (target judged unreachable).

IK target is the **palm center** — a virtual frame offset `PALM_Z_OFFSET = −0.11815 m` along Z from `wrist_yaw`, not the wrist joint itself. Left and right arms are solved independently via `L_joint_mask` / `R_joint_mask` so one arm's IK never touches the other's joints.

Wrist yaw and neck yaw/pitch are **not** part of this IK — they're extracted directly from Quest rotation matrices (relative to the calibration rotation) in `extract_wrist_twist_z()` and `_compute_neck_wrist()`.

---

## Torso Compensation (`USE_TORSO`)

Only active if the URDF's `Waist_joint` is revolute. Compensates when a shoulder hits its roll limit and can't reach the target on its own (`robot_model.apply_torso_compensation`):

1. Checks if `shoulder_roll` is sitting at its limit.
2. Runs a small virtual FK test (`TORSO_ROLL_TEST_DELTA`) to confirm opening the roll further would actually move the palm toward the target — not just that the limit is hit.
3. If only one arm needs it, computes the required torso-yaw delta via the torso→palm Jacobian, capped at `TORSO_MAX_DELTA_PER_FRAME` per frame.
4. If both arms "need" it simultaneously, compensation is cancelled for both (direction conflict).
5. When no longer needed, torso yaw springs back toward 0: `step = max(TORSO_RETURN_MIN, |yaw| × TORSO_RETURN_GAIN)` — faster return from large angles, gentle near zero.

---

## Finger Retargeting

`finger_mapping.py` — AmazingHand-style 3-point angle mapping, no calibration required for FE (flexion/extension); a one-shot per-session offset for AA (abduction/adduction).

Landmark source: **WebXR 25-joint** skeleton (Quest 3S native hand tracking via vuer) — not MediaPipe's 21-point layout, so indices differ from most off-the-shelf hand-tracking code.

**FE (per finger):** three landmarks (wrist → MCP → tip) define a flex angle at the MCP vertex:

```
flex_angle = 180° − angle_at_vertex(wrist, MCP, tip)
```

Linearly mapped: `ANGLE_OPEN[f]` → `0 rad` (open), `ANGLE_CLOSE[f]` → `−1.501 rad` (closed).

**AA (per finger, optional via `USE_FINGER_AA`):** the MCP→PIP vector is projected into a palm-local frame (`x` = wrist→middle-MCP, `z` = palm-normal via index/pinky cross product, `y` = derived) and read as `arctan2(ly, lx)`. Tip is deliberately *not* used for AA since curling a finger pulls the tip inward and contaminates the AA reading; PIP stays close to neutral. Robot's "ring" finger channel (`finger 4`) is mapped from the operator's **pinky** landmarks, by design.

**Output (16 values):**
```
[L_AA_1, L_FE_1, L_AA_2, L_FE_2, L_AA_3, L_FE_3, L_AA_4, L_FE_4,
 R_AA_1, R_FE_1, R_AA_2, R_FE_2, R_AA_3, R_FE_3, R_AA_4, R_FE_4]
```
Finger numbering: 1=Thumb, 2=Index, 3=Middle, 4=Ring (driven by pinky landmarks).

**Tuning** — measured-angle ranges per finger, in `finger_mapping.py`:
```python
ANGLE_OPEN  = {1: 10.0,  2: 6.0,   3: 4.0,   4: 10.0}    # [deg] → FE = 0
ANGLE_CLOSE = {1: 50.0,  2: 135.0, 3: 149.0, 4: 150.0}   # [deg] → FE = FE_MIN
AA_SIGN     = {1: -1.0,  2: -1.0,  3: -1.0,  4: -1.0}    # flip if a finger moves opposite to intent
```
Enable `config.FINGER_DEBUG = True` to print live flex angles per finger to the terminal for re-tuning.

---

## EMA Smoothing

All outputs are smoothed before publishing: `output = α·new + (1−α)·prev`.

| Filter | α (config) | Applied to |
|---|---|---|
| `arm_filter` | `EMA_ARM` = 0.6 | IK output joint angles (shoulder ×3, elbow ×1, per arm) |
| `wrist_filter_l/r` | `EMA_WRIST` = 0.6 | Wrist yaw |
| `quest_pos_filter_l/r` | `EMA_QUEST_POS` = 0.7 | Quest hand position (pre-IK) |
| `neck_filter` | `EMA_NECK` = 0.3 | Neck yaw + pitch |
| `finger_filter` (`FingerEMAFilter`) | `EMA_FINGER` = 0.4 | All 16 finger joint commands |
| torso return step | `EMA_TORSO` = 0.1 | Not an EMA on output — slows the torso *return-to-zero* spring rate |

All filters are explicitly `.reset()` to the robot's actual current pose at the moment `TELEOP` is entered (end of `SYNCING`), so there's no warm-up lag carrying over stale values from the previous segment.

---

## Neck Control

Two mutually exclusive modes (`config.py`):

| Mode | Flag | Behavior |
|---|---|---|
| Direct tracking | `USE_NECK = True` | Neck yaw/pitch mirrors Quest head rotation relative to calibration, scaled by `NECK_SCALE`. |
| Auto-track hands | `USE_NECK_TRACK = True` (only checked if `USE_NECK = False`) | Neck auto-aims at the midpoint of both Quest hand positions. |
| Off | both `False` | Neck joints held at 0. |

A neck-limit warning (`YAW` / `PITCH` / `YAW + PITCH`) is raised in the overlay when either joint exceeds 90% of its URDF range, so the operator can see it coming in-headset before hitting a hard stop.

---

## Configuration (`config.py`) — Frequently Tuned Values

```python
# Subsystems
USE_ARM        = True   # Arm + wrist IK
USE_FINGER_FE  = True    # Finger flex/extend tracking
USE_FINGER_AA  = True    # Finger abduction/adduction tracking
USE_TORSO      = False   # Torso-yaw compensation (needs revolute Waist_joint)
USE_NECK       = True    # Direct neck tracking
USE_NECK_TRACK = False   # Auto hand-midpoint neck tracking (only if USE_NECK=False)
USE_SPHERE     = True    # Guide-sphere overlay in Quest

# Timing
CONTROL_HZ         = 50    # Main loop target frequency [Hz]
TELEOP_START_DELAY = 5.0   # Countdown after Quest connects [s]

# Safety
JUMP_THRESHOLD       = 0.15   # Max per-frame hand movement before FREEZE [m]
FREEZE_DURATION      = 2.0    # Hold time after tracking loss/jump [s]
SYNC_DURATION        = 5.0    # Target time for SYNCING interpolation [s]
SYNC_TIMEOUT         = 10.0   # Force TELEOP if sync hasn't converged [s]
SYNC_BLEND_DURATION  = 3      # Post-sync residual-error blend time [s]
```

---

## ROS Topics

| Topic | Direction | Type | Description |
|---|---|---|---|
| `/arm_controller/command` | Publish | `Float64MultiArray` | 12 values — shoulder×6, elbow×2, wrist_yaw×2, neck×2 |
| `/finger_controller/command` | Publish | `Float64MultiArray` | 16 finger joint angles [rad] |
| `/amazing_hand/finger_angles` | Publish | `Float32MultiArray` | Same 16 finger values, mirrored for the AmazingHand driver |
| `/torso_controller/command` | Publish | `Float64MultiArray` | 1 value — torso yaw [rad] |
| `/neck_controller/command` | Publish | `Float64MultiArray` | 2 values — yaw, pitch [rad] (for Jetson-side neck driver) |
| `/camera/color/image_raw` | Subscribe | `sensor_msgs/Image` | Camera feed, mirrored to Quest headset |
| `/joint_states` | Subscribe | `sensor_msgs/JointState` | Current robot joint positions (fallback to `q_init` if not received within `JOINT_STATE_TIMEOUT`) |

---

## Known Quirk

`ros_interface.py` defines a `CALIBRATING_FINGERS` overlay state/color, but nothing in `HR_teleop.py` currently transitions into it — AA calibration runs silently inside `SYNCING → TELEOP` instead. Leaving this note here so future-me doesn't go looking for a state that isn't reachable.
