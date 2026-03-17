"""
HR_teleop.py  (메인)
─────────────────────────────────────────────────────────────
VR 텔레오퍼레이션 메인 상태 머신.
실행: python3 HR_teleop.py

상태 흐름:
  WAITING_QUEST → CALIBRATING → SYNCING → TELEOP
                                    ↑         ↓
                                 FREEZE ←────┘
"""
import sys, os, json, time
from multiprocessing import shared_memory, Queue, Event
from enum import Enum, auto

import numpy as np
import pinocchio as pin
import rospy
from scipy.spatial.transform import Rotation as Rot

import config
sys.path.insert(0, os.path.join(config.TELEVISION_DIR, 'teleop'))
sys.path.insert(0, config.CURRENT_DIR)
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from TeleVision import OpenTeleVision

from robot_model  import (build_robot_model, compute_ik,
                           extract_wrist_twist_z, calc_target_from_calib)
from motion_utils import (EMAFilter, make_filters, beep,
                           publish_smooth_move, publish_init, publish_fin)
from ros_interface import RosInterface
from finger_mapping import (build_hand_cmd, is_landmark_valid, FingerEMAFilter)


# ── 좌표 변환 헬퍼 ────────────────────────────────────────────
def robot_to_quest(robot_pos, robot_init, quest_init):
    delta = robot_pos - robot_init
    return quest_init + np.array([-delta[1], delta[2], -delta[0]])

def robot_dir_to_quest(rot_matrix, side):
    """로봇 손바닥 방향벡터를 Quest 좌표계로 변환.
    URDF 기준: 오른손=wrist Y축, 왼손=wrist -Y축.
    side: 'L' 또는 'R'
    """
    palm_normal = rot_matrix[:, 1] if side == 'R' else -rot_matrix[:, 1]
    return np.array([-palm_normal[1], palm_normal[2], -palm_normal[0]])


# ══════════════════════════════════════════════════════════════
# 상태 머신
# ══════════════════════════════════════════════════════════════
class TeleopState(Enum):
    WAITING_QUEST = auto()
    CALIBRATING   = auto()
    SYNCING       = auto()
    FREEZE        = auto()
    TELEOP        = auto()


# ══════════════════════════════════════════════════════════════
# 초기화
# ══════════════════════════════════════════════════════════════

# ── 피노키오 모델 ──────────────────────────────────────────
model, data, ids, q_init, robot_init = build_robot_model()
joint_ids    = ids['joint_ids']
L_palm_id    = ids['L_palm']
R_palm_id    = ids['R_palm']
L_joint_mask = ids['L_joint_mask']
R_joint_mask = ids['R_joint_mask']
robot_L_init = robot_init['L']
robot_R_init = robot_init['R']

# ── INIT_POS / CALIB_POS 손바닥 방향 사전 계산 (구체 화살표용) ──
def _fk_palm_pose(model, data, frame_id, q_vals):
    q_tmp = pin.neutral(model)
    for name, val in zip(config.JOINT_ORDER, q_vals):
        if model.existJointName(name):
            jid = model.getJointId(name)
            q_tmp[model.joints[jid].idx_q] = val
    pin.forwardKinematics(model, data, q_tmp)
    pin.updateFramePlacements(model, data)
    return (data.oMf[frame_id].rotation.copy(),
            data.oMf[frame_id].translation.copy())

robot_L_calib_rot, robot_L_calib_pos = _fk_palm_pose(model, data, L_palm_id, config.CALIB_POS)
robot_R_calib_rot, robot_R_calib_pos = _fk_palm_pose(model, data, R_palm_id, config.CALIB_POS)
robot_L_init_rot,  robot_L_init_pos  = _fk_palm_pose(model, data, L_palm_id, config.INIT_POS)
robot_R_init_rot,  robot_R_init_pos  = _fk_palm_pose(model, data, R_palm_id, config.INIT_POS)

# ── WAITING 구체 위치·방향 (calib 로드 후 동적 계산) ──────────
WAITING_L_POS = None
WAITING_R_POS = None
WAITING_L_DIR = robot_dir_to_quest(robot_L_init_rot, 'L')
WAITING_R_DIR = robot_dir_to_quest(robot_R_init_rot, 'R')

def _compute_waiting_pos(quest_L_init, quest_R_init):
    l_pos = robot_to_quest(robot_L_init_pos, robot_L_calib_pos, quest_L_init)
    r_pos = robot_to_quest(robot_R_init_pos, robot_R_calib_pos, quest_R_init)
    return l_pos, r_pos

# ── CALIB 구체 위치·방향 (신체 자세 기반 하드코딩) ─────────────
CALIB_L_POS = np.array([-0.25, 0.9583, -0.50])
CALIB_R_POS = np.array([ 0.25, 0.9583, -0.50])
CALIB_L_DIR = np.array([ 0.9397, -0.3420,  0.0000])
CALIB_R_DIR = np.array([-0.9397, -0.3420,  0.0000])

# ── OpenTeleVision ─────────────────────────────────────────
RES = (720, 1280)
shm = shared_memory.SharedMemory(create=True, size=RES[0] * RES[1] * 3 * 2)
image_array = np.ndarray((RES[0], RES[1] * 2, 3), dtype=np.uint8, buffer=shm.buf)
image_array[:] = 128
image_queue      = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(RES, shm.name, image_queue, toggle_streaming,
                    stream_mode="image",
                    cert_file=config.CERT_FILE,
                    key_file=config.KEY_FILE,
                    ngrok=False)

# ── ROS ───────────────────────────────────────────────────
ros = RosInterface(model, q_init, image_array)

# ── EMA 필터 ──────────────────────────────────────────────
filters            = make_filters()
arm_filter         = filters['arm']
wrist_filter_l     = filters['wrist_l']
wrist_filter_r     = filters['wrist_r']
quest_pos_filter_l = filters['quest_pos_l']
quest_pos_filter_r = filters['quest_pos_r']
neck_filter        = filters['neck']
finger_filter      = FingerEMAFilter(alpha=config.EMA_FINGER, n=16)

FINGER_NEUTRAL = [0.0] * 16

# ── 캘리브레이션 변수 ──────────────────────────────────────
quest_L_init           = None
quest_R_init           = None
quest_neck_rot_init    = np.eye(3)
quest_L_wrist_rot_init = np.eye(3)
quest_R_wrist_rot_init = np.eye(3)
calibrated             = False

calib_samples_L     = []
calib_samples_R     = []
calib_samples_L_rot = []
calib_samples_R_rot = []

# ── calib.json 로드 ────────────────────────────────────────
if os.path.exists(config.CALIB_PATH):
    with open(config.CALIB_PATH) as f:
        cd = json.load(f)
    quest_L_init           = np.array(cd['quest_L_init'])
    quest_R_init           = np.array(cd['quest_R_init'])
    quest_neck_rot_init    = np.array(cd.get('quest_neck_rot_init',    np.eye(3).tolist()))
    quest_L_wrist_rot_init = np.array(cd.get('quest_L_wrist_rot_init', np.eye(3).tolist()))
    quest_R_wrist_rot_init = np.array(cd.get('quest_R_wrist_rot_init', np.eye(3).tolist()))
    calibrated = True
    WAITING_L_POS, WAITING_R_POS = _compute_waiting_pos(quest_L_init, quest_R_init)
    print(f"[INIT] Calibration loaded  L{quest_L_init.round(3)}  R{quest_R_init.round(3)}")
else:
    if not config.USE_ARM:
        quest_L_init = np.zeros(3)
        quest_R_init = np.zeros(3)
        calibrated   = True
        print("[INIT] USE_ARM=False, skipping arm calibration")
    else:
        print("[INIT] No calibration file found. Connect Quest and extend arms forward.")

# ── 초기 q ────────────────────────────────────────────────
q = ros.wait_for_joint_states()
arm_filter.reset(q)
current_q_for_smooth = [float(q[idx]) for idx in joint_ids]

# ── 초기 손바닥 위치 (FK) → 오버레이용 ────────────────────
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
ros.overlay.update({
    'l_actual': data.oMf[L_palm_id].translation.tolist(),
    'r_actual': data.oMf[R_palm_id].translation.tolist(),
})

# ── 상태 머신 보조 변수 ────────────────────────────────────
teleop_state      = TeleopState.WAITING_QUEST
moved_to_init     = False
prev_r_raw        = None
prev_l_raw        = None
q_ref_current     = q.copy()

sync_target_q   = None
sync_start_q    = None
sync_start_time = None
l_sync_target   = None
r_sync_target   = None

tracking_lost       = False
_waiting_printed    = False
freeze_start_time   = None
_first_teleop_start = True
_countdown_start    = None
_calib_wait_start   = None

# Hz 모니터링
_prev_l = np.zeros(3)
_prev_r = np.zeros(3)
_l_update_count = _r_update_count = _frame_count = 0

_last_status_time = 0.0
STATUS_INTERVAL   = 0.5   # [s]

print("\n" + "=" * 44)
print("  HR Teleop")
fe_s = "ON" if config.USE_FINGER_FE else "OFF"
aa_s = "ON" if config.USE_FINGER_AA else "OFF"
nk_s = "ON" if config.USE_NECK      else ("TRACK" if config.USE_NECK_TRACK else "OFF")
print(f"  ARM={str(config.USE_ARM).upper()}  FINGER FE={fe_s}  FINGER AA={aa_s}  NECK={nk_s}")
print("=" * 44 + "\n")


# ══════════════════════════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════════════════════════
try:
    while not rospy.is_shutdown():
        loop_start = time.time()

        left_mat  = tv.left_hand
        right_mat = tv.right_hand
        l_raw     = left_mat[:3, 3]
        r_raw     = right_mat[:3, 3]

        # Quest Hz 모니터링 (50프레임마다 1회)
        _frame_count += 1
        if not np.array_equal(l_raw, _prev_l):
            _l_update_count += 1
            _prev_l = l_raw.copy()
        if not np.array_equal(r_raw, _prev_r):
            _r_update_count += 1
            _prev_r = r_raw.copy()
        if _frame_count % 50 == 0:
            print(f"[Quest] {tv.hand_hz:.1f}Hz | update_rate={100*_l_update_count/_frame_count:.0f}%")

        # ── 트래킹 소실 감지 ───────────────────────────────
        if config.USE_ARM and np.allclose(r_raw, 0):
            if teleop_state == TeleopState.TELEOP:
                print("[WARN] Tracking lost -> FREEZE")
                teleop_state      = TeleopState.FREEZE
                freeze_start_time = time.time()
                sync_target_q     = None
                tracking_lost     = True
                beep('warn')
            elif teleop_state == TeleopState.FREEZE:
                pass
            elif teleop_state not in (TeleopState.WAITING_QUEST,):
                print("[WARN] Tracking lost -> WAITING")
                teleop_state     = TeleopState.WAITING_QUEST
                tracking_lost    = True
                _waiting_printed = False
                _countdown_start = None
            ros.overlay.update({'state': teleop_state.name, 'countdown': -1})
            if not _waiting_printed:
                print("[WAIT] Waiting for Quest connection...")
                _waiting_printed = True
            time.sleep(1.0 / config.CONTROL_HZ)
            continue

        _waiting_printed = False

        # ── WAITING_QUEST: 카운트다운 후 다음 상태 전환 ────
        if teleop_state == TeleopState.WAITING_QUEST:
            if _countdown_start is None:
                _countdown_start = time.time()
                print(f"[WAIT] Quest connected. Starting in {int(config.TELEOP_START_DELAY)}s...")
                beep('teleop_start')

            elapsed_cd  = time.time() - _countdown_start
            remaining_s = config.TELEOP_START_DELAY - elapsed_cd
            countdown_n = max(int(remaining_s) + 1, 0)

            q_cur = ros.get_current_q()
            pin.forwardKinematics(model, data, q_cur)
            pin.updateFramePlacements(model, data)
            ros.overlay.update({
                'state':     'WAITING_QUEST',
                'countdown':  countdown_n,
                'l_actual':   data.oMf[L_palm_id].translation.tolist(),
                'r_actual':   data.oMf[R_palm_id].translation.tolist(),
            })

            if WAITING_L_POS is not None:
                tv._teleop_active.value = 1 if config.USE_SPHERE else 0
                tv.l_palm_quest = WAITING_L_POS
                tv.r_palm_quest = WAITING_R_POS
                tv.l_palm_dir   = WAITING_L_DIR
                tv.r_palm_dir   = WAITING_R_DIR

            if elapsed_cd < config.TELEOP_START_DELAY:
                time.sleep(1.0 / config.CONTROL_HZ)
                continue

            _countdown_start = None
            ros.overlay['countdown'] = 0
            if not calibrated and config.USE_ARM:
                print("[STATE] -> CALIBRATING  (extend arms forward)")
                teleop_state = TeleopState.CALIBRATING
                beep('calib_start')
            else:
                print("[STATE] -> SYNCING")
                teleop_state = TeleopState.SYNCING

        # ── 점프 감지 (TELEOP) ─────────────────────────────
        if config.USE_ARM and teleop_state == TeleopState.TELEOP and prev_r_raw is not None:
            r_jump = np.linalg.norm(r_raw - prev_r_raw)
            l_jump = np.linalg.norm(l_raw - prev_l_raw) if prev_l_raw is not None else 0
            if r_jump > config.JUMP_THRESHOLD or l_jump > config.JUMP_THRESHOLD:
                print(f"[WARN] Position jump detected  R={r_jump:.3f}m L={l_jump:.3f}m -> FREEZE")
                teleop_state      = TeleopState.FREEZE
                freeze_start_time = time.time()
                sync_target_q     = None
                tracking_lost     = True
                beep('warn')

        prev_r_raw = r_raw.copy()
        prev_l_raw = l_raw.copy()


        # ══════════════════════════════════════════════════
        # 상태: CALIBRATING
        # ══════════════════════════════════════════════════
        if teleop_state == TeleopState.CALIBRATING:
            if not moved_to_init:
                publish_init(ros.pub_arm, current_q_for_smooth)
                moved_to_init = True
                current_q_for_smooth = [float(q_init[idx]) for idx in joint_ids]
                _calib_wait_start = time.time()

            # 3초 대기: 사용자가 손을 캘리브 위치에 맞출 시간
            if _calib_wait_start is not None:
                wait_elapsed = time.time() - _calib_wait_start
                wait_remain  = max(3.0 - wait_elapsed, 0.0)
                ros.overlay.update({
                    'state':       'CALIBRATING',
                    'calib_wait':   wait_remain,
                    'calib_n':      0,
                    'calib_total':  config.CALIB_COUNT,
                })
                tv._teleop_active.value = 1 if config.USE_SPHERE else 0
                tv.l_palm_quest = CALIB_L_POS.copy()
                tv.r_palm_quest = CALIB_R_POS.copy()
                tv.l_palm_dir   = CALIB_L_DIR.copy()
                tv.r_palm_dir   = CALIB_R_DIR.copy()
                if wait_elapsed < 3.0:
                    time.sleep(1.0 / config.CONTROL_HZ)
                    continue
                _calib_wait_start = None

            r_rot = right_mat[:3, :3]
            l_rot = left_mat[:3, :3]
            if not np.allclose(l_raw, 0) and not np.allclose(r_raw, 0):
                calib_samples_R.append(r_raw.copy())
                calib_samples_L.append(l_raw.copy())
                calib_samples_R_rot.append(r_rot.copy())
                calib_samples_L_rot.append(l_rot.copy())

            ros.overlay.update({
                'state':       'CALIBRATING',
                'calib_wait':   0.0,
                'calib_n':      len(calib_samples_R),
                'calib_total':  config.CALIB_COUNT,
            })
            tv._teleop_active.value = 1 if config.USE_SPHERE else 0
            tv.l_palm_quest = CALIB_L_POS.copy()
            tv.r_palm_quest = CALIB_R_POS.copy()
            tv.l_palm_dir   = CALIB_L_DIR.copy()
            tv.r_palm_dir   = CALIB_R_DIR.copy()

            if time.time() - _last_status_time > STATUS_INTERVAL:
                print(f"[CALIB] Collecting samples ({len(calib_samples_R)}/{config.CALIB_COUNT})")
                _last_status_time = time.time()

            if (len(calib_samples_R) >= config.CALIB_COUNT and
                    len(calib_samples_L) >= config.CALIB_COUNT):
                quest_R_init           = np.mean(calib_samples_R, axis=0)
                quest_L_init           = np.mean(calib_samples_L, axis=0)
                quest_R_wrist_rot_init = calib_samples_R_rot[-1]
                quest_L_wrist_rot_init = calib_samples_L_rot[-1]
                head_rot               = tv.head_matrix[:3, :3]
                quest_neck_rot_init    = head_rot.copy()
                calibrated = True

                calib_data = {
                    'quest_L_init':           quest_L_init.tolist(),
                    'quest_R_init':           quest_R_init.tolist(),
                    'quest_neck_yaw_init':    float(np.arctan2(-head_rot[2,0],
                                                np.sqrt(head_rot[2,1]**2+head_rot[2,2]**2))),
                    'quest_neck_pitch_init':  float(np.arctan2(head_rot[2,1], head_rot[2,2])),
                    'quest_L_wrist_rot_init': quest_L_wrist_rot_init.tolist(),
                    'quest_R_wrist_rot_init': quest_R_wrist_rot_init.tolist(),
                    'quest_neck_rot_init':    quest_neck_rot_init.tolist(),
                }
                with open(config.CALIB_PATH, 'w') as f:
                    json.dump(calib_data, f)
                print(f"[CALIB] Done  L{quest_L_init.round(3)}  R{quest_R_init.round(3)}")
                beep('calib_done')
                print("[STATE] -> SYNCING")
                teleop_state = TeleopState.SYNCING

            time.sleep(1.0 / config.CONTROL_HZ)
            continue


        # ── 공통: 목/손목 계산 ─────────────────────────────
        if config.USE_ARM:
            head = tv.head_matrix
            if not np.allclose(head, 0):
                head_rot = head[:3, :3]
                R_rel = quest_neck_rot_init.T @ head_rot
                qx, qy, qz, qw = Rot.from_matrix(R_rel).as_quat()
                if qw < 0:
                    qx, qy, qz, qw = -qx, -qy, -qz, -qw
                neck_yaw   = np.clip(config.NECK_SCALE * 2.0 * np.arctan2(qy, qw),
                                     model.lowerPositionLimit[joint_ids[10]],
                                     model.upperPositionLimit[joint_ids[10]])
                neck_pitch = np.clip(config.NECK_SCALE * -2.0 * np.arctan2(qx, qw),
                                     model.lowerPositionLimit[joint_ids[11]],
                                     model.upperPositionLimit[joint_ids[11]])
            else:
                neck_yaw, neck_pitch = 0.0, 0.0

            r_rot         = right_mat[:3, :3]
            l_rot         = left_mat[:3, :3]
            r_wrist_delta = extract_wrist_twist_z(r_rot, quest_R_wrist_rot_init)
            l_wrist_delta = extract_wrist_twist_z(l_rot, quest_L_wrist_rot_init)
            r_wrist_yaw   = np.clip(config.CALIB_POS[9] + config.WRIST_SCALE * r_wrist_delta,
                                    model.lowerPositionLimit[joint_ids[9]],
                                    model.upperPositionLimit[joint_ids[9]])
            l_wrist_yaw   = np.clip(config.CALIB_POS[4] + config.WRIST_SCALE * l_wrist_delta,
                                    model.lowerPositionLimit[joint_ids[4]],
                                    model.upperPositionLimit[joint_ids[4]])


        # ══════════════════════════════════════════════════
        # 상태: FREEZE
        # ══════════════════════════════════════════════════
        if teleop_state == TeleopState.FREEZE:
            elapsed_freeze = time.time() - freeze_start_time
            remaining      = config.FREEZE_DURATION - elapsed_freeze

            if time.time() - _last_status_time > STATUS_INTERVAL:
                print(f"[FREEZE] Resuming in {remaining:.1f}s...")
                _last_status_time = time.time()
            ros.overlay.update({'state': 'FREEZE', 'freeze_remaining': max(remaining, 0.0)})

            if elapsed_freeze >= config.FREEZE_DURATION:
                print("[STATE] FREEZE released -> SYNCING")
                teleop_state  = TeleopState.SYNCING
                sync_target_q = None
            else:
                ros.publish_arm(current_q_for_smooth)
                ros.publish_hand(FINGER_NEUTRAL)
                time.sleep(1.0 / config.CONTROL_HZ)
                continue


        # ══════════════════════════════════════════════════
        # 상태: SYNCING
        # ══════════════════════════════════════════════════
        if teleop_state == TeleopState.SYNCING:

            if not config.USE_ARM:
                print("[STATE] USE_ARM=False, skipping SYNCING -> TELEOP")
                teleop_state = TeleopState.TELEOP
                beep('teleop_start' if _first_teleop_start else 'sync_done')
                _first_teleop_start = False

            else:
                if sync_target_q is None or tracking_lost:
                    q = ros.get_current_q()
                    q_ref_current = q.copy()

                    l_sync_target, r_sync_target = calc_target_from_calib(
                        l_raw, r_raw,
                        quest_L_init, quest_R_init,
                        robot_L_init, robot_R_init
                    )
                    q_sync = compute_ik(model, data, L_palm_id, l_sync_target, q,
                                        q_ref=q_ref_current, q_init=q_init,
                                        joint_mask=L_joint_mask)
                    q_sync = compute_ik(model, data, R_palm_id, r_sync_target, q_sync,
                                        q_ref=q_ref_current, q_init=q_init,
                                        joint_mask=R_joint_mask)

                    q_sync_cmd       = [float(q_sync[idx]) for idx in joint_ids]
                    q_sync_cmd[4]    = float(l_wrist_yaw)
                    q_sync_cmd[9]    = float(r_wrist_yaw)
                    q_sync_cmd[10]   = float(neck_yaw)   if config.USE_NECK else 0.0
                    q_sync_cmd[11]   = float(neck_pitch) if config.USE_NECK else 0.0

                    sync_target_q   = q_sync_cmd
                    sync_start_q    = [float(q[idx]) for idx in joint_ids]
                    sync_start_time = time.time()
                    tracking_lost   = False

                    arm_filter.reset(q)
                    wrist_filter_l.reset(np.array([float(l_wrist_yaw)]))
                    wrist_filter_r.reset(np.array([float(r_wrist_yaw)]))
                    quest_pos_filter_l.reset(l_raw)
                    quest_pos_filter_r.reset(r_raw)
                    neck_filter.reset(np.array([float(neck_yaw), float(neck_pitch)]))

                    print(f"[SYNC] Target  L{l_sync_target.round(3)}  R{r_sync_target.round(3)}")

                elapsed_sync    = time.time() - sync_start_time
                fraction        = min(elapsed_sync / config.SYNC_DURATION, 1.0)
                fraction_smooth = fraction * fraction * (3 - 2 * fraction)
                cmd_data        = [s + (t - s) * fraction_smooth
                                   for s, t in zip(sync_start_q, sync_target_q)]

                q_actual  = ros.get_current_q()
                joint_err = max(abs(float(q_actual[idx]) - t)
                                for idx, t in zip(joint_ids[:8], sync_target_q[:8]))
                pin.forwardKinematics(model, data, q_actual)
                pin.updateFramePlacements(model, data)
                l_pos_err = np.linalg.norm(data.oMf[L_palm_id].translation - l_sync_target)
                r_pos_err = np.linalg.norm(data.oMf[R_palm_id].translation - r_sync_target)

                if time.time() - _last_status_time > STATUS_INTERVAL:
                    print(f"[SYNC] {fraction*100:.0f}%  joint={joint_err:.3f}rad  L={l_pos_err:.3f}m R={r_pos_err:.3f}m")
                    _last_status_time = time.time()

                ros.overlay.update({
                    'state':        'SYNCING',
                    'sync_elapsed': elapsed_sync,
                    'sync_timeout': config.SYNC_TIMEOUT,
                })

                sync_done = (
                    (joint_err  < config.SYNC_JOINT_THRESH and
                     l_pos_err  < config.SYNC_POSITION_THRESH and
                     r_pos_err  < config.SYNC_POSITION_THRESH)
                    or fraction >= 1.0
                    or elapsed_sync >= config.SYNC_TIMEOUT
                )

                if sync_done:
                    if fraction < 1.0:
                        print("[STATE] Sync complete -> TELEOP")
                    else:
                        print("[STATE] Sync timeout, forcing -> TELEOP")
                    q = ros.get_current_q()
                    q_ref_current = q.copy()
                    arm_filter.reset(q)
                    sync_target_q = None
                    teleop_state  = TeleopState.TELEOP
                    beep('teleop_start' if _first_teleop_start else 'sync_done')
                    _first_teleop_start = False

                if np.any(np.isnan(cmd_data)):
                    time.sleep(1.0 / config.CONTROL_HZ)
                    continue

                current_q_for_smooth = cmd_data.copy()
                ros.publish_arm(cmd_data)
                ros.publish_hand(FINGER_NEUTRAL)

                elapsed = time.time() - loop_start
                time.sleep(max(0, 1.0 / config.CONTROL_HZ - elapsed))
                continue


        # ══════════════════════════════════════════════════
        # 상태: TELEOP
        # ══════════════════════════════════════════════════

        if config.USE_ARM:
            # 1. 입력 필터링
            l_filt = quest_pos_filter_l.filter(l_raw)
            r_filt = quest_pos_filter_r.filter(r_raw)

            # 2. 좌표 변환
            l_target, r_target = calc_target_from_calib(
                l_filt, r_filt,
                quest_L_init, quest_R_init,
                robot_L_init, robot_R_init
            )

            # 3. IK
            q = compute_ik(model, data, L_palm_id, l_target, q,
                           q_ref=q_ref_current, q_init=q_init, joint_mask=L_joint_mask)
            q = compute_ik(model, data, R_palm_id, r_target, q,
                           q_ref=q_ref_current, q_init=q_init, joint_mask=R_joint_mask)

            # 4. FK → 실제 손바닥 위치 + IK 오차 (오버레이용)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            l_actual = data.oMf[L_palm_id].translation.copy()
            r_actual = data.oMf[R_palm_id].translation.copy()
            l_err    = float(np.linalg.norm(l_actual - l_target))
            r_err    = float(np.linalg.norm(r_actual - r_target))

            # 5. 스무딩
            l_wrist_yaw_filt = wrist_filter_l.filter(np.array([float(l_wrist_yaw)]))[0]
            r_wrist_yaw_filt = wrist_filter_r.filter(np.array([float(r_wrist_yaw)]))[0]

            q_vis = q.copy()
            q_vis[joint_ids[4]] = l_wrist_yaw_filt
            q_vis[joint_ids[9]] = r_wrist_yaw_filt
            pin.forwardKinematics(model, data, q_vis)
            pin.updateFramePlacements(model, data)

            tv._teleop_active.value = 1 if config.USE_SPHERE else 0
            tv.l_palm_quest = robot_to_quest(l_actual, robot_L_init, quest_L_init)
            tv.r_palm_quest = robot_to_quest(r_actual, robot_R_init, quest_R_init)
            tv.l_palm_dir   = robot_dir_to_quest(data.oMf[L_palm_id].rotation, 'L')
            tv.r_palm_dir   = robot_dir_to_quest(data.oMf[R_palm_id].rotation, 'R')

            q        = arm_filter.filter(q)
            cmd_data = [float(q[idx]) for idx in joint_ids]
            cmd_data[4]  = l_wrist_yaw_filt
            cmd_data[9]  = r_wrist_yaw_filt
            if config.USE_NECK:
                neck_filt    = neck_filter.filter(np.array([float(neck_yaw), float(neck_pitch)]))
                cmd_data[10] = neck_filt[0]
                cmd_data[11] = neck_filt[1]
            elif config.USE_NECK_TRACK:
                # Head-to-midpoint tracking:
                # Compute direction from head position to the midpoint of both hands
                # in Quest frame, then extract yaw/pitch for the neck joints.
                # Quest frame convention: X=right, Y=up, Z=backward (toward user)
                head_pos = tv.head_matrix[:3, 3]
                if (not np.allclose(head_pos, 0) and
                        not np.allclose(l_raw, 0) and not np.allclose(r_raw, 0)):
                    mid = (l_raw + r_raw) * 0.5
                    d   = mid - head_pos
                    d_norm = np.linalg.norm(d)
                    if d_norm > 0.05:
                        d /= d_norm
                        # Yaw  (left-right): arctan2(X, -Z) in Quest frame
                        # Pitch (up-down):   arctan2(-Y, sqrt(X^2+Z^2))
                        nt_yaw   = np.clip(
                            config.NECK_SCALE * float(np.arctan2(-d[0], d[2])),
                            model.lowerPositionLimit[joint_ids[10]],
                            model.upperPositionLimit[joint_ids[10]])
                        nt_pitch = np.clip(
                            config.NECK_SCALE * float(np.arctan2(-d[1],
                                np.sqrt(d[0]**2 + d[2]**2))),
                            model.lowerPositionLimit[joint_ids[11]],
                            model.upperPositionLimit[joint_ids[11]])
                        nt_filt      = neck_filter.filter(np.array([nt_yaw, nt_pitch]))
                        cmd_data[10] = nt_filt[0]
                        cmd_data[11] = nt_filt[1]
                    else:
                        cmd_data[10] = 0.0
                        cmd_data[11] = 0.0
                else:
                    cmd_data[10] = 0.0
                    cmd_data[11] = 0.0
            else:
                cmd_data[10] = 0.0
                cmd_data[11] = 0.0

            # 6. 안전 체크
            if np.any(np.isnan(cmd_data)) or np.any(np.isinf(cmd_data)):
                print("[WARN] IK diverged -> FREEZE")
                q = ros.get_current_q()
                arm_filter.reset(q)
                sync_target_q     = None
                teleop_state      = TeleopState.FREEZE
                freeze_start_time = time.time()
                tracking_lost     = True
                beep('warn')
                time.sleep(1.0 / config.CONTROL_HZ)
                continue

            # 7. 팔/목 publish
            current_q_for_smooth = cmd_data.copy()
            ros.publish_arm(cmd_data)

        else:
            cmd_data = current_q_for_smooth.copy()
            l_err = r_err = 0.0

        # 8. 손가락 publish
        if config.USE_FINGER_FE or config.USE_FINGER_AA:
            left_lm  = tv.left_landmarks
            right_lm = tv.right_landmarks
            l_valid  = is_landmark_valid(left_lm)
            r_valid  = is_landmark_valid(right_lm)

            # 디버그 모드: 실시간 굽힘각 출력
            if config.FINGER_DEBUG and time.time() - _last_status_time > 0.3:
                from finger_mapping import FINGER_ANGLE_POINTS, _flex_angle
                finger_names = {1: "엄지", 2: "검지", 3: "중지", 4: "약지(로봇)←소지"}
                PINKY_RAW = (0, 21, 24)

                def _fmt(lm, side):
                    parts = []
                    for f in range(1, 5):
                        p = FINGER_ANGLE_POINTS[f]
                        parts.append(f"{side} {finger_names[f]}:{_flex_angle(lm[p[0]], lm[p[1]], lm[p[2]]):.0f}°")
                    pinky_flex = _flex_angle(lm[PINKY_RAW[0]], lm[PINKY_RAW[1]], lm[PINKY_RAW[2]])
                    parts.append(f"{side} 소지(Quest):{pinky_flex:.0f}°")
                    return parts

                if l_valid: print("  [FLEX]", "  ".join(_fmt(left_lm,  "L")))
                if r_valid: print("  [FLEX]", "  ".join(_fmt(right_lm, "R")))

            if l_valid or r_valid:
                _left_lm  = left_lm  if l_valid else np.zeros((25, 3))
                _right_lm = right_lm if r_valid else np.zeros((25, 3))

                raw_finger_cmd = build_hand_cmd(
                    _left_lm, _right_lm,
                    use_fe=config.USE_FINGER_FE,
                    use_aa=config.USE_FINGER_AA,
                )

                if not l_valid: raw_finger_cmd[:8]  = FINGER_NEUTRAL[:8]
                if not r_valid: raw_finger_cmd[8:]  = FINGER_NEUTRAL[8:]

                finger_cmd = finger_filter.filter(raw_finger_cmd)
                if not (np.any(np.isnan(finger_cmd)) or np.any(np.isinf(finger_cmd))):
                    ros.publish_hand(finger_cmd.tolist())
                else:
                    ros.publish_hand(FINGER_NEUTRAL)
            else:
                ros.publish_hand(FINGER_NEUTRAL)
        else:
            ros.publish_hand(FINGER_NEUTRAL)

        # 9. Hz 출력 + 오버레이 갱신
        elapsed   = time.time() - loop_start
        loop_time = max(elapsed, 1e-6)
        time.sleep(max(0, 1.0 / config.CONTROL_HZ - elapsed))
        ros.overlay.update({
            'state':    'TELEOP',
            'hz':       1.0 / loop_time,
            'l_err':    l_err,
            'r_err':    r_err,
            'l_joints': cmd_data[:5],
            'r_joints': cmd_data[5:10],
        })
        if time.time() - _last_status_time > STATUS_INTERVAL:
            arm_s  = f"L={l_err*100:.1f}cm R={r_err*100:.1f}cm" if config.USE_ARM else "ARM=OFF"
            fe_s   = "FE=ON"    if config.USE_FINGER_FE  else "FE=OFF"
            aa_s   = "AA=ON"    if config.USE_FINGER_AA  else "AA=OFF"
            if config.USE_NECK:
                nk_s = "NECK=ON"
            elif config.USE_NECK_TRACK:
                nk_s = "NECK=TRACK"
            else:
                nk_s = "NECK=OFF"
            print(f"[TELEOP] {1/loop_time:.1f}Hz | {arm_s} | {fe_s} {aa_s} | {nk_s}")
            if config.USE_ARM:
                print(f"[WRIST]  L={np.degrees(l_wrist_yaw):+.1f}deg  R={np.degrees(r_wrist_yaw):+.1f}deg")
            _last_status_time = time.time()


# ══════════════════════════════════════════════════════════════
# Safe shutdown
# ══════════════════════════════════════════════════════════════
except KeyboardInterrupt:
    print("\n[Interrupt] Ctrl+C received, shutting down...")

finally:
    publish_fin(ros.pub_arm,
                current_vals=cmd_data if 'cmd_data' in locals() else None,
                duration=2.5)
    ros.publish_hand(FINGER_NEUTRAL)

    if 'shm' in locals():
        shm.close()
        shm.unlink()

    print("[DONE] Shutdown complete")