"""
teleop_ik.py  (메인)
─────────────────────────────────────────────────────────────
VR 텔레오퍼레이션 메인 상태 머신.
실행: python3 teleop_ik.py

상태 흐름:
  WAITING_QUEST → CALIBRATING → CALIBRATING_FINGERS → SYNCING → TELEOP
                                                            ↑         ↓
                                                         FREEZE ←────┘
"""
import sys, os, json, time
from multiprocessing import shared_memory, Queue, Event
from enum import Enum, auto

import numpy as np
import pinocchio as pin
import rospy
from std_msgs.msg import Float64MultiArray

import config
sys.path.insert(0, os.path.join(config.TELEVISION_DIR, 'teleop'))
sys.path.insert(0, config.CURRENT_DIR)
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from TeleVision import OpenTeleVision

from robot_model   import (build_robot_model, compute_ik,
                            extract_wrist_twist_z, calc_target_from_calib)
from motion_utils  import (EMAFilter, make_filters, beep,
                            publish_smooth_move, publish_init, publish_fin)
from ros_interface import RosInterface
from finger_mapping import (build_hand_cmd, is_landmark_valid,
                             FingerEMAFilter, compute_finger_calib)
from scipy.spatial.transform import Rotation as Rot

def robot_to_quest(robot_pos, robot_init, quest_init):
    delta = robot_pos - robot_init
    return quest_init + np.array([-delta[1], delta[2], -delta[0]])

def robot_dir_to_quest(rot_matrix, side):
    """로봇 손바닥 방향벡터를 Quest 좌표계로 변환.
    URDF 기준: 오른손=wrist Y축, 왼손=wrist -Y축.
    side: 'L' 또는 'R'
    """
    if side == 'R':
        palm_normal = rot_matrix[:, 1]    # +Y
    else:
        palm_normal = -rot_matrix[:, 1]   # -Y
    return np.array([-palm_normal[1], palm_normal[2], -palm_normal[0]])

# ══════════════════════════════════════════════════════════════
# 상태 머신
# ══════════════════════════════════════════════════════════════
class TeleopState(Enum):
    WAITING_QUEST       = auto()
    CALIBRATING         = auto()
    CALIBRATING_FINGERS = auto()
    SYNCING             = auto()
    FREEZE              = auto()
    TELEOP              = auto()


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
    """q_vals(JOINT_ORDER 순서)로 FK → (rotation 3x3, translation 3,) 반환."""
    import pinocchio as _pin
    q_tmp = _pin.neutral(model)
    for name, val in zip(config.JOINT_ORDER, q_vals):
        if model.existJointName(name):
            jid = model.getJointId(name)
            q_tmp[model.joints[jid].idx_q] = val
    _pin.forwardKinematics(model, data, q_tmp)
    _pin.updateFramePlacements(model, data)
    return (data.oMf[frame_id].rotation.copy(),
            data.oMf[frame_id].translation.copy())

robot_L_calib_rot, robot_L_calib_pos = _fk_palm_pose(model, data, L_palm_id, config.CALIB_POS)
robot_R_calib_rot, robot_R_calib_pos = _fk_palm_pose(model, data, R_palm_id, config.CALIB_POS)
robot_L_init_rot,  robot_L_init_pos  = _fk_palm_pose(model, data, L_palm_id, config.INIT_POS)
robot_R_init_rot,  robot_R_init_pos  = _fk_palm_pose(model, data, R_palm_id, config.INIT_POS)

# ── WAITING 구체 위치·방향 (calib 로드 후 동적 계산, 없으면 None) ──
# INIT_POS 자세 손바닥 위치를 calib 기준으로 Quest 좌표계로 변환
# calib.json이 없으면 None → WAITING_QUEST 상태에서 구체 미표시
WAITING_L_POS = None
WAITING_R_POS = None
WAITING_L_DIR = robot_dir_to_quest(robot_L_init_rot, 'L')   # 방향은 calib 무관
WAITING_R_DIR = robot_dir_to_quest(robot_R_init_rot, 'R')

def _compute_waiting_pos(quest_L_init, quest_R_init):
    """calib.json 로드 후 호출. INIT_POS 자세 손바닥 위치를 Quest 좌표계로 변환."""
    l_pos = robot_to_quest(robot_L_init_pos, robot_L_calib_pos, quest_L_init)
    r_pos = robot_to_quest(robot_R_init_pos, robot_R_calib_pos, quest_R_init)
    return l_pos, r_pos

# CALIB 구체 위치·방향 (하드코딩 유지 - 신체 자세 기반)
CALIB_L_POS   = np.array([-0.25, 0.9583, -0.50])
CALIB_R_POS   = np.array([ 0.25, 0.9583, -0.50])
CALIB_L_DIR   = np.array([ 0.9397, -0.3420,  0.0000])
CALIB_R_DIR   = np.array([-0.9397, -0.3420,  0.0000])

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

# v4 각도 방식: 캘리브 불필요 → 더미값으로 초기화
L_calib_left         = np.zeros(4)
L_calib_right        = np.zeros(4)
finger_calib_done    = True   # v4: 캘리브 스킵
finger_calib_started = False
finger_calib_samples_L = []
finger_calib_samples_R = []

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
    print(f"✅ 캘리브 로드 완료  L{quest_L_init.round(3)}  R{quest_R_init.round(3)}")
    print(f"   WAITING 구체  L{WAITING_L_POS.round(3)}  R{WAITING_R_POS.round(3)}")
else:
    print("📐 캘리브 없음 → Quest 접속 후 팔 앞으로 쭉 뻗어!")

# v4 각도 방식: finger_calib.json 불필요 → 로드 스킵
print("✅ 손가락 캘리브 스킵 (v4 각도 방식: 캘리브 불필요)")

calib_samples_L     = []
calib_samples_R     = []
calib_samples_L_rot = []
calib_samples_R_rot = []

# ── 초기 q ────────────────────────────────────────────────
q = ros.wait_for_joint_states()
arm_filter.reset(q)
current_q_for_smooth = [float(q[idx]) for idx in joint_ids]

# ── 초기 손바닥 위치 (FK) → 카운트다운 중 오버레이용 ─────
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
_countdown_start    = None    # Quest 첫 접속 시각 (카운트다운 기준)
_calib_wait_start   = None    # CALIBRATING 진입 후 3초 대기 시각

# Hz 모니터링
_prev_l = np.zeros(3)
_prev_r = np.zeros(3)
_l_update_count = _r_update_count = _frame_count = 0

# 주기적 출력용 (매 프레임 출력 방지)
_last_status_time = 0.0
STATUS_INTERVAL   = 0.5   # [s] 이 간격으로만 상태 출력

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  🎮 Teleop IK 시작!")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


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

        # Quest Hz 모니터링 (50프레임마다 1회 출력)
        _frame_count += 1
        if not np.array_equal(l_raw, _prev_l):
            _l_update_count += 1
            _prev_l = l_raw.copy()
        if not np.array_equal(r_raw, _prev_r):
            _r_update_count += 1
            _prev_r = r_raw.copy()
        if _frame_count % 50 == 0:
            print(f"[Quest] {tv.hand_hz:.1f}Hz | 업데이트율 {100*_l_update_count/_frame_count:.0f}%")

        # ── 트래킹 소실 감지 ───────────────────────────────
        if np.allclose(r_raw, 0):
            if teleop_state == TeleopState.TELEOP:
                print("🚨 트래킹 소실 → FREEZE")
                teleop_state      = TeleopState.FREEZE
                freeze_start_time = time.time()
                sync_target_q     = None
                tracking_lost     = True
                beep('warn')
            elif teleop_state == TeleopState.FREEZE:
                pass
            elif teleop_state not in (TeleopState.WAITING_QUEST,):
                print("⏳ 트래킹 소실 → 대기")
                teleop_state     = TeleopState.WAITING_QUEST
                tracking_lost    = True
                _waiting_printed = False
                _countdown_start = None   # 재접속 시 카운트다운 처음부터
            ros.overlay.update({'state': teleop_state.name, 'countdown': -1})
            if not _waiting_printed:
                print("⏳ Quest 접속 대기 중...")
                _waiting_printed = True
            time.sleep(1.0 / config.CONTROL_HZ)
            continue

        _waiting_printed = False

        # WAITING_QUEST → 카운트다운 후 다음 상태 전환
        if teleop_state == TeleopState.WAITING_QUEST:
            # Quest가 이제 막 접속됐으면 카운트다운 시작
            if _countdown_start is None:
                _countdown_start = time.time()
                print(f"🎮 Quest 접속 확인! {int(config.TELEOP_START_DELAY)}초 후 시작...")
                beep('teleop_start')

            elapsed_cd  = time.time() - _countdown_start
            remaining_s = config.TELEOP_START_DELAY - elapsed_cd
            countdown_n = max(int(remaining_s) + 1, 0)  # 5→4→3→2→1

            # 현재 로봇 손바닥 위치 실시간 갱신 (로봇이 INIT_POS로 이동 중일 수 있음)
            q_cur = ros.get_current_q()
            pin.forwardKinematics(model, data, q_cur)
            pin.updateFramePlacements(model, data)
            l_actual_cd = data.oMf[L_palm_id].translation.copy()
            r_actual_cd = data.oMf[R_palm_id].translation.copy()
            ros.overlay.update({
                'state':     'WAITING_QUEST',
                'countdown':  countdown_n,
                'l_actual':   l_actual_cd.tolist(),
                'r_actual':   r_actual_cd.tolist(),
            })

            # Quest 3D 공간에 구체로 표시
            # calib.json 로드 후 동적 계산된 값 사용 (없으면 미표시)
            if WAITING_L_POS is not None:
                tv._teleop_active.value = 1 if config.USE_SPHERE else 0
                tv.l_palm_quest = WAITING_L_POS
                tv.r_palm_quest = WAITING_R_POS
                tv.l_palm_dir   = WAITING_L_DIR
                tv.r_palm_dir   = WAITING_R_DIR

            if elapsed_cd < config.TELEOP_START_DELAY:
                # 아직 카운트다운 중 → 루프 계속
                time.sleep(1.0 / config.CONTROL_HZ)
                continue

            # 카운트다운 완료 → 다음 상태로
            _countdown_start = None
            ros.overlay['countdown'] = 0
            if not calibrated:
                print("📐 CALIBRATING 시작 - 팔 앞으로 쭉 뻗어!")
                teleop_state = TeleopState.CALIBRATING
                beep('calib_start')
            elif not finger_calib_done and config.USE_FINGER:
                print("🖐️  CALIBRATING_FINGERS 시작")
                teleop_state = TeleopState.CALIBRATING_FINGERS
                beep('calib_start')
            else:
                print("🔄 SYNCING 시작")
                teleop_state = TeleopState.SYNCING

        # ── 점프 감지 (TELEOP) ─────────────────────────────
        if teleop_state == TeleopState.TELEOP and prev_r_raw is not None:
            r_jump = np.linalg.norm(r_raw - prev_r_raw)
            l_jump = np.linalg.norm(l_raw - prev_l_raw) if prev_l_raw is not None else 0
            if r_jump > config.JUMP_THRESHOLD or l_jump > config.JUMP_THRESHOLD:
                print(f"🚨 손 위치 점프 R={r_jump:.3f}m L={l_jump:.3f}m → FREEZE")
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

            # ── 3초 대기: 사용자가 손을 캘리브 위치에 맞출 시간 ──
            if _calib_wait_start is not None:
                wait_elapsed = time.time() - _calib_wait_start
                wait_remain  = max(3.0 - wait_elapsed, 0.0)
                ros.overlay.update({
                    'state':       'CALIBRATING',
                    'calib_wait':   wait_remain,
                    'calib_n':      0,
                    'calib_total':  config.CALIB_COUNT,
                })
                # 구체: calib_pos 위치 + 방향 (하드코딩, 항상 표시)
                tv._teleop_active.value = 1 if config.USE_SPHERE else 0
                tv.l_palm_quest = CALIB_L_POS.copy()
                tv.r_palm_quest = CALIB_R_POS.copy()
                tv.l_palm_dir   = CALIB_L_DIR.copy()
                tv.r_palm_dir   = CALIB_R_DIR.copy()
                if wait_elapsed < 3.0:
                    time.sleep(1.0 / config.CONTROL_HZ)
                    continue
                _calib_wait_start = None  # 대기 완료

            r_rot = right_mat[:3, :3]
            l_rot = left_mat[:3, :3]
            if not np.allclose(l_raw, 0) and not np.allclose(r_raw, 0):
                calib_samples_R.append(r_raw.copy())
                calib_samples_L.append(l_raw.copy())
                calib_samples_R_rot.append(r_rot.copy())
                calib_samples_L_rot.append(l_rot.copy())

            # 오버레이 업데이트
            ros.overlay.update({
                'state':      'CALIBRATING',
                'calib_wait': 0.0,
                'calib_n':    len(calib_samples_R),
                'calib_total': config.CALIB_COUNT,
            })
            # 구체: calib_pos 위치 + 방향 (수집 중에도 유지, 하드코딩)
            tv._teleop_active.value = 1 if config.USE_SPHERE else 0
            tv.l_palm_quest = CALIB_L_POS.copy()
            tv.r_palm_quest = CALIB_R_POS.copy()
            tv.l_palm_dir   = CALIB_L_DIR.copy()
            tv.r_palm_dir   = CALIB_R_DIR.copy()
            # 진행상황 0.5초마다 출력
            if time.time() - _last_status_time > STATUS_INTERVAL:
                print(f"  📐 캘리브 수집 중 ({len(calib_samples_R)}/{config.CALIB_COUNT})")
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
                print(f"✅ 팔 캘리브 완료  L{quest_L_init.round(3)}  R{quest_R_init.round(3)}")
                beep('calib_done')
                if config.USE_FINGER:
                    teleop_state = TeleopState.CALIBRATING_FINGERS
                else:
                    print("🔄 SYNCING 시작")
                    teleop_state = TeleopState.SYNCING

            time.sleep(1.0 / config.CONTROL_HZ)
            continue


        # ══════════════════════════════════════════════════
        # 상태: CALIBRATING_FINGERS
        # ══════════════════════════════════════════════════
        if teleop_state == TeleopState.CALIBRATING_FINGERS:
            left_lm  = tv.left_landmarks
            right_lm = tv.right_landmarks

            if not finger_calib_started:
                print("🤖 손가락 캘리브 자세로 이동 중...")
                print("   팔꿈치 90° 구부리고, 손바닥이 Quest 카메라를 향하게, 손가락 쭉 펴기!")
                publish_smooth_move(ros.pub_arm, config.FINGER_CALIB_VALS,
                                    current_q_for_smooth, duration=2.0,
                                    label="손가락캘리브자세")
                ros.publish_hand(FINGER_NEUTRAL)
                finger_calib_started = True
                print("⏳ 4초 후 수집 시작...")
                time.sleep(4.0)
                continue

            if is_landmark_valid(left_lm) and is_landmark_valid(right_lm):
                finger_calib_samples_L.append(left_lm.copy())
                finger_calib_samples_R.append(right_lm.copy())

            ros.overlay.update({
                'state':        'CALIBRATING_FINGERS',
                'finger_n':     len(finger_calib_samples_L),
                'finger_total': config.FINGER_CALIB_COUNT,
            })
            if time.time() - _last_status_time > STATUS_INTERVAL:
                print(f"  🖐️  손가락 캘리브 수집 중 ({len(finger_calib_samples_L)}/{config.FINGER_CALIB_COUNT})")
                _last_status_time = time.time()

            if (len(finger_calib_samples_L) >= config.FINGER_CALIB_COUNT and
                    len(finger_calib_samples_R) >= config.FINGER_CALIB_COUNT):
                avg_L = np.mean(finger_calib_samples_L, axis=0)
                avg_R = np.mean(finger_calib_samples_R, axis=0)
                L_calib_left  = compute_finger_calib(avg_L)
                L_calib_right = compute_finger_calib(avg_R)
                finger_calib_done = True

                with open(config.FINGER_CALIB_PATH, 'w') as f:
                    json.dump({
                        'L_calib_left':  L_calib_left.tolist(),
                        'L_calib_right': L_calib_right.tolist(),
                    }, f)

                print(f"✅ 손가락 캘리브 완료  L{(L_calib_left*1000).round(1)}mm  R{(L_calib_right*1000).round(1)}mm")
                current_q_for_smooth = list(config.FINGER_CALIB_VALS)
                beep('calib_done')
                teleop_state = TeleopState.SYNCING

            time.sleep(1.0 / config.CONTROL_HZ)
            continue


        # ── 공통: 목/손목 계산 ─────────────────────────────
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
                print(f"  ⏸️  FREEZE {remaining:.1f}초 후 복귀 시도")
                _last_status_time = time.time()
            ros.overlay.update({'state': 'FREEZE', 'freeze_remaining': max(remaining, 0.0)})

            if elapsed_freeze >= config.FREEZE_DURATION:
                print("🔄 FREEZE 해제 → SYNCING")
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
                q_sync_cmd[10]   = float(neck_yaw)
                q_sync_cmd[11]   = float(neck_pitch)

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

                print(f"🎯 싱크 타겟  L{l_sync_target.round(3)}  R{r_sync_target.round(3)}")

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

            # 0.5초마다 진행상황 출력
            if time.time() - _last_status_time > STATUS_INTERVAL:
                print(f"  🔄 싱크 {fraction*100:.0f}% | 관절 {joint_err:.3f}rad | 위치 L{l_pos_err:.3f} R{r_pos_err:.3f}m")
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
                print("✅ 싱크 완료 → 텔레옵 시작!" if fraction < 1.0
                      else "⚠️  싱크 타임아웃 → 텔레옵 강제 시작")
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

        # 3-1. FK → 실제 손바닥 위치 + IK 오차 계산 (오버레이용)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        l_actual = data.oMf[L_palm_id].translation.copy()
        r_actual = data.oMf[R_palm_id].translation.copy()
        l_err    = float(np.linalg.norm(l_actual - l_target))
        r_err    = float(np.linalg.norm(r_actual - r_target))

        # 4. 스무딩 (wrist는 방향 시각화에도 쓰이므로 먼저 계산)
        l_wrist_yaw_filt = wrist_filter_l.filter(np.array([float(l_wrist_yaw)]))[0]
        r_wrist_yaw_filt = wrist_filter_r.filter(np.array([float(r_wrist_yaw)]))[0]

        # wrist_yaw 필터 후 값으로 FK → 방향 벡터 계산 (로봇과 동일한 값 사용)
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
        neck_filt    = neck_filter.filter(np.array([float(neck_yaw), float(neck_pitch)]))
        cmd_data[10] = neck_filt[0]
        cmd_data[11] = neck_filt[1]

        # 5. 안전 체크
        if np.any(np.isnan(cmd_data)) or np.any(np.isinf(cmd_data)):
            print("⚠️  IK 발산 → FREEZE")
            q = ros.get_current_q()
            arm_filter.reset(q)
            sync_target_q     = None
            teleop_state      = TeleopState.FREEZE
            freeze_start_time = time.time()
            tracking_lost     = True
            beep('warn')
            time.sleep(1.0 / config.CONTROL_HZ)
            continue

        # 6. 팔/목 publish
        current_q_for_smooth = cmd_data.copy()
        ros.publish_arm(cmd_data)

        # 7. 손가락 publish (한 손만 보여도 동작, 안 보이는 손은 NEUTRAL 유지)
        left_lm  = tv.left_landmarks
        right_lm = tv.right_landmarks
        l_valid  = is_landmark_valid(left_lm)
        r_valid  = is_landmark_valid(right_lm)

        if l_valid or r_valid:
            # 유효하지 않은 쪽은 이전 명령 유지를 위해 NEUTRAL로 대체
            _left_lm  = left_lm  if l_valid else np.zeros((25, 3))
            _right_lm = right_lm if r_valid else np.zeros((25, 3))

            raw_finger_cmd = build_hand_cmd(_left_lm, _right_lm, L_calib_left, L_calib_right)

            # 유효하지 않은 손 쪽(8개)은 NEUTRAL로 덮어쓰기
            if not l_valid:
                raw_finger_cmd[:8]  = FINGER_NEUTRAL[:8]
            if not r_valid:
                raw_finger_cmd[8:] = FINGER_NEUTRAL[8:]

            finger_cmd = finger_filter.filter(raw_finger_cmd)
            if not (np.any(np.isnan(finger_cmd)) or np.any(np.isinf(finger_cmd))):
                ros.publish_hand(finger_cmd.tolist())
            else:
                ros.publish_hand(FINGER_NEUTRAL)
        else:
            ros.publish_hand(FINGER_NEUTRAL)

        # 8. Hz 출력 (0.5초마다) + 오버레이 갱신
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
            print(f"  [TELEOP] {1/loop_time:.1f}Hz | L err={l_err*100:.1f}cm  R err={r_err*100:.1f}cm")
            print(f"  [WRIST] l={np.degrees(l_wrist_yaw):.1f}°  r={np.degrees(r_wrist_yaw):.1f}°")
            print(f"  [DIR_L] {tv.l_palm_dir.round(3)}  [DIR_R] {tv.r_palm_dir.round(3)}")
            _last_status_time = time.time()


# ══════════════════════════════════════════════════════════════
# 안전 종료
# ══════════════════════════════════════════════════════════════
except KeyboardInterrupt:
    print("\n[Interrupt] Ctrl+C → 안전 종료 시작...")

finally:
    publish_fin(ros.pub_arm,
                current_vals=cmd_data if 'cmd_data' in locals() else None,
                duration=2.5)
    ros.publish_hand(FINGER_NEUTRAL)

    if 'shm' in locals():
        shm.close()
        shm.unlink()

    print("🧹 종료 완료")