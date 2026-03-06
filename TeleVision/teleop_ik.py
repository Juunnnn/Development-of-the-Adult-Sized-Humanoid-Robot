import sys
import os
import json
import time

current_dir = os.path.dirname(os.path.abspath(__file__)) # TeleVision 폴더 위치
sys.path.insert(0, os.path.join(current_dir, 'teleop'))
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from multiprocessing import shared_memory, Queue, Event
from enum import Enum, auto
import numpy as np
from scipy.spatial.transform import Rotation
import pinocchio as pin
from pinocchio import SE3
import rospy
from std_msgs.msg import Float64MultiArray
from TeleVision import OpenTeleVision

# ── 텔레옵 상태 머신 ──
class TeleopState(Enum):
    WAITING_QUEST   = auto()  # Quest 접속 대기
    CALIBRATING     = auto()  # 캘리브레이션 수집 중
    SYNCING         = auto()  # 로봇을 사람 자세에 맞게 이동 중
    FREEZE          = auto()  # 소실/이동불가 감지 → 경고음 + N초 대기 후 SYNCING
    TELEOP          = auto()  # 본격 텔레옵

# 싱크 완료 판정 임계값
SYNC_POSITION_THRESH = 0.05   # [m] 5cm 이내
SYNC_JOINT_THRESH    = 0.1   # [rad] 관절각 기준 보조 판정

# FREEZE 대기 시간: 경고음 후 이 시간만큼 현재 자세 유지 후 SYNCING으로 전환
FREEZE_DURATION = 2.0  # [s]

# ── 소리 파일 경로 (원하는 파일로 교체) ──
sound_teleop_start = '/usr/share/sounds/freedesktop/stereo/service-login.oga'     # 텔레옵 시작 (항상)
sound_warn         = '/usr/share/sounds/freedesktop/stereo/window-attention.oga'  # 소실/이동불가 경고
sound_sync_done    = '/usr/share/sounds/freedesktop/stereo/power-plug.oga'        # 싱크 완료 (텔레옵 재개)
sound_calib_start  = '/usr/share/sounds/freedesktop/stereo/service-login.oga'     # 캘리브 시작
sound_calib_done   = '/usr/share/sounds/freedesktop/stereo/service-logout.oga'    # 캘리브 완료

def beep(kind='warn'):
    """
    kind:
      'teleop_start' → 텔레옵 첫 시작 (초기 싱크 완료 포함, 항상 이 소리)
      'warn'         → 소실/이동불가 경고
      'sync_done'    → 소실 복귀 후 싱크 완료 (텔레옵 재개)
      'calib_start'  → 캘리브레이션 시작
      'calib_done'   → 캘리브레이션 완료
    """
    if kind == 'teleop_start':
        os.system(f'paplay {sound_teleop_start} &')
    elif kind == 'warn':
        os.system(f'paplay {sound_warn} &')
    elif kind == 'sync_done':
        os.system(f'paplay {sound_sync_done} &')
    elif kind == 'calib_start':
        os.system(f'paplay {sound_calib_start} &')
    elif kind == 'calib_done':
        os.system(f'paplay {sound_calib_done} &')

# ── 피노키오 모델 로드 ──
urdf_path = '/home/teleopstation/catkin_ws/src/Wholebody_39_DoF_URDF/urdf/Wholebody_39_DoF_URDF.urdf'
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# 가상 손바닥 프레임 (wrist_yaw에서 z=-0.11815 오프셋)
palm_offset = SE3(np.eye(3), np.array([0, 0, -0.11815]))
L_wrist_id = model.getFrameId('L_wrist_yaw')
R_wrist_id = model.getFrameId('R_wrist_yaw')
model.addFrame(pin.Frame('L_palm',
    model.frames[L_wrist_id].parentJoint, L_wrist_id,
    model.frames[L_wrist_id].placement * palm_offset,
    pin.FrameType.OP_FRAME))
model.addFrame(pin.Frame('R_palm',
    model.frames[R_wrist_id].parentJoint, R_wrist_id,
    model.frames[R_wrist_id].placement * palm_offset,
    pin.FrameType.OP_FRAME))
data = model.createData()
L_palm_id = model.getFrameId('L_palm')
R_palm_id = model.getFrameId('R_palm')

# ── 초기 자세 (앞으로 나란히) / 종료 자세 ──
joint_order = [
    'L_shoulder_pitch_joint', 'L_shoulder_roll_joint', 'L_shoulder_yaw_joint',
    'L_elbow_pitch_joint', 'L_wrist_yaw_joint',           # index 0~4  (L_wrist_yaw = index 4)
    'R_shoulder_pitch_joint', 'R_shoulder_roll_joint', 'R_shoulder_yaw_joint',
    'R_elbow_pitch_joint', 'R_wrist_yaw_joint',           # index 5~9  (R_wrist_yaw = index 9)
    'Neck_Yaw_Joint', 'Neck_Pitch_Joint'                   # index 10~11
]
init_vals = [-1.57, 0, 0, 0, 0, -1.57, 0, 0, 0, 0, 0, 0]
final_vals = [0, 0, 0, -1.57, 0, 0, 0, 0, -1.57, 0, 0, 0]

q_init = pin.neutral(model)
joint_ids = []
for name, val in zip(joint_order, init_vals):
    jid = model.getJointId(name)
    idx = model.joints[jid].idx_q
    q_init[idx] = val
    joint_ids.append(idx)

# 초기 자세에서 손바닥 위치 계산
pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)
robot_L_init = data.oMf[L_palm_id].translation.copy()
robot_R_init = data.oMf[R_palm_id].translation.copy()
print(f"로봇 초기 L_palm: {robot_L_init.round(3)}")
print(f"로봇 초기 R_palm: {robot_R_init.round(3)}")

def calc_target_from_calib(l_raw, r_raw):
    """
    캘리브 절대 기준으로 사람 손 위치 → 로봇 타겟 위치 계산.
    항상 quest_L_init / robot_L_init 기준. FIX5 제거.
    """
    l_target = quest_to_robot_L(l_raw, quest_L_init, robot_L_init)
    r_target = quest_to_robot_R(r_raw, quest_R_init, robot_R_init)
    return l_target, r_target

# ── Quest → 로봇 좌표 변환 ──
def quest_to_robot_R(pos, quest_init, robot_init):
    delta = pos - quest_init
    dx = -delta[2]
    dy = -delta[0]
    dz =  delta[1]
    return robot_init + np.array([dx, dy, dz])

def quest_to_robot_L(pos, quest_init, robot_init):
    delta = pos - quest_init
    dx = -delta[2]
    dy = -delta[0]
    dz =  delta[1]
    return robot_init + np.array([dx, dy, dz])

# ── IK 함수 (null space + position only) ──
def compute_ik(model, data, frame_id, target_pos, q_cur,
               q_ref=None, max_iter=50, eps=1e-3,
               null_weight=0.3, joint_mask=None):
    """
    joint_mask: 실제 IK로 제어할 관절 인덱스 리스트 (wrist_yaw 제외, 반대쪽 팔 제외)
    q_ref:      null space 기준 자세 (트래킹 시작 시점의 실제 로봇 자세)
    null_weight: null space에서 q_ref 방향으로 끌어당기는 강도

    수렴 전략:
    - adaptive λ:         에러 크기에 비례 → 특이점/작업공간 경계에서 안정적
    - adaptive step size: dq norm을 0.3rad 이하로 clamp → overshooting 방지
    - 조기 종료 (발산):   에러가 이전보다 커지면 즉시 중단
    - 조기 종료 (클램핑): 관절 한계에 막혀 수렴 불가능한 경우 감지 후 중단
    """
    q = q_cur.copy()
    if q_ref is None:
        q_ref = q_init.copy()

    prev_err = np.inf
    clamp_count = 0          # 관절 한계 클램핑 연속 횟수
    CLAMP_LIMIT = 5          # 이 횟수 이상 클램핑되면 도달 불가로 판단, 조기 종료

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id]
        pos_error = target_pos - current.translation
        err_norm = np.linalg.norm(pos_error)

        if err_norm < eps:
            break

        # 조기 종료 1: 에러가 이전 iteration보다 커지면 발산 중 → 중단
        if i > 5 and err_norm > prev_err * 1.01:
            break
        prev_err = err_norm

        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        J_pos = J[:3, :]

        # joint_mask 적용: 해당 팔 관절만 활성화, 반대쪽 팔·wrist_yaw 간섭 차단
        if joint_mask is not None:
            mask = np.zeros(J_pos.shape[1])
            mask[joint_mask] = 1.0
            J_pos = J_pos * mask

        # adaptive λ: 에러 크기에 비례 → 특이점/작업공간 경계에서 DLS 안정화
        # 에러 0.001m → λ≈1e-4 (정밀), 에러 0.1m → λ≈1e-2 (보수적)
        lam = np.clip(err_norm * 0.1, 1e-4, 5e-2)

        # DLS pseudo-inverse: J†= Jᵀ(JJᵀ + λI)⁻¹
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lam * np.eye(3))

        # null space projector: (I - J†J) → 위치 제어 후 남는 자유도 추출
        N = np.eye(len(q)) - J_dls @ J_pos

        # null space task: q_ref(시작 자세) 방향으로 끌어당김 → elbow 자연 자세 유지
        q_null = -null_weight * (q - q_ref)

        dq = J_dls @ pos_error + N @ q_null

        # joint_mask: 해당 팔 관절만 q 업데이트, 반대쪽 팔 보존
        if joint_mask is not None:
            dq_masked = np.zeros_like(dq)
            dq_masked[joint_mask] = dq[joint_mask]
            dq = dq_masked

        # adaptive step size: 한 iteration에 0.3rad 이상 이동 금지
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.3:
            dq = dq * (0.3 / dq_norm)

        q_new = pin.integrate(model, q, dq)
        q_clipped = np.clip(q_new, model.lowerPositionLimit, model.upperPositionLimit)

        # 조기 종료 2: 클램핑 감지 → 관절이 한계에 막혀 수렴 불가
        # q_new와 q_clipped 차이가 크면 관절 한계에 부딪힌 것
        if np.linalg.norm(q_new - q_clipped) > 0.01:
            clamp_count += 1
            if clamp_count >= CLAMP_LIMIT:
                break   # 타겟이 현재 팔 길이로 도달 불가 → 조기 종료
        else:
            clamp_count = 0  # 클램핑 해소되면 카운터 리셋

        q = q_clipped
    return q

# ── Swing-Twist decomposition으로 손목 yaw 추출 ──
def extract_wrist_twist_z(rot_mat, init_rot_mat):
    """
    현재 rotation과 초기 rotation의 상대 회전에서
    Z축 twist 성분만 추출 (arctan2 불연속 문제 없음)
    """
    R_rel = init_rot_mat.T @ rot_mat
    r = Rotation.from_matrix(R_rel)
    qx, qy, qz, qw = r.as_quat()
    angle = 2.0 * np.arctan2(qz, qw)
    
    # [-π, +π] 범위로 wrap
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

# ── EMA 필터 (IK 출력 스무딩) ──
class EMAFilter:
    def __init__(self, alpha, size):
        # alpha 높을수록 빠른 반응 (1=필터없음, 0=안움직임)
        self.alpha = alpha
        self.size = size
        self.prev = None

    def filter(self, q_new):
        if self.prev is None:
            self.prev = q_new.copy()
            return q_new.copy()
        self.prev = self.alpha * q_new + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self, q):
        self.prev = q.copy()

arm_filter = EMAFilter(alpha=0.6, size=len(q_init))  # 0.4→0.6: 반응속도 개선
wrist_filter_l = EMAFilter(alpha=0.6, size=1)
wrist_filter_r = EMAFilter(alpha=0.6, size=1)
quest_pos_filter_l = EMAFilter(alpha=0.7, size=3)
quest_pos_filter_r = EMAFilter(alpha=0.7, size=3)
neck_filter = EMAFilter(alpha=0.3, size=2)   # 목은 더 강하게 스무딩 (α낮을수록 부드러움)

# ── 부드러운 자세 이동 함수 ──
def publish_smooth_move(target_vals, current_q_data=None, duration=2.0, label="이동"):
    hz = 50
    steps = int(duration * hz)

    if current_q_data is None:
        # 현재 자세를 모르면 즉시 전송
        cmd = Float64MultiArray()
        cmd.data = target_vals
        pub.publish(cmd)
        return

    print(f"🎬 {duration}초 동안 {label} 자세로 부드럽게 이동합니다...")
    for i in range(1, steps + 1):
        fraction = i / steps
        interp_vals = [
            curr + (tar - curr) * fraction
            for curr, tar in zip(current_q_data, target_vals)
        ]
        cmd = Float64MultiArray()
        cmd.data = interp_vals
        pub.publish(cmd)
        time.sleep(1.0 / hz)  # rospy.Rate 대신 time.sleep 사용

def publish_init(current_q_data=None):
    publish_smooth_move(init_vals, current_q_data, duration=2.0, label="캘리브레이션(초기)")

def publish_fin(current_q_data=None, duration=2.5):
    publish_smooth_move(final_vals, current_q_data, duration=duration, label="종료")

# ── OpenTeleVision ──
resolution_cropped = (720, 1280)  # D435i to Quest 카메라 해상도
shm = shared_memory.SharedMemory(create=True, size=resolution_cropped[0] * resolution_cropped[1] * 3 * 2)
image_array = np.ndarray((resolution_cropped[0], resolution_cropped[1] * 2, 3), dtype=np.uint8, buffer=shm.buf)
image_array[:] = 128
image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming,
                    stream_mode="image",
                    cert_file="/home/teleopstation/Development of the Adult Sized Humanoid Robot/TeleVision/cert.pem",
                    key_file="/home/teleopstation/Development of the Adult Sized Humanoid Robot/TeleVision/key.pem",
                    ngrok=False)

# ── ROS ──
rospy.init_node('teleop_ik', disable_signals=True)
pub = rospy.Publisher('/arm_controller/command', Float64MultiArray, queue_size=10)
CONTROL_HZ = 50  # Quest 수신 주기 약 45Hz에 맞춤

# D435i Image -> Quest Streaming
from sensor_msgs.msg import Image, JointState  # [FIX] JointState 추가
import cv2

def camera_callback(msg):
    try:
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        frame = cv2.resize(frame, (1280, 720))
        image_array[:, :1280, :] = frame
        image_array[:, 1280:, :] = frame
    except Exception as e:
        print(f"카메라 오류: {e}")

rospy.Subscriber('/camera/color/image_raw', Image, camera_callback)

# ── [FIX] 로봇 실제 관절값 수신 ──
# 이 값을 써야 텔레옵 시작 시 로봇이 현재 자세를 유지한 채로 시작됨
# (이전엔 q_init=앞으로나란히로 하드코딩되어 있어서 시작하자마자 발사됐음)
current_joint_state = None

def joint_state_callback(msg):
    global current_joint_state
    current_joint_state = msg

rospy.Subscriber('/joint_states', JointState, joint_state_callback)

def get_current_q():
    """
    /joint_states 토픽에서 받은 실제 관절값으로 pinocchio q 벡터를 구성.
    수신 전이면 q_init으로 대체 (경고 출력).
    """
    if current_joint_state is None:
        print("⚠️  /joint_states 미수신 → q_init으로 대체 (로봇이 앞으로나란히 자세여야 안전)")
        return q_init.copy()
    q_cur = pin.neutral(model)
    for name, val in zip(current_joint_state.name, current_joint_state.position):
        if model.existJointName(name):
            jid = model.getJointId(name)
            idx = model.joints[jid].idx_q
            q_cur[idx] = val
    return q_cur

# ── 캘리브레이션 로드/초기화 ──
CALIB_PATH = os.path.join(current_dir, 'calib.json')
quest_L_init = None
quest_R_init = None
quest_neck_rot_init = np.eye(3)
quest_L_wrist_rot_init = np.eye(3)  # 왼손목 rotation matrix 기준값
quest_R_wrist_rot_init = np.eye(3)  # 오른손목 rotation matrix 기준값
calibrated = False

if os.path.exists(CALIB_PATH):
    with open(CALIB_PATH, 'r') as f:
        calib_data = json.load(f)
    quest_L_init             = np.array(calib_data['quest_L_init'])
    quest_R_init             = np.array(calib_data['quest_R_init'])
    quest_neck_yaw_init      = calib_data['quest_neck_yaw_init']
    quest_neck_pitch_init    = calib_data['quest_neck_pitch_init']
    quest_neck_rot_init = np.array(calib_data.get('quest_neck_rot_init', np.eye(3).tolist()))
    quest_L_wrist_rot_init = np.array(calib_data.get('quest_L_wrist_rot_init', np.eye(3).tolist()))
    quest_R_wrist_rot_init = np.array(calib_data.get('quest_R_wrist_rot_init', np.eye(3).tolist()))
    calibrated = True
    print("✅ 저장된 캘리브레이션 로드 완료")
    print(f"   Quest L init: {quest_L_init.round(3)}")
    print(f"   Quest R init: {quest_R_init.round(3)}")
else:
    print("📐 캘리브레이션 파일 없음 - Quest 접속 후 팔 앞으로 쭉 뻗어!")

calib_samples_L = []
calib_samples_R = []
calib_samples_L_rot = []
calib_samples_R_rot = []

CALIB_COUNT = 50 # Quest가 45Hz로 데이터를 보내니까 50샘플 = 약 1초 평균 → 안정적인 기준점

WRIST_SCALE = 1.0 # 손목 회전 스케일
NECK_SCALE = 1.0 # 목 회전 스케일 

print("=== Teleop IK 시작! ===")

# ── [FIX] /joint_states 수신 대기 후 q와 current_q_for_smooth를 실제 로봇 자세로 초기화 ──
# 기존: q = q_init.copy() → 로봇 실제 자세 무시, 텔레옵 시작 시 앞으로나란히로 발사
# 변경: /joint_states 수신 후 실제 관절값으로 초기화 → 현재 자세 그대로 텔레옵 시작
print("⏳ /joint_states 수신 대기 중... (rostopic echo /joint_states 로 토픽명 확인 필요)")
timeout = 5.0  # 최대 5초 대기
t_start = time.time()
while current_joint_state is None and not rospy.is_shutdown():
    if time.time() - t_start > timeout:
        print("⚠️  /joint_states 수신 타임아웃 → q_init으로 대체")
        break
    time.sleep(0.05)

q = get_current_q()
arm_filter.reset(q)
# current_q_for_smooth도 실제 관절값으로 초기화
# 기존: final_vals.copy() → 실제 자세와 달라서 publish_smooth_move 시 점프 가능
current_q_for_smooth = [float(q[idx]) for idx in joint_ids]
print(f"✅ 초기 q 설정: {[round(q[i], 2) for i in joint_ids]}")

moved_to_init = False

# ── 상태 머신 ──
teleop_state = TeleopState.WAITING_QUEST

# 트래킹 소실 감지
tracking_lost = False
TRACKING_RECOVER_FRAMES = 25
tracking_recover_count = 0
JUMP_THRESHOLD = 0.15
prev_r_raw = None
prev_l_raw = None

# IK 기준 자세 (트래킹 복귀 시 갱신)
q_ref_current = q.copy()

# 싱크 관련
sync_target_q = None      # 싱크 목표 관절각 (IK로 계산)
_waiting_printed = False  # Quest 대기 메시지 중복 출력 방지
freeze_start_time = None  # FREEZE 상태 진입 시각
_first_teleop_start = True  # 첫 텔레옵 시작 여부 (True면 teleop_start 소리)
sync_start_q  = None      # 싱크 시작 시점 관절각
sync_start_time = None    # 싱크 시작 시각
SYNC_DURATION    = 1.5   # 이동 완료 목표 시간 (빠르게)
SYNC_TIMEOUT     = 5.0   # 강제 시작까지 최대 대기 시간

# 루프 밖
_prev_l = np.zeros(3)
_prev_r = np.zeros(3)
_l_update_count = 0
_r_update_count = 0
_frame_count = 0

try:
    while not rospy.is_shutdown():
        loop_start = time.time()

        left_mat  = tv.left_hand
        right_mat = tv.right_hand
        l_raw = left_mat[:3, 3]
        r_raw = right_mat[:3, 3]

        # ← 여기에 추가
        print(f"L pos: {l_raw.round(3)}  R pos: {r_raw.round(3)}")

        # Quest Hz 모니터링
        _frame_count += 1
        if not np.array_equal(l_raw, _prev_l):
            _l_update_count += 1
            _prev_l = l_raw.copy()
        if not np.array_equal(r_raw, _prev_r):
            _r_update_count += 1
            _prev_r = r_raw.copy()
        if _frame_count % 50 == 0:
            quest_hz = tv.hand_hz
            update_pct = 100 * _l_update_count / _frame_count
            print(f"[Quest] 전송Hz={quest_hz:.1f}Hz | 업데이트율={update_pct:.0f}%")

        # ══════════════════════════════════════════
        # 상태: WAITING_QUEST
        # ══════════════════════════════════════════
        if np.allclose(r_raw, 0):
            if teleop_state == TeleopState.TELEOP:
                # 텔레옵 중 소실 → FREEZE (경고음 + 대기)
                print("🚨 Quest 트래킹 소실 (텔레옵 중) → FREEZE")
                teleop_state = TeleopState.FREEZE
                freeze_start_time = time.time()
                sync_target_q = None
                tracking_lost = True
                beep('warn')
                time.sleep(1.0 / CONTROL_HZ)
                continue
            elif teleop_state == TeleopState.FREEZE:
                # FREEZE 중 계속 소실 → 그냥 대기
                time.sleep(1.0 / CONTROL_HZ)
                continue
            elif teleop_state not in (TeleopState.WAITING_QUEST,):
                print("⏳ Quest 트래킹 소실 → 대기 상태로")
                teleop_state = TeleopState.WAITING_QUEST
                tracking_lost = True
                _waiting_printed = False
            if not _waiting_printed:
                print("⏳ Quest 접속 대기 중... (연결되면 자동 시작)")
                _waiting_printed = True
            time.sleep(1.0 / CONTROL_HZ)
            continue

        # Quest 연결됨 → WAITING_QUEST에서 다음 상태로 전환
        _waiting_printed = False  # 다음 소실 시 다시 출력되도록 리셋
        if teleop_state == TeleopState.WAITING_QUEST:
            if not calibrated:
                teleop_state = TeleopState.CALIBRATING
                print("📐 Quest 연결됨 → 캘리브레이션 시작")
                beep('calib_start')
            else:
                teleop_state = TeleopState.SYNCING
                print("🔄 Quest 연결됨 → 싱크 단계 시작 (손 고정 유지)")

        # ── 점프 감지 (TELEOP 상태에서만) ──
        if teleop_state == TeleopState.TELEOP:
            if prev_r_raw is not None:
                r_jump = np.linalg.norm(r_raw - prev_r_raw)
                l_jump = np.linalg.norm(l_raw - prev_l_raw) if prev_l_raw is not None else 0
                if r_jump > JUMP_THRESHOLD or l_jump > JUMP_THRESHOLD:
                    print(f"🚨 손 위치 점프! R={r_jump:.3f}m L={l_jump:.3f}m → FREEZE")
                    teleop_state = TeleopState.FREEZE
                    freeze_start_time = time.time()
                    sync_target_q = None
                    tracking_lost = True
                    beep('warn')
        prev_r_raw = r_raw.copy()
        prev_l_raw = l_raw.copy()

        # ══════════════════════════════════════════
        # 상태: CALIBRATING
        # ══════════════════════════════════════════
        if teleop_state == TeleopState.CALIBRATING:
            if not moved_to_init:
                publish_init(current_q_data=current_q_for_smooth)
                moved_to_init = True
                current_q_for_smooth = [float(q_init[idx]) for idx in joint_ids]

            r_rot = right_mat[:3, :3]
            l_rot = left_mat[:3, :3]
            if not np.allclose(l_raw, 0) and not np.allclose(r_raw, 0):
                calib_samples_R.append(r_raw.copy())
                calib_samples_L.append(l_raw.copy())
                calib_samples_R_rot.append(r_rot.copy())
                calib_samples_L_rot.append(l_rot.copy())

            print(f"📐 캘리브레이션 중... ({len(calib_samples_R)}/{CALIB_COUNT}) - 팔 앞으로 쭉 뻗어!")

            if len(calib_samples_R) >= CALIB_COUNT and len(calib_samples_L) >= CALIB_COUNT:
                quest_R_init = np.mean(calib_samples_R, axis=0)
                quest_L_init = np.mean(calib_samples_L, axis=0)
                quest_R_wrist_rot_init = calib_samples_R_rot[-1]
                quest_L_wrist_rot_init = calib_samples_L_rot[-1]
                head = tv.head_matrix
                head_rot = head[:3, :3]
                quest_neck_rot_init = head_rot.copy()
                quest_neck_yaw_init = float(np.arctan2(-head_rot[2,0], np.sqrt(head_rot[2,1]**2+head_rot[2,2]**2)))
                quest_neck_pitch_init = float(np.arctan2(head_rot[2,1], head_rot[2,2]))
                calibrated = True
                calib_data = {
                    'quest_L_init': quest_L_init.tolist(),
                    'quest_R_init': quest_R_init.tolist(),
                    'quest_neck_yaw_init': quest_neck_yaw_init,
                    'quest_neck_pitch_init': quest_neck_pitch_init,
                    'quest_L_wrist_rot_init': quest_L_wrist_rot_init.tolist(),
                    'quest_R_wrist_rot_init': quest_R_wrist_rot_init.tolist(),
                    'quest_neck_rot_init': quest_neck_rot_init.tolist()
                }
                with open(CALIB_PATH, 'w') as f:
                    json.dump(calib_data, f)
                print("✅ 캘리브레이션 완료! → 싱크 단계 시작 (손 고정 유지)")
                beep('calib_done')
                teleop_state = TeleopState.SYNCING

            time.sleep(1.0 / CONTROL_HZ)
            continue

        # ── 공통: 목/손목 매핑 (SYNCING, TELEOP 모두 사용) ──
        head = tv.head_matrix
        if not np.allclose(head, 0):
            head_rot = head[:3, :3]
            R_rel = quest_neck_rot_init.T @ head_rot
            r_neck = Rotation.from_matrix(R_rel)
            qx, qy, qz, qw = r_neck.as_quat()
            if qw < 0:
                qx, qy, qz, qw = -qx, -qy, -qz, -qw
            neck_yaw   = np.clip(NECK_SCALE * 2.0 * np.arctan2(-qz, qw),
                                model.lowerPositionLimit[joint_ids[10]],
                                model.upperPositionLimit[joint_ids[10]])
            neck_pitch = np.clip(NECK_SCALE * -2.0 * np.arctan2(qx, qw),
                                model.lowerPositionLimit[joint_ids[11]],
                                model.upperPositionLimit[joint_ids[11]])
        else:
            neck_yaw, neck_pitch = 0.0, 0.0

        r_rot = right_mat[:3, :3]
        l_rot = left_mat[:3, :3]
        r_wrist_delta = extract_wrist_twist_z(r_rot, quest_R_wrist_rot_init)
        l_wrist_delta = extract_wrist_twist_z(l_rot, quest_L_wrist_rot_init)
        r_wrist_yaw = np.clip(init_vals[9] + WRIST_SCALE * r_wrist_delta,
                              model.lowerPositionLimit[joint_ids[9]],
                              model.upperPositionLimit[joint_ids[9]])
        l_wrist_yaw = np.clip(init_vals[4] + WRIST_SCALE * l_wrist_delta,
                              model.lowerPositionLimit[joint_ids[4]],
                              model.upperPositionLimit[joint_ids[4]])

        L_joint_mask = joint_ids[0:4]
        R_joint_mask = joint_ids[5:9]

        # ══════════════════════════════════════════
        # 상태: FREEZE
        # 소실/점프/이동불가 감지 후 경고음 + N초 현재 자세 유지
        # N초 후 자동으로 SYNCING 전환
        # ══════════════════════════════════════════
        if teleop_state == TeleopState.FREEZE:
            elapsed_freeze = time.time() - freeze_start_time
            remaining = FREEZE_DURATION - elapsed_freeze
            print(f"⏸️  FREEZE 중... {remaining:.1f}초 후 복귀 시도 (손을 목표 위치에 고정)")

            if elapsed_freeze >= FREEZE_DURATION:
                print("🔄 FREEZE 해제 → SYNCING 시작")
                teleop_state = TeleopState.SYNCING
                sync_target_q = None
            else:
                # 현재 자세 그대로 유지 (마지막 cmd_data 재발행)
                if 'cmd_data' in dir() or 'cmd_data' in locals():
                    cmd = Float64MultiArray()
                    cmd.data = current_q_for_smooth
                    pub.publish(cmd)
                time.sleep(1.0 / CONTROL_HZ)
                continue

        # ══════════════════════════════════════════
        # 상태: SYNCING
        # 사람 손 위치를 캘리브 기준으로 변환해서 로봇 타겟 계산
        # 로봇을 그 타겟으로 부드럽게 이동, 도달하면 TELEOP 시작
        # ══════════════════════════════════════════
        if teleop_state == TeleopState.SYNCING:
            # 싱크 시작 시점에 타겟 IK 계산 (한 번만)
            if sync_target_q is None or tracking_lost:
                q = get_current_q()
                q_ref_current = q.copy()

                # 캘리브 절대 기준으로 현재 사람 손 위치 → 로봇이 가야 할 위치
                l_sync_target, r_sync_target = calc_target_from_calib(l_raw, r_raw)

                # IK로 목표 관절각 계산
                q_sync = compute_ik(model, data, L_palm_id, l_sync_target, q,
                                    q_ref=q_ref_current, joint_mask=L_joint_mask)
                q_sync = compute_ik(model, data, R_palm_id, r_sync_target, q_sync,
                                    q_ref=q_ref_current, joint_mask=R_joint_mask)
                # 손목/목도 목표에 포함
                q_sync_cmd = [float(q_sync[idx]) for idx in joint_ids]
                q_sync_cmd[4]  = float(l_wrist_yaw)
                q_sync_cmd[9]  = float(r_wrist_yaw)
                q_sync_cmd[10] = float(neck_yaw)
                q_sync_cmd[11] = float(neck_pitch)

                sync_target_q   = q_sync_cmd
                sync_start_q    = [float(q[idx]) for idx in joint_ids]
                sync_start_time = time.time()
                tracking_lost   = False

                # 필터 리셋
                arm_filter.reset(q)
                wrist_filter_l.reset(np.array([float(l_wrist_yaw)]))
                wrist_filter_r.reset(np.array([float(r_wrist_yaw)]))
                quest_pos_filter_l.reset(l_raw)
                quest_pos_filter_r.reset(r_raw)
                neck_filter.reset(np.array([float(neck_yaw), float(neck_pitch)]))

                print(f"🎯 싱크 타겟 계산 완료")
                print(f"   L타겟: {l_sync_target.round(3)}  R타겟: {r_sync_target.round(3)}")

            # 경과 시간 기반 보간 (부드러운 이동)
            elapsed_sync = time.time() - sync_start_time
            fraction = min(elapsed_sync / SYNC_DURATION, 1.0)
            # ease-in-out: 시작과 끝을 부드럽게
            fraction_smooth = fraction * fraction * (3 - 2 * fraction)

            cmd_data = [
                s + (t - s) * fraction_smooth
                for s, t in zip(sync_start_q, sync_target_q)
            ]

            # ── 싱크 완료 판정: 실제 로봇 관절값으로 확인 ──
            q_actual = get_current_q()
            actual_cmd = [float(q_actual[idx]) for idx in joint_ids]
            joint_err = max(abs(a - t) for a, t in zip(actual_cmd[:8], sync_target_q[:8]))

            # 손바닥 위치 오차도 확인
            pin.forwardKinematics(model, data, q_actual)
            pin.updateFramePlacements(model, data)
            l_actual_pos = data.oMf[L_palm_id].translation.copy()
            r_actual_pos = data.oMf[R_palm_id].translation.copy()
            l_pos_err = np.linalg.norm(l_actual_pos - l_sync_target)
            r_pos_err = np.linalg.norm(r_actual_pos - r_sync_target)

            print(f"🔄 싱크 중... {fraction*100:.0f}% | 관절오차={joint_err:.3f}rad | 위치오차 L={l_pos_err:.3f}m R={r_pos_err:.3f}m")

            sync_done = (
                (joint_err < SYNC_JOINT_THRESH and
                l_pos_err < SYNC_POSITION_THRESH and
                r_pos_err < SYNC_POSITION_THRESH)
                or fraction >= 1.0
                or elapsed_sync >= SYNC_TIMEOUT
            )

            if sync_done:
                if fraction >= 1.0:
                    print("⚠️  싱크 시간 초과 → 텔레옵 강제 시작")
                else:
                    print("✅ 싱크 완료! → 텔레옵 시작")
                # 텔레옵 시작 시점 q를 실제 로봇 값으로 갱신
                q = get_current_q()
                q_ref_current = q.copy()
                arm_filter.reset(q)
                sync_target_q = None
                teleop_state = TeleopState.TELEOP
                if _first_teleop_start:
                    beep('teleop_start')
                    _first_teleop_start = False
                else:
                    beep('sync_done')

            # 싱크 중에는 계산된 보간값 발행
            if np.any(np.isnan(cmd_data)):
                time.sleep(1.0 / CONTROL_HZ)
                continue

            current_q_for_smooth = cmd_data.copy()
            cmd = Float64MultiArray()
            cmd.data = cmd_data
            pub.publish(cmd)

            elapsed = time.time() - loop_start
            time.sleep(max(0, (1.0 / CONTROL_HZ) - elapsed))
            loop_time = time.time() - loop_start
            print(f"[Hz] {1/loop_time:.1f}Hz ({loop_time*1000:.1f}ms)")
            continue

        # ══════════════════════════════════════════
        # 상태: TELEOP
        # 캘리브 절대 기준으로 매 프레임 IK 계산
        # ══════════════════════════════════════════
        # 입력 필터링
        l_raw_filtered = quest_pos_filter_l.filter(l_raw)
        r_raw_filtered = quest_pos_filter_r.filter(r_raw)

        # 캘리브 절대 기준 타겟 (FIX5 제거 - 항상 quest_L_init 기준)
        l_target, r_target = calc_target_from_calib(l_raw_filtered, r_raw_filtered)
        print(f"[delta] L={(l_raw_filtered-quest_L_init).round(3)} R={(r_raw_filtered-quest_R_init).round(3)}")

        # IK
        _ik_t0 = time.time()
        q = compute_ik(model, data, L_palm_id, l_target, q, q_ref=q_ref_current, joint_mask=L_joint_mask)
        _ik_t1 = time.time()
        q = compute_ik(model, data, R_palm_id, r_target, q, q_ref=q_ref_current, joint_mask=R_joint_mask)
        _ik_t2 = time.time()
        print(f"[IK시간] L={(_ik_t1-_ik_t0)*1000:.1f}ms R={(_ik_t2-_ik_t1)*1000:.1f}ms")

        # 스무딩
        q = arm_filter.filter(q)
        cmd_data = [float(q[idx]) for idx in joint_ids]
        cmd_data[4]  = wrist_filter_l.filter(np.array([float(l_wrist_yaw)]))[0]
        cmd_data[9]  = wrist_filter_r.filter(np.array([float(r_wrist_yaw)]))[0]
        neck_filtered = neck_filter.filter(np.array([float(neck_yaw), float(neck_pitch)]))
        cmd_data[10] = neck_filtered[0]
        cmd_data[11] = neck_filtered[1]

        if np.any(np.isnan(cmd_data)) or np.any(np.isinf(cmd_data)):
            print("⚠️  IK nan/이동불가 → FREEZE")
            q = get_current_q()
            arm_filter.reset(q)
            sync_target_q = None
            teleop_state = TeleopState.FREEZE
            freeze_start_time = time.time()
            tracking_lost = True
            beep('warn')
            time.sleep(1.0 / CONTROL_HZ)
            continue

        current_q_for_smooth = cmd_data.copy()
        cmd = Float64MultiArray()
        cmd.data = cmd_data
        pub.publish(cmd)

        elapsed = time.time() - loop_start
        time.sleep(max(0, (1.0 / CONTROL_HZ) - elapsed))
        loop_time = time.time() - loop_start
        print(f"[Hz] {1/loop_time:.1f}Hz ({loop_time*1000:.1f}ms) | [L→] {l_target.round(3)} [R→] {r_target.round(3)}")
        print(f"[q] {[round(v,2) for v in cmd_data]}")

except KeyboardInterrupt:
    print("\n[Interrupt] Ctrl+C detected. Starting safe shutdown...")

finally:
    if 'cmd_data' in locals():
        publish_fin(current_q_data=cmd_data, duration=2.5)
    else:
        publish_fin()

    if 'shm' in locals():
        shm.close()
        shm.unlink()
        print("✅ Shared memory unlinked.")

    print("🧹 Safe shutdown complete. Bye!")