import sys
import os
import json
import time

current_dir = os.path.dirname(os.path.abspath(__file__)) # TeleVision 폴더 위치
sys.path.insert(0, os.path.join(current_dir, 'teleop'))
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from multiprocessing import shared_memory, Queue, Event
import numpy as np
from scipy.spatial.transform import Rotation
import pinocchio as pin
from pinocchio import SE3
import rospy
from std_msgs.msg import Float64MultiArray
from TeleVision import OpenTeleVision

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

q = q_init.copy()
q_prev = q_init.copy()
ARM_SMOOTH = 0.6  # 0=즉시반응, 1=안움직임
WRIST_SMOOTH = 0.3 

# 초기 자세에서 손바닥 위치 계산
pin.forwardKinematics(model, data, q_init)
pin.updateFramePlacements(model, data)
robot_L_init = data.oMf[L_palm_id].translation.copy()
robot_R_init = data.oMf[R_palm_id].translation.copy()
print(f"로봇 초기 L_palm: {robot_L_init.round(3)}")
print(f"로봇 초기 R_palm: {robot_R_init.round(3)}")

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
    dy =  delta[0]
    dz =  delta[1]
    return robot_init + np.array([dx, dy, dz])

# ── IK 함수 ──
def compute_ik(model, data, frame_id, target_pos, q_cur, max_iter=50, eps=1e-3):
    q = q_cur.copy()
    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id]
        pos_error = target_pos - current.translation
        if np.linalg.norm(pos_error) < eps:
            break
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        J_pos = J[:3, :]
        lam = 1e-3
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lam * np.eye(3))
        dq = J_dls @ pos_error
        q = pin.integrate(model, q, dq * 0.3)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
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
resolution_cropped = (480, 640)
shm = shared_memory.SharedMemory(create=True, size=resolution_cropped[0] * resolution_cropped[1] * 3 * 2)
image_array = np.ndarray((resolution_cropped[0], resolution_cropped[1] * 2, 3), dtype=np.uint8, buffer=shm.buf)
image_array[:] = 128
image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming, ngrok=False)

# ── ROS ──
rospy.init_node('teleop_ik', disable_signals=True)
pub = rospy.Publisher('/arm_controller/command', Float64MultiArray, queue_size=10)
CONTROL_HZ = 50  # Quest 수신 주기 약 45Hz에 맞춤

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

WRIST_SCALE = 1.25 # 손목 회전 스케일
NECK_SCALE = 1.0 # 목 회전 스케일 

print("=== Teleop IK 시작! ===")

moved_to_init = False
current_q_for_smooth = final_vals.copy()
prev_l_wrist = float(q_init[joint_ids[4]])
prev_r_wrist = float(q_init[joint_ids[9]])

try:
    while not rospy.is_shutdown():
        loop_start = time.time()

        left_mat  = tv.left_hand
        right_mat = tv.right_hand
        l_raw = left_mat[:3, 3]
        r_raw = right_mat[:3, 3]

        # ── Quest 미접속 ──
        if np.allclose(r_raw, 0):
            print("⏳ Quest 접속 대기 중...")
            if not calibrated and not moved_to_init:
                publish_init(current_q_data=current_q_for_smooth)
                moved_to_init = True
                current_q_for_smooth = [float(q_init[idx]) for idx in joint_ids]
            time.sleep(1.0 / CONTROL_HZ)
            continue

        # ── 캘리브레이션 ──
        if not calibrated:
            if not moved_to_init:
                publish_init(current_q_data=current_q_for_smooth)
                moved_to_init = True
                current_q_for_smooth = [float(q_init[idx]) for idx in joint_ids]

            calib_samples_R.append(r_raw.copy())
            if not np.allclose(l_raw, 0):
                calib_samples_L.append(l_raw.copy())

            # 손목 roll 기준값 수집
            r_rot = right_mat[:3, :3]
            l_rot = left_mat[:3, :3]
            calib_samples_R_rot.append(r_rot.copy())
            if not np.allclose(l_raw, 0):
                calib_samples_L_rot.append(l_rot.copy())

            print(f"📐 캘리브레이션 중... ({len(calib_samples_R)}/{CALIB_COUNT}) - 팔 앞으로 쭉 뻗어!")

            if len(calib_samples_R) >= CALIB_COUNT:
                quest_R_init = np.mean(calib_samples_R, axis=0)
                quest_L_init = np.mean(calib_samples_L, axis=0) if calib_samples_L else quest_R_init * np.array([1, 1, -1])
                quest_R_wrist_rot_init = calib_samples_R_rot[-1]
                quest_L_wrist_rot_init = calib_samples_L_rot[-1] if calib_samples_L_rot else np.eye(3)

                head = tv.head_matrix
                head_rot = head[:3, :3]
                quest_neck_yaw_init   = np.arctan2(-head_rot[2, 0], np.sqrt(head_rot[2, 1]**2 + head_rot[2, 2]**2))
                quest_neck_pitch_init = np.arctan2(head_rot[2, 1], head_rot[2, 2])
                quest_neck_rot_init = head[:3, :3].copy()
                calibrated = True

                calib_data = {
                    'quest_L_init':            quest_L_init.tolist(),
                    'quest_R_init':            quest_R_init.tolist(),
                    'quest_neck_yaw_init':     float(quest_neck_yaw_init),
                    'quest_neck_pitch_init':   float(quest_neck_pitch_init),
                    'quest_L_wrist_rot_init': quest_L_wrist_rot_init.tolist(),
                    'quest_R_wrist_rot_init': quest_R_wrist_rot_init.tolist(),
                    'quest_neck_rot_init': quest_neck_rot_init.tolist()
                }
                with open(CALIB_PATH, 'w') as f:
                    json.dump(calib_data, f)
                print(f"✅ 캘리브레이션 완료 및 저장!")

            time.sleep(1.0 / CONTROL_HZ)
            continue

        # ── 목 매핑 ──
        head = tv.head_matrix
        if not np.allclose(head, 0):
            head_rot = head[:3, :3]
            raw_yaw   = np.arctan2(-head_rot[2, 0], np.sqrt(head_rot[2, 1]**2 + head_rot[2, 2]**2))
            raw_pitch = np.arctan2(head_rot[2, 1], head_rot[2, 2])

            # ±π 점프 방지: 상대 rotation의 quaternion으로 변환 후 추출
            R_rel = quest_neck_rot_init.T @ head_rot
            r = Rotation.from_matrix(R_rel)
            qx, qy, qz, qw = r.as_quat()
            if qw < 0:
                qx, qy, qz, qw = -qx, -qy, -qz, -qw

            # 기존 축 방향 그대로 유지하면서 불연속 제거
            neck_yaw   = np.clip(NECK_SCALE * 2.0 * np.arctan2(-qz, qw),
                                model.lowerPositionLimit[joint_ids[10]],
                                model.upperPositionLimit[joint_ids[10]])
            neck_pitch = np.clip(NECK_SCALE * -2.0 * np.arctan2(qx, qw),
                                model.lowerPositionLimit[joint_ids[11]],
                                model.upperPositionLimit[joint_ids[11]])
        else:
            neck_yaw, neck_pitch = 0.0, 0.0

        # ── 손목 yaw 매핑 (Quest roll → 로봇 wrist_yaw) ──
        r_rot = right_mat[:3, :3]
        l_rot = left_mat[:3, :3]
        r_wrist_delta = extract_wrist_twist_z(r_rot, quest_R_wrist_rot_init)
        l_wrist_delta = extract_wrist_twist_z(l_rot, quest_L_wrist_rot_init)

        r_wrist_yaw = np.clip(
            init_vals[9] + WRIST_SCALE * r_wrist_delta,
            model.lowerPositionLimit[joint_ids[9]],
            model.upperPositionLimit[joint_ids[9]]
        )
        l_wrist_yaw = np.clip(
            init_vals[4] + WRIST_SCALE * l_wrist_delta,
            model.lowerPositionLimit[joint_ids[4]],
            model.upperPositionLimit[joint_ids[4]]
        )

        # ── 목표 위치 계산 ──
        l_target = quest_to_robot_L(l_raw, quest_L_init, robot_L_init)
        r_target = quest_to_robot_R(r_raw, quest_R_init, robot_R_init)

        # ── IK ──
        q = compute_ik(model, data, L_palm_id, l_target, q)
        q = compute_ik(model, data, R_palm_id, r_target, q)

        # ── 스무딩 ──
        q = ARM_SMOOTH * q_prev + (1 - ARM_SMOOTH) * q
        q_prev = q.copy()

        cmd_data = [float(q[idx]) for idx in joint_ids]
        cmd_data[4]  = WRIST_SMOOTH * prev_l_wrist + (1 - WRIST_SMOOTH) * float(l_wrist_yaw)
        cmd_data[9]  = WRIST_SMOOTH * prev_r_wrist + (1 - WRIST_SMOOTH) * float(r_wrist_yaw)
        cmd_data[10] = float(neck_yaw)
        cmd_data[11] = float(neck_pitch)

        if np.any(np.isnan(cmd_data)) or np.any(np.isinf(cmd_data)):
            print("⚠️  IK nan - 초기화")
            q = q_init.copy()
            q_prev = q_init.copy()
            time.sleep(1.0 / CONTROL_HZ)
            continue

        current_q_for_smooth = cmd_data.copy()

        cmd = Float64MultiArray()
        cmd.data = cmd_data
        print(f"[wrist] delta={r_wrist_delta:.3f} → cmd={r_wrist_yaw:.3f}")
        pub.publish(cmd)

        time.sleep(1.0 / CONTROL_HZ)

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