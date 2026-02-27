import sys
import os
import json
import time
sys.path.insert(0, '/home/teleopstation/TeleVision/teleop')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from multiprocessing import shared_memory, Queue, Event
import numpy as np
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

# ── 초기 자세 (앞으로 나란히) ──
joint_order = [
    'L_shoulder_pitch_joint', 'L_shoulder_roll_joint', 'L_shoulder_yaw_joint',
    'L_elbow_pitch_joint', 'L_wrist_yaw_joint',
    'R_shoulder_pitch_joint', 'R_shoulder_roll_joint', 'R_shoulder_yaw_joint',
    'R_elbow_pitch_joint', 'R_wrist_yaw_joint',
    'Neck_Yaw_Joint', 'Neck_Pitch_Joint'
]
init_vals = [-1.57, 0, 0, 0, 0, -1.57, 0, 0, 0, 0, 0, 0]

q_init = pin.neutral(model)
joint_ids = []
for name, val in zip(joint_order, init_vals):
    jid = model.getJointId(name)
    idx = model.joints[jid].idx_q
    q_init[idx] = val
    joint_ids.append(idx)

q = q_init.copy()
q_prev = q_init.copy()
SMOOTH = 0.6  # 0=즉시반응, 1=안움직임

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
    dy =  delta[0]   # 왼손은 부호 반전
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
CONTROL_HZ = 50 #IK 50Hz: Quest 수신 주기 약 45Hz

# ── 캘리브레이션 로드/초기화 ──
CALIB_PATH = '/home/teleopstation/TeleVision/calib.json'
quest_L_init = None
quest_R_init = None
quest_neck_yaw_init = 0.0
quest_neck_pitch_init = 0.0
calibrated = False

if os.path.exists(CALIB_PATH):
    with open(CALIB_PATH, 'r') as f:
        calib_data = json.load(f)
    quest_L_init = np.array(calib_data['quest_L_init'])
    quest_R_init = np.array(calib_data['quest_R_init'])
    quest_neck_yaw_init = calib_data['quest_neck_yaw_init']
    quest_neck_pitch_init = calib_data['quest_neck_pitch_init']
    calibrated = True
    print("✅ 저장된 캘리브레이션 로드 완료")
    print(f"   Quest L init: {quest_L_init.round(3)}")
    print(f"   Quest R init: {quest_R_init.round(3)}")
else:
    print("📐 캘리브레이션 파일 없음 - Quest 접속 후 팔 앞으로 쭉 뻗어!")

def publish_init():
    cmd = Float64MultiArray()
    cmd.data = init_vals
    pub.publish(cmd)

calib_samples_L = []
calib_samples_R = []
CALIB_COUNT = 50 # 캘리브레이션할 때 평균 내는 샘플 수. Quest가 45Hz로 데이터를 보내니까 50샘플 = 약 1초 동안 손 위치를 평균내서 기준점으로 저장. 손이 살짝 떨려도 안정적인 기준점을 잡기 위해.

print("=== Teleop IK 시작! ===")

try:
    while not rospy.is_shutdown():
        loop_start = time.time()

        left_mat  = tv.left_hand
        right_mat = tv.right_hand
        l_raw = left_mat[:3, 3]
        r_raw = right_mat[:3, 3]

        # 퀘스트에서 데이터 수신 주기 확인
        # if not hasattr(compute_ik, 'prev_r'):
        #     compute_ik.prev_r = r_raw.copy()
        #     compute_ik.last_change = time.time()
        #     compute_ik.change_intervals = []
        # if not np.allclose(r_raw, compute_ik.prev_r):
        #     now = time.time()
        #     compute_ik.change_intervals.append(now - compute_ik.last_change)
        #     compute_ik.last_change = now
        #     compute_ik.prev_r = r_raw.copy()
        #     if len(compute_ik.change_intervals) % 30 == 0:
        #         avg_hz = 1.0 / np.mean(compute_ik.change_intervals[-30:])
        #         print(f"📡 Quest 데이터 실제 업데이트: {avg_hz:.1f}Hz")

        # Quest 미접속
        if np.allclose(r_raw, 0):
            print("⏳ Quest 접속 대기 중...")
            if not calibrated:
                publish_init()
            time.sleep(1.0 / CONTROL_HZ)
            
            continue

        # 캘리브레이션
        if not calibrated:
            calib_samples_R.append(r_raw.copy())
            if not np.allclose(l_raw, 0):
                calib_samples_L.append(l_raw.copy())
            print(f"📐 캘리브레이션 중... ({len(calib_samples_R)}/{CALIB_COUNT}) - 팔 앞으로 쭉 뻗어!")
            publish_init()
            if len(calib_samples_R) >= CALIB_COUNT:
                quest_R_init = np.mean(calib_samples_R, axis=0)
                quest_L_init = np.mean(calib_samples_L, axis=0) if calib_samples_L else quest_R_init * np.array([1, 1, -1])
                head = tv.head_matrix
                head_rot = head[:3, :3]
                quest_neck_yaw_init   = np.arctan2(-head_rot[2, 0], np.sqrt(head_rot[2, 1]**2 + head_rot[2, 2]**2))
                quest_neck_pitch_init = np.arctan2(head_rot[2, 1], head_rot[2, 2])
                calibrated = True
                # 저장
                calib_data = {
                    'quest_L_init': quest_L_init.tolist(),
                    'quest_R_init': quest_R_init.tolist(),
                    'quest_neck_yaw_init': float(quest_neck_yaw_init),
                    'quest_neck_pitch_init': float(quest_neck_pitch_init)
                }
                with open(CALIB_PATH, 'w') as f:
                    json.dump(calib_data, f)
                print(f"✅ 캘리브레이션 완료 및 저장!")
                print(f"   L 샘플 수: {len(calib_samples_L)}")
                print(f"   Quest L init: {quest_L_init.round(3)}")
                print(f"   Quest R init: {quest_R_init.round(3)}")
            time.sleep(1.0 / CONTROL_HZ)
            continue

        # 목 매핑
        head = tv.head_matrix
        if not np.allclose(head, 0):
            head_rot = head[:3, :3]
            raw_yaw   = np.arctan2(-head_rot[2, 0], np.sqrt(head_rot[2, 1]**2 + head_rot[2, 2]**2))
            raw_pitch = np.arctan2(head_rot[2, 1], head_rot[2, 2])
            NECK_SCALE = 0.75
            neck_yaw   = np.clip(NECK_SCALE * (raw_yaw   - quest_neck_yaw_init),   model.lowerPositionLimit[joint_ids[10]], model.upperPositionLimit[joint_ids[10]])
            neck_pitch = np.clip(NECK_SCALE * -(raw_pitch - quest_neck_pitch_init), model.lowerPositionLimit[joint_ids[11]], model.upperPositionLimit[joint_ids[11]])
        else:
            neck_yaw, neck_pitch = 0.0, 0.0

        # 목표 위치 계산
        l_target = quest_to_robot_L(l_raw, quest_L_init, robot_L_init)
        r_target = quest_to_robot_R(r_raw, quest_R_init, robot_R_init)

        # IK
        q = compute_ik(model, data, L_palm_id, l_target, q)
        q = compute_ik(model, data, R_palm_id, r_target, q)

        # 스무딩
        q = SMOOTH * q_prev + (1 - SMOOTH) * q
        q_prev = q.copy()

        cmd_data = [float(q[idx]) for idx in joint_ids]
        cmd_data[10] = float(neck_yaw)
        cmd_data[11] = float(neck_pitch)

        if np.any(np.isnan(cmd_data)) or np.any(np.isinf(cmd_data)):
            print("⚠️  IK nan - 초기화")
            q = q_init.copy()
            q_prev = q_init.copy()
            time.sleep(1.0 / CONTROL_HZ)
            continue

        cmd = Float64MultiArray()
        cmd.data = cmd_data
        pub.publish(cmd)

        time.sleep(1.0 / CONTROL_HZ)

        loop_time = time.time() - loop_start
        print(f"[Hz] {1/loop_time:.1f}Hz ({loop_time*1000:.1f}ms) | [L→] {l_target.round(3)} [R→] {r_target.round(3)}")
        print(f"[q] {[round(v,2) for v in cmd_data]}")
        # 손목 회전 확인용
        r_rot = right_mat[:3, :3]
        roll  = np.arctan2(r_rot[2,1], r_rot[2,2])
        pitch = np.arctan2(-r_rot[2,0], np.sqrt(r_rot[2,1]**2 + r_rot[2,2]**2))
        yaw   = np.arctan2(r_rot[1,0], r_rot[0,0])
        print(f"[wrist] roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f}")

except KeyboardInterrupt:
    shm.unlink()
    print("종료")