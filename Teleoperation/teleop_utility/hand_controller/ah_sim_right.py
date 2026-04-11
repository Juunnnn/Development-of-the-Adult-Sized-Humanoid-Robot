#!/usr/bin/env python3
"""
Amazing Hand 오른손 시뮬레이션 (mink IK 방식)
튜닝 파라미터:
  ALPHA    = 0.2  → amazing_hand_arduino.py의 ALPHA와 동일하게
  MAX_STEP = 3도  → Arduino IDE의 maxStep=3과 동일하게
  R_FE_OPEN       → amazing_hand_arduino.py의 R_FE_OPEN과 동일하게
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DISPLAY'] = ':1'

import mujoco
import mujoco.viewer
import threading
import numpy as np
from pathlib import Path
import mink
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation
import rospy
from std_msgs.msg import Float32MultiArray

ROOT_PATH = Path("/home/Jetson/catkin_ws/src/handtrack_amazing_hand/Demo/AHSimulation/AHSimulation")

# ── 튜닝 파라미터 ──────────────────────────────────────────
ALPHA    = 0.85
MAX_STEP = np.radians(3)
# ──────────────────────────────────────────────────────────

R_fe_indices = [11, 13, 15, 9]
R_aa_indices = [10, 12, 14, 8]
R_FE_OPEN    = [0.2, 0.2, 0.2, 0.2]

latest_angles = None
angles_lock = threading.Lock()

def angles_callback(msg):
    global latest_angles
    if len(msg.data) == 16:
        with angles_lock:
            latest_angles = np.array(msg.data)

def fe_aa_to_quat(fe, aa, finger_idx, neutral_quat):
    abd = -aa
    pitch = fe if finger_idx == 3 else -fe
    r_delta = Rotation.from_euler('XYZ', [abd, pitch, 0.0])
    r_neutral = Rotation.from_quat([neutral_quat[1], neutral_quat[2],
                                     neutral_quat[3], neutral_quat[0]])
    r_result = r_neutral * r_delta
    q = r_result.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT_PATH / "AH_Right/mjcf/scene.xml"))
    configuration = mink.Configuration(model)

    posture_task = mink.PostureTask(model, cost=1e-2)
    tip_tasks = [
        mink.FrameTask('tip1', 'site', position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        mink.FrameTask('tip2', 'site', position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        mink.FrameTask('tip3', 'site', position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
        mink.FrameTask('tip4', 'site', position_cost=0.0, orientation_cost=1.0, lm_damping=1.0),
    ]
    tasks = [mink.EqualityConstraintTask(model, cost=1000.0), posture_task] + tip_tasks

    data = configuration.data
    configuration.update_from_keyframe("zero")
    posture_task.set_target_from_configuration(configuration)

    for i in range(4):
        mink.move_mocap_to_frame(model, data, f"finger{i+1}_target", f"tip{i+1}", "site")

    extend_angles = [-0.95, -0.95, -0.95, 0.95]
    for i in range(4):
        q_current = data.mocap_quat[i]
        r_current = Rotation.from_quat([q_current[1], q_current[2],
                                         q_current[3], q_current[0]])
        r_extend = Rotation.from_euler('Y', extend_angles[i])
        r_new = r_current * r_extend
        q = r_new.as_quat()
        data.mocap_quat[i] = [q[3], q[0], q[1], q[2]]

    neutral_quats = data.mocap_quat.copy()

    # EMA용 현재 FE/AA값
    current_fe = [0.0] * 4
    current_aa = [0.0] * 4

    rate = RateLimiter(frequency=50.0)

    rospy.init_node('ah_sim_right')
    rospy.Subscriber('/amazing_hand/finger_angles', Float32MultiArray, angles_callback)
    print("오른손 시뮬레이션 시작")

    fe_indices = [11, 13, 15, 9]
    aa_indices = [10, 12, 14, 8]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not rospy.is_shutdown():
            with angles_lock:
                angles = latest_angles.copy() if latest_angles is not None else None

            if angles is not None:
                for i in range(4):
                    fe_target = float(angles[fe_indices[i]]) - R_FE_OPEN[i]
                    aa_target = float(angles[aa_indices[i]])

                    # MAX_STEP 제한
                    fe_diff = np.clip(fe_target - current_fe[i], -MAX_STEP, MAX_STEP)
                    aa_diff = np.clip(aa_target - current_aa[i], -MAX_STEP, MAX_STEP)
                    current_fe[i] += fe_diff
                    current_aa[i] += aa_diff

                    # EMA 필터
                    current_fe[i] = ALPHA * current_fe[i] + (1 - ALPHA) * current_fe[i]
                    current_aa[i] = ALPHA * current_aa[i] + (1 - ALPHA) * current_aa[i]

                    quat = fe_aa_to_quat(current_fe[i], current_aa[i], i, neutral_quats[i])
                    data.mocap_quat[i] = quat

            for i, task in enumerate(tip_tasks):
                task.set_target(
                    mink.SE3.from_mocap_name(model, data, f"finger{i+1}_target")
                )

            vel = mink.solve_ik(configuration, tasks, rate.dt, "daqp", 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            viewer.sync()
            rate.sleep()

if __name__ == '__main__':
    main()