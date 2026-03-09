#!/usr/bin/env python3
"""
mimic_fe_follower.py
─────────────────────────────────────────────────────────────
hand_controller 통합 명령 노드

hand_controller joints 순서 (24개):
  L_AA_1, L_FE_1, L_FE_follower_1,
  L_AA_2, L_FE_2, L_FE_follower_2,
  L_AA_3, L_FE_3, L_FE_follower_3,
  L_AA_4, L_FE_4, L_FE_follower_4,
  R_AA_1, R_FE_1, R_FE_follower_1,
  R_AA_2, R_FE_2, R_FE_follower_2,
  R_AA_3, R_FE_3, R_FE_follower_3,
  R_AA_4, R_FE_4, R_FE_follower_4

FE_follower = FE * 0.93 (mimic 소프트웨어 구현)
AA, FE는 /joint_states에서 읽은 현재값 그대로 relay
(VR 텔레오퍼레이션 명령은 별도 노드가 hand_controller/command에 publish)

구독: /joint_states  (sensor_msgs/JointState)
발행: /hand_controller/command  (std_msgs/Float64MultiArray)
"""

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

RATIO = 0.93

# hand_controller joints 순서와 완전히 일치해야 함
JOINTS = []
for lr in ['L', 'R']:
    for i in range(1, 5):
        JOINTS.append((f'{lr}_AA_{i}_joint',         'aa'))
        JOINTS.append((f'{lr}_FE_{i}_joint',          'fe'))
        JOINTS.append((f'{lr}_FE_follower_{i}_joint', 'follower'))

# FE_follower의 소스 FE joint 매핑
FE_SOURCE = {}
for lr in ['L', 'R']:
    for i in range(1, 5):
        FE_SOURCE[f'{lr}_FE_follower_{i}_joint'] = f'{lr}_FE_{i}_joint'

pub = None

def joint_state_callback(msg):
    name_to_pos = dict(zip(msg.name, msg.position))

    cmd = Float64MultiArray()
    data = []
    for joint_name, role in JOINTS:
        if role == 'follower':
            fe_name = FE_SOURCE[joint_name]
            val = name_to_pos.get(fe_name, 0.0) * RATIO
        else:
            val = name_to_pos.get(joint_name, 0.0)
        data.append(val)

    cmd.data = data
    pub.publish(cmd)

def main():
    global pub
    rospy.init_node('mimic_fe_follower')
    pub = rospy.Publisher(
        '/hand_controller/command',
        Float64MultiArray,
        queue_size=10
    )
    rospy.Subscriber('/joint_states', JointState, joint_state_callback)
    rospy.loginfo('mimic_fe_follower 시작: FE_follower = FE * %.2f', RATIO)
    rospy.spin()

if __name__ == '__main__':
    main()