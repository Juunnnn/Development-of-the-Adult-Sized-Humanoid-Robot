#!/usr/bin/env python3
"""
/amazing_hand/finger_angles → Arduino SG90 시리얼 제어
"""
import rospy
from std_msgs.msg import Float32MultiArray
import serial
import numpy as np
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE   = 115200

# Amazing_Hand_Demo_sg92r.ino의 MiddlePos와 동일하게
MIDDLE_POS = [3, 0, -5, -8, -2, 5, -12, 0]

# 매핑: 검지, 중지, 약지, 엄지
fe_indices = [11, 13, 15, 9]
aa_indices = [10, 12, 14, 8]

ser = None

def angles_callback(msg):
    if len(msg.data) != 16 or ser is None or not ser.is_open:
        return

    angles = np.array(msg.data)
    servo_angles = [90] * 8

    for i in range(4):
        fe = float(angles[fe_indices[i]])
        aa = float(angles[aa_indices[i]])

        # 차동 변환 (라디안 → 도)
        motor1_deg = np.degrees(fe - aa)
        motor2_deg = np.degrees(-fe -aa)

        servo_angles[i*2]   = int(np.clip(90 + MIDDLE_POS[i*2]   + motor1_deg, 0, 180))
        servo_angles[i*2+1] = int(np.clip(90 + MIDDLE_POS[i*2+1] + motor2_deg, 0, 180))

    cmd = ' '.join(map(str, servo_angles)) + '\n'
    rospy.loginfo_throttle(1.0, f"Servo: {servo_angles}")
    ser.write(cmd.encode())

def main():
    global ser
    rospy.init_node('amazing_hand_arduino')

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # 아두이노 리셋 대기
        rospy.loginfo(f"Arduino 연결: {SERIAL_PORT}")
        # 초기 응답 읽기
        resp = ser.readline()
        rospy.loginfo(f"Arduino: {resp.decode().strip()}")
    except Exception as e:
        rospy.logerr(f"Arduino 연결 실패: {e}")
        return

    rospy.Subscriber('/amazing_hand/finger_angles', Float32MultiArray, angles_callback)
    rospy.loginfo("Amazing Hand Arduino 노드 시작 — /amazing_hand/finger_angles 대기 중")
    rospy.spin()

    if ser:
        ser.close()

if __name__ == '__main__':
    main()