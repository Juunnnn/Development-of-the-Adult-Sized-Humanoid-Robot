#!/usr/bin/env python3
"""
/amazing_hand/finger_angles → Arduino SG90 시리얼 제어 (양손)
아두이노 형식: "L1,L2,...,L8/R1,R2,...,R8\n"
왼손 보드: 0x40, 오른손 보드: 0x41

[확인된 방향 - 양손 동일]
구부림(fe=-1.5): motor1=10,  motor2=170
펴기  (fe=0):   motor1=130, motor2=50

[튜닝 파라미터]
ALPHA      → EMA 필터 (낮을수록 부드러움, 0.1~0.4)
MIDDLE_POS → 서보 중립 오프셋 캘리브레이션
R/L_FE_OPEN → 손 폈을 때 기준값 보정
"""
import rospy
from std_msgs.msg import Float32MultiArray
import serial
import numpy as np
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE   = 115200

# ── 튜닝 파라미터 ──────────────────────────────────────────
ALPHA    = 0.4
# ──────────────────────────────────────────────────────────

# 펼침(fe=0)일 때 130이 나오려면 MIDDLE_POS[0] = 40 필요
# motor1 = 90 + MIDDLE_POS + degrees(fe)
# 130 = 90 + 40 + 0 → MIDDLE_POS = 40
# 50  = 90 - 40 + 0 → MIDDLE_POS = -40
MIDDLE_POS = [40, -40, 40, -40, 40, -40, 40, -40]

# 오른손 매핑: 검지, 중지, 약지, 엄지
R_fe_indices = [11, 13, 15, 9]
R_aa_indices = [10, 12, 14, 8]
R_FE_OPEN    = [0.0, 0.0, 0.0, 0.0]

# 왼손 매핑: 검지, 중지, 약지, 엄지
L_fe_indices = [3, 5, 7, 1]
L_aa_indices = [2, 4, 6, 0]
L_FE_OPEN    = [0.0, 0.0, 0.0, 0.0]

current_servo_angles = [90.0] * 16  # 0~7: 왼손, 8~15: 오른손

ser = None

def compute_servo(angles, fe_indices, aa_indices, fe_open):
    """
    양손 동일 공식:
    motor1 = 90 + MIDDLE_POS + degrees(fe - aa)
    motor2 = 90 + MIDDLE_POS + degrees(-fe - aa)
    fe=-1.5 → motor1≈10, motor2≈170 (구부림) ✅
    fe=0    → motor1=130, motor2=50 (펼침) ✅
    """
    servo_angles = [90.0] * 8
    for i in range(4):
        fe = float(angles[fe_indices[i]]) - fe_open[i]
        aa = float(angles[aa_indices[i]])

        motor1_deg = np.degrees(fe - aa)
        motor2_deg = np.degrees(-fe - aa)

        servo_angles[i*2]   = float(np.clip(90 + MIDDLE_POS[i*2]   + motor1_deg, 0, 180))
        servo_angles[i*2+1] = float(np.clip(90 + MIDDLE_POS[i*2+1] + motor2_deg, 0, 180))
    return servo_angles

def angles_callback(msg):
    global current_servo_angles
    if len(msg.data) != 16 or ser is None or not ser.is_open:
        return

    angles = np.array(msg.data)

    l_servos = compute_servo(angles, L_fe_indices, L_aa_indices, L_FE_OPEN)
    r_servos = compute_servo(angles, R_fe_indices, R_aa_indices, R_FE_OPEN)

    # EMA 필터
    for i in range(8):
        current_servo_angles[i]   = ALPHA * l_servos[i] + (1-ALPHA) * current_servo_angles[i]
        current_servo_angles[i+8] = ALPHA * r_servos[i] + (1-ALPHA) * current_servo_angles[i+8]

    l_final = [int(a) for a in current_servo_angles[:8]]
    r_final = [int(a) for a in current_servo_angles[8:]]

    # 형식: "L1,L2,...,L8/R1,R2,...,R8\n"
    l_str = ','.join(map(str, l_final))
    r_str = ','.join(map(str, r_final))
    cmd = f"{l_str}/{r_str}\n"

    rospy.loginfo_throttle(1.0, f"L: {l_final}\nR: {r_final}")
    ser.write(cmd.encode())

def main():
    global ser
    rospy.init_node('amazing_hand_arduino')

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        rospy.loginfo(f"Arduino 연결: {SERIAL_PORT}")
        resp = ser.readline()
        rospy.loginfo(f"Arduino: {resp.decode().strip()}")

        # 초기 펼침 자세 (양손: motor1=130, motor2=50)
        ser.write(b'130,50,130,50,130,50,130,50/130,50,130,50,130,50,130,50\n')
        rospy.loginfo("초기 자세: 양손 펼침")
        time.sleep(1)

    except Exception as e:
        rospy.logerr(f"Arduino 연결 실패: {e}")
        return

    rospy.Subscriber('/amazing_hand/finger_angles', Float32MultiArray, angles_callback)
    rospy.loginfo("Amazing Hand Arduino 노드 시작 (양손)")
    rospy.spin()

    if ser:
        ser.close()

if __name__ == '__main__':
    main()
