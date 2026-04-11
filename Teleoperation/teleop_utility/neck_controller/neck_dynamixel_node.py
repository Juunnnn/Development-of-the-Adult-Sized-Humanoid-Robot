#!/usr/bin/env python3
"""
neck_dynamixel_node.py  —  teleop_neck_controller
──────────────────────────────────────────────────
/neck_controller/command (Float64MultiArray [yaw_rad, pitch_rad]) 토픽을
subscribe해서 Dynamixel XM430 모터로 전송합니다.

의존성 설치:
  pip install dynamixel-sdk

실행 전 환경변수 (Station이 ROS 마스터인 경우):
  export ROS_MASTER_URI=http://192.168.0.1:11311
  export ROS_IP=192.168.0.20

실행:
  cd ~/teleop_utility/neck_controller
  python3 neck_dynamixel_node.py
"""

import math
import sys
import os

# 같은 폴더의 config.py를 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import rospy
from std_msgs.msg import Float64MultiArray

try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError:
    raise ImportError(
        "dynamixel_sdk 패키지가 없습니다.\n"
        "설치: pip install dynamixel-sdk"
    )

# ── Dynamixel 제어 테이블 주소 (XM430, Protocol 2.0) ──────────
ADDR_TORQUE_ENABLE        = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_GOAL_POSITION        = 116


def rad_to_tick(angle_rad: float) -> int:
    """rad → XM430 tick 변환.  0 rad = 2048 tick."""
    return int((angle_rad / (2.0 * math.pi)) * 4096 + 2048)


class NeckDynamixelNode:

    def __init__(self):
        rospy.init_node("neck_dynamixel_node", anonymous=False)

        # ── 이전 명령값 (dead zone 비교용) ───────────────────
        self._prev_yaw_rad   = None
        self._prev_pitch_rad = None

        # ── Dynamixel 초기화 ──────────────────────────────────
        self._port   = PortHandler(config.DEVICE_PORT)
        self._packet = PacketHandler(config.PROTOCOL_VERSION)

        if not self._port.openPort():
            rospy.logfatal(f"[Neck] 포트 열기 실패: {config.DEVICE_PORT}")
            raise RuntimeError(f"Cannot open port: {config.DEVICE_PORT}")

        if not self._port.setBaudRate(config.BAUD_RATE):
            rospy.logfatal(f"[Neck] 보드레이트 설정 실패: {config.BAUD_RATE}")
            raise RuntimeError("Cannot set baud rate")

        self._torque_on(config.MOTOR_ID_YAW)
        self._torque_on(config.MOTOR_ID_PITCH)

        # 홈 이동 — 느린 프로파일로 부드럽게
        self._set_profile(config.MOTOR_ID_YAW,
                          config.HOME_PROFILE_VEL_YAW,
                          config.HOME_PROFILE_ACC_YAW)
        self._set_profile(config.MOTOR_ID_PITCH,
                          config.HOME_PROFILE_VEL_PITCH,
                          config.HOME_PROFILE_ACC_PITCH)

        self._send_rad(config.MOTOR_ID_YAW,   0.0)
        self._send_rad(config.MOTOR_ID_PITCH, 0.0)
        rospy.loginfo("[Neck] 홈(0 rad)으로 이동 중... 2초 대기")
        rospy.sleep(2.0)

        # 텔레옵 추종용 프로파일로 전환
        self._set_profile(config.MOTOR_ID_YAW,
                          config.TELEOP_PROFILE_VEL_YAW,
                          config.TELEOP_PROFILE_ACC_YAW)
        self._set_profile(config.MOTOR_ID_PITCH,
                          config.TELEOP_PROFILE_VEL_PITCH,
                          config.TELEOP_PROFILE_ACC_PITCH)
        rospy.loginfo("[Neck] 초기화 완료 → 텔레옵 대기")

        # ── ROS subscriber ────────────────────────────────────
        rospy.Subscriber(
            config.NECK_TOPIC,
            Float64MultiArray,
            self._neck_cb,
            queue_size=1,
            buff_size=2**16,
        )
        rospy.loginfo(f"[Neck] {config.NECK_TOPIC} 수신 대기 중...")
        rospy.loginfo(
            f"[Neck] Dead zone — YAW: {math.degrees(config.DEAD_ZONE_YAW):.2f}°  "
            f"PITCH: {math.degrees(config.DEAD_ZONE_PITCH):.2f}°"
        )

        rospy.on_shutdown(self._shutdown)

    # ── 콜백 ──────────────────────────────────────────────────
    def _neck_cb(self, msg: Float64MultiArray):
        """[yaw_rad, pitch_rad] 수신 → dead zone 체크 → 모터 전송."""
        if len(msg.data) < 2:
            rospy.logwarn_once("[Neck] 메시지 크기 < 2, 무시")
            return

        yaw_rad   = config.YAW_DIRECTION   * float(msg.data[0])
        pitch_rad = config.PITCH_DIRECTION * float(msg.data[1])

        # ── Dead zone 체크 ────────────────────────────────────
        # 처음 수신이면 바로 전송 (prev가 None)
        send_yaw   = True
        send_pitch = True

        if self._prev_yaw_rad is not None:
            if abs(yaw_rad - self._prev_yaw_rad) < config.DEAD_ZONE_YAW:
                send_yaw = False

        if self._prev_pitch_rad is not None:
            if abs(pitch_rad - self._prev_pitch_rad) < config.DEAD_ZONE_PITCH:
                send_pitch = False

        if send_yaw:
            self._send_rad(config.MOTOR_ID_YAW, yaw_rad)
            self._prev_yaw_rad = yaw_rad

        if send_pitch:
            self._send_rad(config.MOTOR_ID_PITCH, pitch_rad)
            self._prev_pitch_rad = pitch_rad

    # ── 내부 헬퍼 ─────────────────────────────────────────────
    def _send_rad(self, motor_id: int, angle_rad: float):
        """rad → tick 변환 + 클램핑 + 모터 전송."""
        tick = rad_to_tick(angle_rad)

        if motor_id == config.MOTOR_ID_YAW:
            tick = max(config.YAW_TICK_MIN, min(tick, config.YAW_TICK_MAX))
        else:
            tick = max(config.PITCH_TICK_MIN, min(tick, config.PITCH_TICK_MAX))

        result, err = self._packet.write4ByteTxRx(
            self._port, motor_id, ADDR_GOAL_POSITION, tick
        )
        if result != COMM_SUCCESS:
            rospy.logwarn_throttle(
                1.0,
                f"[Neck] ID={motor_id} write 실패: "
                f"{self._packet.getTxRxResult(result)}"
            )
        elif err != 0:
            rospy.logwarn_throttle(
                1.0,
                f"[Neck] ID={motor_id} 에러: "
                f"{self._packet.getRxPacketError(err)}"
            )

    def _torque_on(self, motor_id: int):
        result, _ = self._packet.write1ByteTxRx(
            self._port, motor_id, ADDR_TORQUE_ENABLE, 1
        )
        if result != COMM_SUCCESS:
            rospy.logwarn(f"[Neck] 토크 ON 실패 (ID={motor_id})")

    def _set_profile(self, motor_id: int, velocity: int, acceleration: int):
        """ProfileAcceleration → ProfileVelocity 순서로 설정."""
        self._packet.write4ByteTxRx(
            self._port, motor_id, ADDR_PROFILE_ACCELERATION, acceleration
        )
        self._packet.write4ByteTxRx(
            self._port, motor_id, ADDR_PROFILE_VELOCITY, velocity
        )

    def _shutdown(self):
        rospy.loginfo("[Neck] 종료 → 홈(0 rad) 복귀 후 토크 OFF")
        self._set_profile(config.MOTOR_ID_YAW,
                          config.HOME_PROFILE_VEL_YAW,
                          config.HOME_PROFILE_ACC_YAW)
        self._set_profile(config.MOTOR_ID_PITCH,
                          config.HOME_PROFILE_VEL_PITCH,
                          config.HOME_PROFILE_ACC_PITCH)
        self._send_rad(config.MOTOR_ID_YAW,   0.0)
        self._send_rad(config.MOTOR_ID_PITCH, 0.0)

        import time; time.sleep(2.0)

        self._packet.write1ByteTxRx(
            self._port, config.MOTOR_ID_YAW, ADDR_TORQUE_ENABLE, 0
        )
        self._packet.write1ByteTxRx(
            self._port, config.MOTOR_ID_PITCH, ADDR_TORQUE_ENABLE, 0
        )
        self._port.closePort()
        rospy.loginfo("[Neck] 종료 완료")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    NeckDynamixelNode().spin()