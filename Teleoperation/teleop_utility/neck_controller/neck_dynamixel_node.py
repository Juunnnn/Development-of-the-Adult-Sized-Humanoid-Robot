#!/usr/bin/env python3
"""
neck_dynamixel_node.py  —  teleop_neck_controller
──────────────────────────────────────────────────
/neck_controller/command (Float64MultiArray [yaw_rad, pitch_rad]) 토픽을
subscribe해서 Dynamixel XM430 모터로 전송합니다.

[그리퍼 추가] 목(yaw/pitch)과 같은 U2D2 버스에 물린 그리퍼용 XM430도
같은 프로세스/같은 PortHandler에서 함께 제어합니다. 시리얼 포트는
한 프로세스만 열어야 해서(동시에 두 프로세스가 같은 포트를 열면 패킷이
깨질 수 있음) 별도 노드로 분리하지 않고 여기에 통합했습니다.
/gripper_controller/command (Float64MultiArray [grip_ratio]) 를 subscribe.
  grip_ratio: 0.0 = 완전히 열림, 1.0 = 완전히 닫힘
config.USE_GRIPPER = True 로 설정하고 아래 config 항목들을 채우면 활성화됩니다:
  MOTOR_ID_GRIPPER, GRIPPER_TOPIC, GRIPPER_TICK_OPEN, GRIPPER_TICK_CLOSE,
  HOME_PROFILE_VEL_GRIPPER, HOME_PROFILE_ACC_GRIPPER,
  TELEOP_PROFILE_VEL_GRIPPER, TELEOP_PROFILE_ACC_GRIPPER, DEAD_ZONE_GRIPPER

실행:
  cd ~/teleop_neck_controller
  python3 neck_dynamixel_node.py
"""

import math
import sys
import os

# DynamixelSDK 경로 (pip 없이 소스로 설치한 경우)
sys.path.insert(0, '/home/gene/DynamixelSDK/python/src')

# 같은 폴더의 config.py를 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import rospy
from std_msgs.msg import Float64MultiArray, Bool

try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError:
    raise ImportError(
        "dynamixel_sdk 패키지가 없습니다.\n"
        "설치: pip install dynamixel-sdk\n"
        "또는 소스: sys.path에 DynamixelSDK/python/src 추가"
    )

# ── Dynamixel 제어 테이블 주소 (XM430, Protocol 2.0) ──────────
ADDR_OPERATING_MODE       = 11   # EEPROM 영역 — Torque Enable=0(OFF)일 때만 기록 가능
ADDR_TORQUE_ENABLE        = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_GOAL_CURRENT         = 102  # 2바이트! write2ByteTxRx로만 써야 함
ADDR_GOAL_POSITION        = 116
ADDR_PRESENT_CURRENT      = 126  # 2바이트, signed (read2ByteTxRx)
ADDR_PRESENT_POSITION     = 132  # 4바이트

# Operating Mode(11) 값
OPERATING_MODE_CURRENT_BASED_POSITION = 5  # position + current(torque) 동시 제어


def rad_to_tick(angle_rad: float) -> int:
    """rad → XM430 tick 변환.  0 rad = 2048 tick."""
    return int((angle_rad / (2.0 * math.pi)) * 4096 + 2048)


def to_signed16(value: int) -> int:
    """read2ByteTxRx는 unsigned 16bit로 반환하므로, Present Current처럼
    부호 있는 값은 이 함수로 변환해서 써야 함 (안 그러면 음의 전류가
    큰 양수로 읽혀서 grasp 판정이 완전히 틀어짐)."""
    return value - 0x10000 if value > 0x7FFF else value


def ratio_to_tick(ratio: float, tick_open: int, tick_close: int) -> int:
    """그리퍼 개폐 비율(0=열림,1=닫힘) → tick 선형 매핑.

    tick_open이 tick_close보다 크든 작든(모터 장착 방향에 따라 다름)
    상관없이 그대로 선형보간 — 순서는 호출하는 쪽에서 클램핑으로 처리.
    """
    ratio = max(0.0, min(1.0, ratio))
    return int(round(tick_open + ratio * (tick_close - tick_open)))


class NeckDynamixelNode:

    def __init__(self):
        rospy.init_node("neck_dynamixel_node", anonymous=False)

        # ── 이전 명령값 (dead zone 비교용) ───────────────────
        self._prev_yaw_rad   = None
        self._prev_pitch_rad = None
        self._prev_grip_ratio = None

        # ── grasp 판정용 상태 ─────────────────────────────────
        self._gripper_goal_tick = None   # 마지막으로 보낸 goal position tick
        self._grasp_state  = False       # 현재 publish 중인(디바운스 반영된) 상태
        self._match_count  = 0           # grasp 조건 연속 만족 횟수
        self._miss_count   = 0           # grasp 조건 연속 불만족 횟수

        # config.py에 그리퍼 항목이 아직 없어도 죽지 않도록 opt-in 플래그로 처리
        self._gripper_enabled = getattr(config, 'USE_GRIPPER', False)

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
        if self._gripper_enabled:
            # Operating Mode(11)는 EEPROM 영역이라 Torque Enable이 0일 때만 쓸 수 있음.
            # (이전 실행이 비정상 종료돼 토크가 이미 ON인 채로 남아있을 수 있으므로 방어적으로 한 번 꺼줌)
            self._packet.write1ByteTxRx(
                self._port, config.MOTOR_ID_GRIPPER, ADDR_TORQUE_ENABLE, 0
            )
            self._set_operating_mode(config.MOTOR_ID_GRIPPER,
                                     OPERATING_MODE_CURRENT_BASED_POSITION)
            self._torque_on(config.MOTOR_ID_GRIPPER)
            # 순수 Position Control 대신 Current-based Position Control을 쓰면
            # goal position(닫힘 각도)은 그대로 두되, 이 전류값 이상 힘을 못 쓰게 상한이 걸림
            # → 물체 크기와 무관하게 "완전히 닫혀라"라고만 명령해도 안전하게 파지 가능.
            self._set_goal_current(config.MOTOR_ID_GRIPPER,
                                   config.GRIPPER_GOAL_CURRENT)

        # 홈 이동 — 느린 프로파일로 부드럽게
        self._set_profile(config.MOTOR_ID_YAW,
                          config.HOME_PROFILE_VEL_YAW,
                          config.HOME_PROFILE_ACC_YAW)
        self._set_profile(config.MOTOR_ID_PITCH,
                          config.HOME_PROFILE_VEL_PITCH,
                          config.HOME_PROFILE_ACC_PITCH)
        if self._gripper_enabled:
            self._set_profile(config.MOTOR_ID_GRIPPER,
                              config.HOME_PROFILE_VEL_GRIPPER,
                              config.HOME_PROFILE_ACC_GRIPPER)

        self._send_rad(config.MOTOR_ID_YAW,   0.0)
        self._send_rad(config.MOTOR_ID_PITCH, 0.0)
        if self._gripper_enabled:
            self._send_gripper(0.0)   # 시작은 항상 '열림'으로 (안전)
        rospy.loginfo("[Neck] 홈(0 rad)으로 이동 중... 2초 대기")
        rospy.sleep(2.0)

        # 텔레옵 추종용 프로파일로 전환
        self._set_profile(config.MOTOR_ID_YAW,
                          config.TELEOP_PROFILE_VEL_YAW,
                          config.TELEOP_PROFILE_ACC_YAW)
        self._set_profile(config.MOTOR_ID_PITCH,
                          config.TELEOP_PROFILE_VEL_PITCH,
                          config.TELEOP_PROFILE_ACC_PITCH)
        if self._gripper_enabled:
            self._set_profile(config.MOTOR_ID_GRIPPER,
                              config.TELEOP_PROFILE_VEL_GRIPPER,
                              config.TELEOP_PROFILE_ACC_GRIPPER)
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
            f"[Neck] Dead zone — "
            f"YAW: {math.degrees(config.DEAD_ZONE_YAW):.2f}°  "
            f"PITCH: {math.degrees(config.DEAD_ZONE_PITCH):.2f}°"
        )

        if self._gripper_enabled:
            rospy.Subscriber(
                config.GRIPPER_TOPIC,
                Float64MultiArray,
                self._gripper_cb,
                queue_size=1,
                buff_size=2**16,
            )
            rospy.loginfo(f"[Gripper] {config.GRIPPER_TOPIC} 수신 대기 중... "
                          f"(dead zone: {config.DEAD_ZONE_GRIPPER:.3f})")

            # ── grasp 상태 publish + 주기적 판정 ──────────────
            self._pub_gripper_status = rospy.Publisher(
                config.GRIPPER_STATUS_TOPIC, Bool, queue_size=1
            )
            self._grasp_timer = rospy.Timer(
                rospy.Duration(1.0 / config.GRIPPER_STATUS_HZ),
                self._check_grasp,
            )
            rospy.loginfo(
                f"[Gripper] grasp 판정 시작 → {config.GRIPPER_STATUS_TOPIC} "
                f"({config.GRIPPER_STATUS_HZ}Hz)"
            )
        else:
            rospy.loginfo("[Gripper] USE_GRIPPER=False → 그리퍼 비활성화")

        rospy.on_shutdown(self._shutdown)

    # ── 콜백 ──────────────────────────────────────────────────
    def _neck_cb(self, msg: Float64MultiArray):
        """[yaw_rad, pitch_rad] 수신 → dead zone 체크 → 모터 전송."""
        if len(msg.data) < 2:
            rospy.logwarn_once("[Neck] 메시지 크기 < 2, 무시")
            return

        yaw_rad   = config.YAW_DIRECTION   * float(msg.data[0])
        pitch_rad = config.PITCH_DIRECTION * float(msg.data[1])

        # dead zone: 이전 값과 차이가 작으면 전송 생략
        if self._prev_yaw_rad is None or \
                abs(yaw_rad - self._prev_yaw_rad) >= config.DEAD_ZONE_YAW:
            self._send_rad(config.MOTOR_ID_YAW, yaw_rad)
            self._prev_yaw_rad = yaw_rad

        if self._prev_pitch_rad is None or \
                abs(pitch_rad - self._prev_pitch_rad) >= config.DEAD_ZONE_PITCH:
            self._send_rad(config.MOTOR_ID_PITCH, pitch_rad)
            self._prev_pitch_rad = pitch_rad

    def _gripper_cb(self, msg: Float64MultiArray):
        """[grip_ratio] 수신 (0=열림, 1=닫힘) → dead zone 체크 → 모터 전송."""
        if len(msg.data) < 1:
            rospy.logwarn_once("[Gripper] 메시지 크기 < 1, 무시")
            return

        ratio = float(msg.data[0])

        if self._prev_grip_ratio is None or \
                abs(ratio - self._prev_grip_ratio) >= config.DEAD_ZONE_GRIPPER:
            self._send_gripper(ratio)
            self._prev_grip_ratio = ratio

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

    def _send_gripper(self, ratio: float):
        """grip ratio(0~1) → tick 변환 + 클램핑 + 모터 전송."""
        tick = ratio_to_tick(ratio, config.GRIPPER_TICK_OPEN, config.GRIPPER_TICK_CLOSE)
        tick_lo = min(config.GRIPPER_TICK_OPEN, config.GRIPPER_TICK_CLOSE)
        tick_hi = max(config.GRIPPER_TICK_OPEN, config.GRIPPER_TICK_CLOSE)
        tick    = max(tick_lo, min(tick, tick_hi))
        self._gripper_goal_tick = tick   # grasp 판정용 (position gap 계산)

        result, err = self._packet.write4ByteTxRx(
            self._port, config.MOTOR_ID_GRIPPER, ADDR_GOAL_POSITION, tick
        )
        if result != COMM_SUCCESS:
            rospy.logwarn_throttle(
                1.0,
                f"[Gripper] write 실패: {self._packet.getTxRxResult(result)}"
            )
        elif err != 0:
            rospy.logwarn_throttle(
                1.0,
                f"[Gripper] 에러: {self._packet.getRxPacketError(err)}"
            )

    def _check_grasp(self, _event):
        """주기적으로 Present Current/Position을 읽어 grasp 여부 판정 후 publish.

        판정 조건 (AND):
          1) |Present Current| >= GRASP_CURRENT_RATIO_THRESH * GRIPPER_GOAL_CURRENT
             → 모터가 상한 근처까지 힘을 쓰고 있음
          2) |goal tick - present tick| >= GRASP_POSITION_GAP_TICK
             → 그런데도 목표 위치까지 못 감 (뭔가에 막혀서 저항 중)
        디바운스: 연속 GRASP_DEBOUNCE_COUNT번 만족/불만족해야 상태를 뒤집어서
        순간적인 전류 스파이크로 인한 채터링을 막음.
        """
        if self._gripper_goal_tick is None:
            return

        cur_raw, result_c, _ = self._packet.read2ByteTxRx(
            self._port, config.MOTOR_ID_GRIPPER, ADDR_PRESENT_CURRENT
        )
        pos_raw, result_p, _ = self._packet.read4ByteTxRx(
            self._port, config.MOTOR_ID_GRIPPER, ADDR_PRESENT_POSITION
        )
        if result_c != COMM_SUCCESS or result_p != COMM_SUCCESS:
            # 통신 에러난 주기는 그냥 건너뜀 (상태 유지, 굳이 로그 스팸 안 냄)
            return

        present_current = to_signed16(cur_raw)
        current_ok = abs(present_current) >= \
            config.GRASP_CURRENT_RATIO_THRESH * config.GRIPPER_GOAL_CURRENT
        position_ok = abs(self._gripper_goal_tick - pos_raw) >= \
            config.GRASP_POSITION_GAP_TICK

        condition_met = current_ok and position_ok

        if condition_met:
            self._match_count += 1
            self._miss_count = 0
        else:
            self._miss_count += 1
            self._match_count = 0

        if not self._grasp_state and self._match_count >= config.GRASP_DEBOUNCE_COUNT:
            self._grasp_state = True
        elif self._grasp_state and self._miss_count >= config.GRASP_DEBOUNCE_COUNT:
            self._grasp_state = False

        self._pub_gripper_status.publish(Bool(data=self._grasp_state))

    def _torque_on(self, motor_id: int):
        result, _ = self._packet.write1ByteTxRx(
            self._port, motor_id, ADDR_TORQUE_ENABLE, 1
        )
        if result != COMM_SUCCESS:
            rospy.logwarn(f"[Neck] 토크 ON 실패 (ID={motor_id})")

    def _set_operating_mode(self, motor_id: int, mode: int):
        """Operating Mode(11) 설정. EEPROM 영역이므로 Torque Enable=0일 때만 반영됨
        (Torque가 이미 ON이면 조용히 무시되므로 호출 전에 반드시 토크가 꺼져 있어야 함)."""
        result, err = self._packet.write1ByteTxRx(
            self._port, motor_id, ADDR_OPERATING_MODE, mode
        )
        if result != COMM_SUCCESS:
            rospy.logwarn(
                f"[Gripper] Operating Mode 설정 실패 (ID={motor_id}): "
                f"{self._packet.getTxRxResult(result)}"
            )

    def _set_goal_current(self, motor_id: int, current: int):
        """Goal Current(102) 설정 — 반드시 write2ByteTxRx 사용.
        (write4ByteTxRx를 쓰면 4바이트가 기록되면서 바로 뒤 Goal Velocity(104)의
        앞 2바이트까지 덮어써서 예상치 못한 동작이 생길 수 있음)"""
        result, err = self._packet.write2ByteTxRx(
            self._port, motor_id, ADDR_GOAL_CURRENT, current
        )
        if result != COMM_SUCCESS:
            rospy.logwarn(
                f"[Gripper] Goal Current 설정 실패 (ID={motor_id}): "
                f"{self._packet.getTxRxResult(result)}"
            )

    def _set_profile(self, motor_id: int, velocity: int, acceleration: int):
        """ProfileAcceleration → ProfileVelocity 순서로 설정."""
        self._packet.write4ByteTxRx(
            self._port, motor_id, ADDR_PROFILE_ACCELERATION, acceleration
        )
        self._packet.write4ByteTxRx(
            self._port, motor_id, ADDR_PROFILE_VELOCITY, velocity
        )

    def _shutdown(self):
        if self._gripper_enabled and hasattr(self, '_grasp_timer'):
            self._grasp_timer.shutdown()   # 종료 절차 중 타이머 콜백이 끼어들지 않게 먼저 정지

        rospy.loginfo("[Neck] 종료 → 홈(0 rad) 복귀 후 토크 OFF")
        self._set_profile(config.MOTOR_ID_YAW,
                          config.HOME_PROFILE_VEL_YAW,
                          config.HOME_PROFILE_ACC_YAW)
        self._set_profile(config.MOTOR_ID_PITCH,
                          config.HOME_PROFILE_VEL_PITCH,
                          config.HOME_PROFILE_ACC_PITCH)
        self._send_rad(config.MOTOR_ID_YAW,   0.0)
        self._send_rad(config.MOTOR_ID_PITCH, 0.0)

        if self._gripper_enabled:
            rospy.loginfo("[Gripper] 종료 → 열림 복귀 후 토크 OFF")
            self._set_profile(config.MOTOR_ID_GRIPPER,
                              config.HOME_PROFILE_VEL_GRIPPER,
                              config.HOME_PROFILE_ACC_GRIPPER)
            self._send_gripper(0.0)   # 종료 시 항상 '열림'으로 복귀 (안전)

        import time; time.sleep(2.0)

        self._packet.write1ByteTxRx(
            self._port, config.MOTOR_ID_YAW, ADDR_TORQUE_ENABLE, 0
        )
        self._packet.write1ByteTxRx(
            self._port, config.MOTOR_ID_PITCH, ADDR_TORQUE_ENABLE, 0
        )
        if self._gripper_enabled:
            self._packet.write1ByteTxRx(
                self._port, config.MOTOR_ID_GRIPPER, ADDR_TORQUE_ENABLE, 0
            )
        self._port.closePort()
        rospy.loginfo("[Neck] 종료 완료")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    NeckDynamixelNode().spin()
