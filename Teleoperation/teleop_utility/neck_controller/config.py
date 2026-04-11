# ============================================================
# config.py  — teleop_neck_controller 파라미터 설정
# ============================================================
# 이 파일의 값만 수정하면 됩니다.
# neck_dynamixel_node.py는 건드리지 않아도 됩니다.

# ── 하드웨어 ─────────────────────────────────────────────────
DEVICE_PORT      = "/dev/ttyUSB0"
BAUD_RATE        = 4_000_000
PROTOCOL_VERSION = 2.0

MOTOR_ID_YAW   = 1
MOTOR_ID_PITCH = 2

# ── 방향 보정 ─────────────────────────────────────────────────
# 모터 설치 방향에 따라 +1.0 또는 -1.0으로 변경
YAW_DIRECTION   = +1.0
PITCH_DIRECTION = +1.0

# ── 관절 한계 (URDF Wholebody_39_DoF 기준) ───────────────────
# rad 기준 (텔레옵과 동일한 값)
YAW_RAD_MIN   = -1.39626
YAW_RAD_MAX   = +1.39626
PITCH_RAD_MIN = -0.872665
PITCH_RAD_MAX = +0.610865238

# tick 기준 (위 rad값을 변환한 것, 변경 불필요)
# tick = (rad / 2π) × 4096 + 2048
YAW_TICK_MIN   = 1138
YAW_TICK_MAX   = 2958
PITCH_TICK_MIN = 1480
PITCH_TICK_MAX = 2617

# ── 프로파일 설정 ─────────────────────────────────────────────
# 홈 이동용 (느리고 부드럽게)
HOME_PROFILE_VEL_YAW   = 75
HOME_PROFILE_VEL_PITCH = 100
HOME_PROFILE_ACC_YAW   = 25
HOME_PROFILE_ACC_PITCH = 30

# 텔레옵 추종용 (값이 클수록 빠르고 드드득 가능성 있음, 조절하면서 사용)
TELEOP_PROFILE_VEL_YAW   = 175
TELEOP_PROFILE_VEL_PITCH = 175
TELEOP_PROFILE_ACC_YAW   = 50
TELEOP_PROFILE_ACC_PITCH = 50

# ── Dead Zone ─────────────────────────────────────────────────
# 이전 명령과 차이가 이 값(rad) 이하면 모터에 전송하지 않음
# 값이 클수록 미세 진동 억제, 너무 크면 작은 움직임이 무시됨
# 권장 범위: 0.005 ~ 0.02 rad (약 0.3 ~ 1.1도)
DEAD_ZONE_YAW   = 0.01   # rad
DEAD_ZONE_PITCH = 0.01   # rad

# ── ROS ──────────────────────────────────────────────────────
NECK_TOPIC = "/neck_controller/command"