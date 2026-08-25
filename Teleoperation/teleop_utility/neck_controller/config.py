# ============================================================
# config.py  — teleop_neck_controller 파라미터 설정
# ============================================================
# 이 파일의 값만 수정하면 됩니다.
# neck_dynamixel_node.py는 건드리지 않아도 됩니다.

# ── 그리퍼 ───────────────────────────────────────────────────
# False면 그리퍼 관련 코드(토크 on/프로파일/subscriber)를 전부 건너뜀 →
# 그리퍼 안 물려 있어도(하드웨어 없어도) 목 전용이던 예전과 완전히 동일하게 동작.
USE_GRIPPER = True

# ── 하드웨어 ─────────────────────────────────────────────────
DEVICE_PORT      = "/dev/ttyUSB0"
BAUD_RATE        = 4_000_000
PROTOCOL_VERSION = 2.0

MOTOR_ID_YAW   = 1
MOTOR_ID_PITCH = 2
MOTOR_ID_GRIPPER = 3   # YAW/PITCH와 안 겹침. 버스에 물리기 전 위자드로 이 ID로 미리 바꿔둘 것
                        # (그리퍼 모터가 아직 공장 기본값(보통 1번)이면 YAW랑 충돌함)

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

# 그리퍼는 rad 대신 위자드 각도로 실측 (닫힘 0°, 열림 135~140° 확인됨)
# tick = (deg / 360) × 4096, 양쪽 다 하드스탑에서 살짝 여유 둠
# OPEN이 CLOSE보다 커도/작아도 상관없이 코드가 알아서 그 사이로 매핑함
GRIPPER_TICK_CLOSE = 15     # 완전 닫힘(0°)보다 살짝 열린 쪽으로
GRIPPER_TICK_OPEN  = 1536   # 135° 기준 (실측 135~140°의 보수적인 쪽) — 하드스탑까지 여유 있음

# ── 프로파일 설정 ─────────────────────────────────────────────
# 홈 이동용 (느리고 부드럽게)
HOME_PROFILE_VEL_YAW   = 75
HOME_PROFILE_VEL_PITCH = 100
HOME_PROFILE_ACC_YAW   = 25
HOME_PROFILE_ACC_PITCH = 30
HOME_PROFILE_VEL_GRIPPER = 60   # 목 값들과 비슷한 범위로 시작 (실측치 아님, 조절 필요)
HOME_PROFILE_ACC_GRIPPER = 20

# 텔레옵 추종용 (값이 클수록 빠르고 드드득 가능성 있음, 조절하면서 사용)
TELEOP_PROFILE_VEL_YAW   = 175
TELEOP_PROFILE_VEL_PITCH = 175
TELEOP_PROFILE_ACC_YAW   = 50
TELEOP_PROFILE_ACC_PITCH = 50
TELEOP_PROFILE_VEL_GRIPPER = 150   # 위와 동일 — 시작점일 뿐, 실제로 쥐어보고 조절할 것
TELEOP_PROFILE_ACC_GRIPPER = 50

# ── Dead Zone ─────────────────────────────────────────────────
# 이전 명령과 차이가 이 값(rad) 이하면 모터에 전송하지 않음
# 값이 클수록 미세 진동 억제, 너무 크면 작은 움직임이 무시됨
# 권장 범위: 0.005 ~ 0.02 rad (약 0.3 ~ 1.1도)
DEAD_ZONE_YAW   = 0.01   # rad
DEAD_ZONE_PITCH = 0.01   # rad
DEAD_ZONE_GRIPPER = 0.02   # ratio 기준 0~1 (rad 아님!) — grip_ratio 변화가 이보다 작으면 전송 생략

# ── ROS ──────────────────────────────────────────────────────
NECK_TOPIC           = "/neck_controller/command"
GRIPPER_TOPIC        = "/gripper_controller/command"
GRIPPER_STATUS_TOPIC = "/gripper_controller/status"   # grasp 성공 여부(Bool) publish. 텔레옵쪽 config.py의
                                                        # TOPIC_GRIPPER_STATUS와 문자열이 반드시 같아야 함.

# ── 그리퍼 파지력 (Current-based Position Control) ─────────────
# Goal Current(raw unit, 1 unit ≈ 2.69mA). 작게 시작해서 실제로 원하는 물체를
# 쥘 수 있을 때까지 조금씩 올리면서 튜닝할 것. (예: 150 ≈ 400mA)
GRIPPER_GOAL_CURRENT = 150

# ── grasp 판정 (grasp_status publish 용) ────────────────────────
# 아래 두 조건을 동시에 만족하면 "물체를 쥐고 저항 중"으로 판단:
#   1) |Present Current| >= GRASP_CURRENT_RATIO_THRESH * GRIPPER_GOAL_CURRENT
#   2) |Goal tick - Present tick| >= GRASP_POSITION_GAP_TICK
# (전류만 보면 가속 구간에서도 순간적으로 튈 수 있어서, 목표 위치에 못 미친 채
#  버티고 있다는 조건을 같이 봐서 오탐을 줄임)
GRASP_CURRENT_RATIO_THRESH = 0.85
GRASP_POSITION_GAP_TICK    = 40     # 4096tick/360° 기준 약 3.5°
# 판정 채터링 방지 — 연속 이 횟수만큼 조건을 만족/불만족해야 상태를 뒤집음
GRASP_DEBOUNCE_COUNT = 3
# grasp 판정 체크 주기 [Hz]
GRIPPER_STATUS_HZ = 20