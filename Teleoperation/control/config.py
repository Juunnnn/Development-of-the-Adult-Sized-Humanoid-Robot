"""
config.py
─────────────────────────────────────────────────────────────
모든 상수/경로/파라미터를 한 곳에서 관리합니다.
숫자를 바꾸거나 경로를 수정할 때는 이 파일만 건드리면 됩니다.
로직(계산 코드)은 없고, 순수하게 값 선언만 있습니다.
"""

import os

# ══════════════════════════════════════════════════════════════
# 파일 경로
# ══════════════════════════════════════════════════════════════

# 이 파일(config.py)이 있는 폴더. 상대경로 기준점으로 사용됨.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TELEOPERATION_DIR = os.path.dirname(CURRENT_DIR)                        # Teleoperation/
TELEVISION_DIR    = os.path.join(TELEOPERATION_DIR, 'TeleVision')       # Teleoperation/TeleVision/

# 로봇 URDF 경로. 피노키오가 이 파일을 읽어서 관절 구조/한계를 파악함.
URDF_PATH = '/home/teleopstation/catkin_ws/src/Wholebody_39_DoF_URDF/urdf/Wholebody_39_DoF_URDF.urdf'

# 팔 캘리브레이션 저장/로드 경로.
# Quest의 손 기준 위치(quest_L/R_init)와 손목/목 초기 rotation 행렬을 JSON으로 저장함.
CALIB_PATH = os.path.join(CURRENT_DIR, 'calib.json')

# 손가락 캘리브레이션 저장/로드 경로.
# 손 펼쳤을 때의 각 손가락 MCP~Tip 거리(L_calib)를 JSON으로 저장함.
FINGER_CALIB_PATH = os.path.join(CURRENT_DIR, 'finger_calib.json')

# TeleVision(Quest 스트리밍 서버) SSL 인증서 경로.
# Quest와 HTTPS로 통신하기 위해 필요함.
CERT_FILE = os.path.join(TELEVISION_DIR, 'cert.pem')
KEY_FILE  = os.path.join(TELEVISION_DIR, 'key.pem')

# 팔/목 트래킹 사용 여부. False면 IK 계산 없이 현재 자세 유지.
USE_ARM    = True
# 손가락 트래킹 사용 여부. False면 캘리브/텔레옵 모두 스킵하고 FINGER_NEUTRAL 유지.
USE_FINGER = True
FINGER_DEBUG = False
# Quest 3D 구체 오버레이 표시 여부. False면 손바닥 위치/방향 구체를 표시하지 않음.
USE_SPHERE = True


# ══════════════════════════════════════════════════════════════
# 사운드 파일 경로
# 상태 전환 시 beep() 함수가 이 경로의 파일을 재생함.
# ══════════════════════════════════════════════════════════════
# 사운드 파일 경로 설정.
# 원하는 .oga 또는 .wav 파일 경로로 교체하면 됨.
# 사용 가능한 기본 사운드 목록:
#   ls /usr/share/sounds/freedesktop/stereo/
SOUND = {
    'teleop_start':       '/usr/share/sounds/freedesktop/stereo/service-login.oga',    # 텔레옵 첫 시작
    'warn':               '/usr/share/sounds/freedesktop/stereo/window-attention.oga', # 트래킹 소실/점프 경고
    'sync_done':          '/usr/share/sounds/freedesktop/stereo/power-plug.oga',       # 소실 복귀 후 싱크 완료
    'calib_start':        '/usr/share/sounds/freedesktop/stereo/service-login.oga',    # 캘리브 시작
    'calib_done':         '/usr/share/sounds/freedesktop/stereo/service-logout.oga',   # 팔 캘리브 완료
    'finger_calib_done':  '/usr/share/sounds/freedesktop/stereo/complete.oga',         # 손가락 캘리브 완료 ← 여기서 변경
}


# ══════════════════════════════════════════════════════════════
# 로봇 관절 순서 및 초기/종료 자세
# JOINT_ORDER의 인덱스 순서가 INIT_VALS, FINAL_VALS와 1:1 대응됨.
# ══════════════════════════════════════════════════════════════

# 제어 대상 관절 이름 목록 (12개).
# joint_ids 리스트 인덱스: L어깨=0~2, L팔꿈치=3, L손목=4 / R어깨=5~7, R팔꿈치=8, R손목=9 / 목=10~11
JOINT_ORDER = [
    'L_shoulder_pitch_joint', 'L_shoulder_roll_joint', 'L_shoulder_yaw_joint',
    'L_elbow_pitch_joint',    'L_wrist_yaw_joint',        # index 0~4
    'R_shoulder_pitch_joint', 'R_shoulder_roll_joint', 'R_shoulder_yaw_joint',
    'R_elbow_pitch_joint',    'R_wrist_yaw_joint',        # index 5~9
    'Neck_Yaw_Joint',         'Neck_Pitch_Joint',         # index 10~11
]

# 초기/종료 자세: 프로그램 시작 및 Ctrl+C 종료 시 이 자세로 복귀함.
# 팔꿈치 접힌 편안한 자세.
INIT_POS  = [-1.0472, -0.0872, 0, -1.57, 0,  -1.0472, 0.0872, 0, -1.57, 0,  0, 0]

# 팔 캘리브레이션 자세: 양팔 앞으로 나란히. 캘리브레이션 시 이 자세로 이동한다.
CALIB_POS = [-1.57, 0, 0, 0, 0,  -1.57, 0, 0,  0, 0, 0, 0]

# 손바닥 가상 프레임 오프셋.
# wrist_yaw 프레임에서 z축으로 -0.11815m 떨어진 곳이 실제 손바닥 중심.
# IK 목표점을 손목이 아닌 손바닥 기준으로 맞추기 위해 사용.
PALM_Z_OFFSET = -0.11815  # [m]


# ══════════════════════════════════════════════════════════════
# 제어 파라미터
# ══════════════════════════════════════════════════════════════

# 메인 루프 실행 주파수. Quest가 약 45Hz로 데이터를 보내므로 50Hz로 맞춤.
CONTROL_HZ = 50

# 팔 캘리브 샘플 수. 50샘플 ≈ 1초치 데이터를 평균내서 기준점으로 사용.
CALIB_COUNT = 50

# 손가락 캘리브 샘플 수. (USE_FINGER=False 이면 사용 안 함)
FINGER_CALIB_COUNT = 50

# 손목 회전(wrist_yaw)에 곱하는 스케일. 1.0 = 사람 손목 회전량 그대로 반영.
WRIST_SCALE = 1.0

# 목 회전(neck yaw/pitch)에 곱하는 스케일. 1.0 = 사람 머리 회전량 그대로 반영.
NECK_SCALE  = 1.0

# /joint_states 수신 대기 최대 시간. 이 시간 내에 못 받으면 q_init으로 대체.
JOINT_STATE_TIMEOUT = 5.0  # [s]

# Quest 접속 후 텔레옵 시작까지 대기 시간 [s]
TELEOP_START_DELAY = 5.0


# ══════════════════════════════════════════════════════════════
# SYNCING 파라미터
# SYNCING 상태: 텔레옵 시작 전 로봇을 사람 손 위치에 맞게 이동시키는 단계.
# ══════════════════════════════════════════════════════════════

# 싱크 완료 조건 ①: 실제 손바닥 위치 오차가 이 값 이하면 완료로 판정.
SYNC_POSITION_THRESH = 0.05   # [m] = 5cm

# 싱크 완료 조건 ②: 실제 관절각 오차가 이 값 이하면 완료로 판정.
SYNC_JOINT_THRESH    = 0.1    # [rad]

# 보간 이동 목표 시간. 이 시간 안에 sync_start_q → sync_target_q로 ease-in-out 보간함.
SYNC_DURATION = 3.0  # [s]

# 싱크 강제 종료 시간. 이 시간이 지나도 완료 판정이 안 나면 텔레옵을 강제 시작함.
SYNC_TIMEOUT  = 10.0  # [s]


# ══════════════════════════════════════════════════════════════
# FREEZE 파라미터
# FREEZE 상태: 트래킹 소실/점프 감지 시 진입. N초 동안 현재 자세를 유지한 뒤 SYNCING으로 전환.
# ══════════════════════════════════════════════════════════════

# FREEZE 후 SYNCING 전환까지 대기 시간. 이 시간 동안 사용자가 자세를 잡을 수 있음.
FREEZE_DURATION = 2.0  # [s]


# ══════════════════════════════════════════════════════════════
# 점프 감지
# Quest 트래킹 오류 등으로 손 위치가 한 프레임 만에 크게 튀는 경우를 걸러냄.
# ══════════════════════════════════════════════════════════════

# 연속 두 프레임 사이 손 이동량이 이 값을 초과하면 점프로 판정 → FREEZE 진입.
JUMP_THRESHOLD = 0.15  # [m] = 15cm


# ══════════════════════════════════════════════════════════════
# EMA(지수이동평균) 필터 alpha 값
# alpha가 1에 가까울수록 빠른 반응(노이즈 많음), 0에 가까울수록 부드러움(지연 발생).
# ══════════════════════════════════════════════════════════════

EMA_ARM       = 0.6  # 팔 IK 출력 스무딩
EMA_WRIST     = 0.6  # 손목 yaw 스무딩
EMA_QUEST_POS = 0.7  # Quest 손 위치 입력 스무딩 (IK 입력 전 전처리)
EMA_NECK      = 0.3  # 목 회전 스무딩 (목은 더 부드럽게 → alpha 낮게)
EMA_FINGER    = 0.4  # 손가락 관절각 스무딩


# ══════════════════════════════════════════════════════════════
# ROS 토픽명
# ══════════════════════════════════════════════════════════════

# 팔/목 관절 명령 토픽. Float64MultiArray 12개 값을 publish함.
TOPIC_ARM          = '/arm_controller/command'

# 손가락 관절 명령 토픽. Float64MultiArray 16개 값을 publish함.
TOPIC_HAND         = '/finger_controller/command'

# D435i 카메라 영상 토픽. 이 영상을 Quest로 스트리밍함.
TOPIC_CAMERA       = '/camera/color/image_raw'

# 로봇 실제 관절각 수신 토픽. 현재 자세 파악 및 싱크 완료 판정에 사용.
TOPIC_JOINT_STATES = '/joint_states'