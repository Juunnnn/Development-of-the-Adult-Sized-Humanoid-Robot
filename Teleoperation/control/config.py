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

CURRENT_DIR       = os.path.dirname(os.path.abspath(__file__))
TELEOPERATION_DIR = os.path.dirname(CURRENT_DIR)
TELEVISION_DIR    = os.path.join(TELEOPERATION_DIR, 'TeleVision')

# 로봇 URDF 경로
URDF_PATH = '/home/teleopstation/catkin_ws/src/Wholebody_39_DoF_URDF/urdf/Wholebody_39_DoF_URDF.urdf'

# 팔 캘리브레이션 저장/로드 경로
# Quest의 손 기준 위치(quest_L/R_init)와 손목/목 초기 rotation 행렬을 JSON으로 저장함
CALIB_PATH = os.path.join(CURRENT_DIR, 'calib.json')

# TeleVision SSL 인증서 경로 (Quest HTTPS 통신용)
CERT_FILE = os.path.join(TELEVISION_DIR, 'cert.pem')
KEY_FILE  = os.path.join(TELEVISION_DIR, 'key.pem')

# ── 서브시스템 활성화 플래그 ──────────────────────────────────
# 팔 트래킹. False면 IK 계산 없이 현재 자세 유지.
USE_ARM = True

# 손가락 굽힘/펼침(FE) 트래킹. False면 FE는 NEUTRAL 유지.
USE_FINGER_FE = True
# 손가락 옆벌림/모음(AA) 트래킹. False면 AA는 0 유지.
# USE_FINGER_FE가 False여도 USE_FINGER_AA만 True로 켤 수 있음.
USE_FINGER_AA = True

# 목 회전 트래킹. False면 목 관절 0으로 고정.
USE_NECK = False
# 손 중점을 바라보는 목 트래킹 모드 (USE_NECK=False일 때 유효).
# Quest 양손 위치 중점을 향해 neck yaw/pitch를 자동 계산.
USE_NECK_TRACK = True

# 손가락 굽힘각 실시간 출력 (튜닝용). True로 켜면 터미널에 각도 출력.
FINGER_DEBUG = False
# Quest 3D 구체 오버레이 표시 여부.
USE_SPHERE = True

# ── 인트로 영상 설정 ──────────────────────────────────────────
# Quest 연결 직후 이 영상을 재생하고, 끝나면 카운트다운을 시작함.
# False로 끄면 Quest 연결 즉시 카운트다운 시작.
USE_INTRO_VIDEO = True
VIDEO_PATH      = '/home/teleopstation/Downloads/exia_startup.mp4'


# ══════════════════════════════════════════════════════════════
# 사운드 파일 경로
# ══════════════════════════════════════════════════════════════

SOUND = {
    'teleop_start': '/usr/share/sounds/freedesktop/stereo/service-login.oga',
    'warn':         '/usr/share/sounds/freedesktop/stereo/window-attention.oga',
    'sync_done':    '/usr/share/sounds/freedesktop/stereo/power-plug.oga',
    'calib_start':  '/usr/share/sounds/freedesktop/stereo/service-login.oga',
    'calib_done':   '/usr/share/sounds/freedesktop/stereo/service-logout.oga',
}


# ══════════════════════════════════════════════════════════════
# 로봇 관절 순서 및 초기/종료 자세
# ══════════════════════════════════════════════════════════════

# 제어 대상 관절 이름 목록 (12개)
# 인덱스: L어깨=0~2, L팔꿈치=3, L손목=4 / R어깨=5~7, R팔꿈치=8, R손목=9 / 목=10~11
JOINT_ORDER = [
    'L_shoulder_pitch_joint', 'L_shoulder_roll_joint', 'L_shoulder_yaw_joint',
    'L_elbow_pitch_joint',    'L_wrist_yaw_joint',
    'R_shoulder_pitch_joint', 'R_shoulder_roll_joint', 'R_shoulder_yaw_joint',
    'R_elbow_pitch_joint',    'R_wrist_yaw_joint',
    'Neck_Yaw_Joint',         'Neck_Pitch_Joint',
]

# 종료 자세: Ctrl+C 시 이 자세로 복귀 (팔꿈치 접힌 편안한 자세)
INIT_POS  = [-1.0472, -0.0872, 0, -1.57, 0,  -1.0472, 0.0872, 0, -1.57, 0,  0, 0]

# 캘리브레이션 자세: 양팔 앞으로 나란히
CALIB_POS = [-1.57, 0, 0, 0, 0,  -1.57, 0, 0, 0, 0,  0, 0]

# 손바닥 가상 프레임 오프셋 (wrist_yaw → 손바닥 중심까지 z축 거리)
PALM_Z_OFFSET = -0.11815  # [m]


# ══════════════════════════════════════════════════════════════
# 제어 파라미터
# ══════════════════════════════════════════════════════════════

CONTROL_HZ = 50       # 메인 루프 주파수 [Hz]
CALIB_COUNT = 50      # 팔 캘리브 샘플 수 (~1초)

WRIST_SCALE = 1.0     # 손목 회전 스케일 (1.0 = 사람 손목 1:1 반영)
NECK_SCALE  = 1.0     # 목 회전 스케일

JOINT_STATE_TIMEOUT = 5.0    # /joint_states 수신 대기 최대 시간 [s]
TELEOP_START_DELAY  = 5.0    # Quest 접속 후 텔레옵 시작까지 대기 시간 [s]


# ══════════════════════════════════════════════════════════════
# SYNCING 파라미터
# ══════════════════════════════════════════════════════════════

SYNC_POSITION_THRESH = 0.05   # 싱크 완료: 손바닥 위치 오차 [m]
SYNC_JOINT_THRESH    = 0.1    # 싱크 완료: 관절각 오차 [rad]
SYNC_DURATION        = 3.0    # 보간 이동 목표 시간 [s]
SYNC_TIMEOUT         = 10.0   # 싱크 강제 종료 시간 [s]


# ══════════════════════════════════════════════════════════════
# FREEZE 파라미터
# ══════════════════════════════════════════════════════════════

FREEZE_DURATION = 2.0   # 트래킹 소실 후 현재 자세 유지 시간 [s]


# ══════════════════════════════════════════════════════════════
# 점프 감지
# ══════════════════════════════════════════════════════════════

JUMP_THRESHOLD = 0.15   # 한 프레임 최대 손 이동량 [m] 초과 시 FREEZE


# ══════════════════════════════════════════════════════════════
# EMA 필터 alpha 값
# alpha → 1: 빠른 반응(노이즈 많음), alpha → 0: 부드러움(지연)
# ══════════════════════════════════════════════════════════════

EMA_ARM       = 0.6   # 팔 IK 출력
EMA_WRIST     = 0.6   # 손목 yaw
EMA_QUEST_POS = 0.7   # Quest 손 위치 입력
EMA_NECK      = 0.3   # 목 회전 (더 부드럽게)
EMA_FINGER    = 0.4   # 손가락 관절각


# ══════════════════════════════════════════════════════════════
# ROS 토픽명
# ══════════════════════════════════════════════════════════════

TOPIC_ARM          = '/arm_controller/command'
TOPIC_HAND         = '/finger_controller/command'
TOPIC_CAMERA       = '/camera/color/image_raw'
TOPIC_JOINT_STATES = '/joint_states'