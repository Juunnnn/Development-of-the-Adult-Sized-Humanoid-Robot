"""
config.py
─────────────────────────────────────────────────────────────
모든 상수/경로/파라미터를 한 곳에서 관리합니다.
숫자를 바꾸거나 경로를 수정할 때는 이 파일만 건드리면 됩니다.
로직(계산 코드)은 없고, 순수하게 값 선언만 있습니다.
"""

import os
import numpy as np

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

# 오른손 그리퍼(Dynamixel XM430) 트래킹. False면 그립 계산/publish 자체를 안 함.
# 왼손 AmazingHand(USE_FINGER_FE/AA) 경로와는 완전히 독립적 — 서로 영향 없음.
USE_GRIPPER = False
# 그립 비율 계산에 쓸 손가락 번호 (1=엄지,2=검지,3=중지,4=약지←소지).
# 엄지는 굽힘 범위가 달라 기본에서 제외. 주먹 쥘 때 유독 안 접히는 손가락이
# 있으면 빼거나, 반대로 판정이 애매하면 (1,2,3,4)로 넓혀서 실험 가능.
GRIP_FINGERS = (2, 3, 4)

# 허리(torso) yaw 보상. shoulder_roll이 한계에 닿으면 torso를 돌려 보완.
# URDF의 Waist_joint가 revolute로 변경된 경우에만 실제 동작함.
USE_TORSO = False

# 목 회전 트래킹. False면 목 관절 0으로 고정.
USE_NECK = True
# 손 중점을 바라보는 목 트래킹 모드 (USE_NECK=False일 때 유효).
# Quest 양손 위치 중점을 향해 neck yaw/pitch를 자동 계산.
USE_NECK_TRACK = False

# 손가락 굽힘각 실시간 출력 (튜닝용). True로 켜면 터미널에 각도 출력.
FINGER_DEBUG = False
GRIPPER_DEBUG = True
# Quest 3D 구체 오버레이 표시 여부.
USE_SPHERE = True

# ── 인트로 영상 설정 ──────────────────────────────────────────
# Quest 연결 직후 이 영상을 재생하고, 끝나면 카운트다운을 시작함.
# False로 끄면 Quest 연결 즉시 카운트다운 시작.
USE_INTRO_VIDEO = False
VIDEO_PATH      = '/home/teleopstation/Downloads/exia_startup_2.mp4'


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
# INIT_POS  = [-1.05, 0.35, 0, -1.57, 0, -1.05, -0.35, 0, -1.57, 0,  0, 0]
INIT_POS  = [0.0, 0.15, 0.0, -1.57, 0.0, 0.0, -0.15, 0.0, -1.57, 0.0, 0.0, 0.0]

# 캘리브레이션 자세: 양팔 앞으로 나란히
CALIB_POS = [-1.57, 0.2618, 0, 0, 0,  -1.57, -0.2618, 0, 0, 0,  0, 0]

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

SYNC_POSITION_THRESH = 0.025   # 싱크 완료: 손바닥 위치 오차 [m]
SYNC_JOINT_THRESH    = 0.1    # 싱크 완료: 관절각 오차 [rad]
SYNC_DURATION        = 5.0    # 보간 이동 목표 시간 [s]
SYNC_TIMEOUT         = 10.0   # 싱크 강제 종료 시간 [s]

# 보조 싱킹(SYNCING → TELEOP 전환 시 잔여 오차 흡수) 시간 [s]
# SYNCING은 SYNC_JOINT_THRESH/SYNC_POSITION_THRESH 안에만 들어오면 끝나버려서
# 약간의 오차(최대 회전 0.1rad, 위치 0.05m)가 남은 채로 TELEOP에 진입할 수 있음.
# 이 잔여 오차를 IK가 1프레임(20ms)에 메우면 덜컹거리므로,
# TELEOP 진입 직후 이 시간 동안 한 번 더 부드럽게 흡수한다.
SYNC_BLEND_DURATION  = 3


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
EMA_TORSO     = 0.1   # 허리 yaw (더 느리게, 급격한 보상 방지)
EMA_GRIPPER   = 0.4   # 그리퍼 개폐 비율 (일단 EMA_FINGER와 동일하게 시작)


# ══════════════════════════════════════════════════════════════
# TORSO 보상 파라미터
# ══════════════════════════════════════════════════════════════

# 프레임당 최대 torso 변화량 [rad]. 클수록 빠른 반응, 작을수록 안정적.
TORSO_MAX_DELTA_PER_FRAME = 0.05

# torso 보상 후 IK 재실행 횟수. 많을수록 정확하지만 연산 증가.
TORSO_IK_ITER = 3

# 복귀 스프링 gain: return_step = max(MIN, |torso| * GAIN)
# torso가 많이 돌아갔을수록 빠르게 복귀, 0 근처에서 자연스럽게 감속.
TORSO_RETURN_GAIN = 0.15  # [rad/frame per rad]
TORSO_RETURN_MIN  = 0.003 # 최소 복귀 속도 [rad/frame]

# 가상 FK 테스트에서 roll을 여는 크기 [rad]
TORSO_ROLL_TEST_DELTA = 0.1


# ══════════════════════════════════════════════════════════════
# ORIENTATION 보조 과제 파라미터 (shoulder_yaw 활용도 개선)
# ══════════════════════════════════════════════════════════════
# position IK의 null space(1차원)에서 전완 방향을 추가로 맞추는 보조 과제.
# roll-yaw 겹침(yaw≈0 부근) 문제를 tie-break 해줘서, roll 리밋에 걸리기 전부터
# shoulder_yaw가 미리 반응하게 만듦. USE_TORSO=False인 지금 상태에서도
# (torso 없이) 바로 효과가 있음 - 지금은 roll이 리밋에 막히면 그냥 위치 오차로 남던 상황.

# Quest 핸드트래킹 좌표계 기준 "전완이 몸 바깥을 향하는" 로컬 축.
# ⚠️ L/R을 별도로 둠 - Quest 핸드트래킹은 보통 왼손/오른손 로컬 축이 거울대칭이라,
# 같은 축을 양쪽에 쓰면 한쪽만 맞고 반대쪽은 틀리게 됨 (실제로 겪은 문제:
# R만 검증했고 L은 한 번도 검증 안 한 채로 같은 값을 썼더니 L 팔꿈치가 이상하게 움직임).
#
# R값: 차렷 자세에서 전완을 안/밖으로 돌리며 실측으로 확인 완료.
# L값: 거울대칭 추정치(부호 반전)일 뿐 아직 미검증 - R 검증 때와 동일한 절차로
# (차렷 자세에서 왼쪽 전완을 안/밖으로 돌려보며 [FOREARM] L 오차와 [SHOULDER_YAW] L이
# 예상 방향으로 반응하는지 확인) 반드시 실측 확인할 것. 반대로 움직이면 부호를
# 다시 뒤집거나 다른 축(예: np.array([1,0,0]))으로 교체.
QUEST_FOREARM_LOCAL_AXIS_R = np.array([0.0, 0.0, -1.0])
QUEST_FOREARM_LOCAL_AXIS_L = np.array([0.0, 0.0, -1.0])

# orientation 정렬 gain. NULL_WEIGHT보다 충분히 커야 orientation이 우선함.
# 처음엔 0.1 정도로 낮게 시작해서 축 부호부터 확인한 뒤 점진적으로 올릴 것 (권장 상한 0.5~0.8).
ORIENT_WEIGHT = 0.15

# null-space 자세 복원 gain. "yaw를 자연스러운 자세로 당기는 스프링" 역할.
#
# 0.15: 스프링이 너무 세서 목표까지 못 가고 잔여오차 약 20°에서 멈춤 (실측)
# 0.03: 스프링이 거의 없어져서 브레이크 없이 계속 밀어붙임 → 하드 리밋(90°)까지
#       밀려서 눌어붙는 문제 발생 (실측, /joint_states에서 하드 리밋 값 확인됨)
# 0.08: yaw가 30~37° 선에서 안정적으로 멈춤, 하드리밋 문제 없음 (실측 확인)
# → 조금 더 안쪽으로 풀어주기 위해 0.06으로 소폭 완화.
# (ORIENT_LIMIT_MARGIN 안전장치는 그대로라 하드리밋까지 갈 위험은 여전히 없음)
NULL_WEIGHT = 0.06

# 전완 방향 타겟용 EMA 필터 alpha. Quest 손 orientation은 위치보다 노이즈가
# 심한 경우가 많아 EMA_QUEST_POS(0.7)보다 낮게 잡음.
EMA_FOREARM_DIR = 0.3

# position이 이미 eps 이내로 수렴해도 orientation 보정을 위해 최소 이만큼은 더 반복.
# 50Hz 연속 추종 중엔 위치가 첫 iteration부터 수렴하는 경우가 많아서 이게 없으면
# orientation 보정이 실행될 기회를 거의 못 얻음 (실제로 겪었던 문제).
ORIENT_MIN_ITER = 5

# orientation 정렬 오차(rad, cross product 크기 기준)가 이 이하면 조기 종료.
# 대략 사인값 기준이라 작은 각도에서는 라디안≈각도 오차로 봐도 무방 (0.02rad≈1.1°).
ORIENT_EPS = 0.02

# orientation 보정이 한 iteration에 낼 수 있는 최대 관절 변화량 [rad].
# 정렬오차가 클 때(예: 90°) closed-form 스텝이 한 번에 확 튀는 걸 막는 안전장치.
# 0.05는 폭주는 잘 막았지만, 팔꿈치+전완을 동시에 빠르게 움직이는 복합 동작에서는
# 프레임당 최대 변화량(0.05*ORIENT_MIN_ITER≈14°)에 막혀 추종이 눈에 띄게 뒤처짐.
# 0.08로 완화 - 여전히 예전(제한 없음)보다는 훨씬 보수적인 수준.
ORIENT_MAX_DELTA = 0.08

# 관절이 자기 하드리밋에서 이 거리(rad) 이내로 들어오면 orientation 스텝을
# 부드럽게 감쇠시킴. 실측으로 확인된 문제: NULL_WEIGHT 스프링과 MAX_DELTA
# 클램프만으로는 shoulder_yaw가 결국 하드리밋(±90°)까지 밀려서 눌러붙는 걸
# 못 막았음 - 이게 진짜 마지막 안전장치. 0.15rad ≈ 8.6°.
ORIENT_LIMIT_MARGIN = 0.15


# ══════════════════════════════════════════════════════════════
# ROS 토픽명
# ══════════════════════════════════════════════════════════════

TOPIC_ARM             = '/arm_controller/command'
TOPIC_HAND            = '/finger_controller/command'
TOPIC_TORSO           = '/torso_controller/command'
TOPIC_CAMERA          = '/camera/color/image_raw'
TOPIC_JOINT_STATES    = '/joint_states'
TOPIC_GRIPPER = '/gripper_controller/command'  
TOPIC_GRIPPER_STATUS  = '/gripper_controller/status'  # 젯슨 config_jetson.py의
                                                        # GRIPPER_STATUS_TOPIC과 문자열 일치해야 함

# ══════════════════════════════════════════════════════════════
# 관절별 물리적 한계 [rad] — Wholebody_39_DoF_URDF.urdf 기준
# /joint_states 수신값 검증(get_current_q)에 사용
# ══════════════════════════════════════════════════════════════
JOINT_SANITY_RANGE = {
    'Waist_joint':            (-1.57,     1.57),
    'L_shoulder_pitch_joint': (-3.14,     1.04),
    'L_shoulder_roll_joint':  (-0.34,     2.79),
    'L_shoulder_yaw_joint':   (-1.57,     1.57),
    'L_elbow_pitch_joint':    (-2.18166,  0.0),
    'L_wrist_yaw_joint':      (-1.57,     1.57),
    'R_shoulder_pitch_joint': (-3.14,     1.04),
    'R_shoulder_roll_joint':  (-2.79,     0.34),
    'R_shoulder_yaw_joint':   (-1.57,     1.57),
    'R_elbow_pitch_joint':    (-2.18166,  0.0),
    'R_wrist_yaw_joint':      (-1.57,     1.57),
    'Neck_Yaw_Joint':         (-1.39626,  1.39626),
    'Neck_Pitch_Joint':       (-0.872665, 0.872665),
}
