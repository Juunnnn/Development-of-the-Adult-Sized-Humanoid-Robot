"""
motion_utils.py
─────────────────────────────────────────────────────────────
움직임 관련 유틸리티를 모읍니다.

- EMAFilter: IK 출력/입력값 스무딩 (떨림 제거)
- make_filters: 텔레옵에 필요한 필터 인스턴스 일괄 생성
- beep: 상태 전환 시 알림음 재생
- publish_smooth_move: 현재 자세 → 목표 자세 보간 이동
- publish_init / publish_fin: 초기/종료 자세로 이동하는 단축 함수
"""

import os
import time
import numpy as np
from std_msgs.msg import Float64MultiArray

import config


# ══════════════════════════════════════════════════════════════
# EMA(지수이동평균) 필터
# ══════════════════════════════════════════════════════════════

class EMAFilter:
    """
    지수이동평균(Exponential Moving Average) 필터.

    IK 출력이나 Quest 입력값의 프레임 간 떨림(노이즈)을 부드럽게 만들기 위해 사용함.
    수식: output = alpha * new_value + (1 - alpha) * prev_output

    alpha 값에 따른 특성:
      alpha = 1.0 → 필터 없음 (new_value 그대로 통과)
      alpha = 0.6 → 빠른 반응, 약간의 노이즈 감쇠 (팔/손목에 사용)
      alpha = 0.3 → 느린 반응, 강한 노이즈 감쇠 (목 제어에 사용)
      alpha → 0   → 거의 움직이지 않음

    적용 대상별 alpha 값 (config.py에서 조정):
      arm_filter         : 0.6  팔 IK 관절각
      wrist_filter_l/r   : 0.6  손목 yaw
      quest_pos_filter   : 0.7  Quest 손 위치 입력 (IK 넣기 전 전처리)
      neck_filter        : 0.3  목 회전 (목은 과하게 움직이면 이상해서 강하게 스무딩)
    """

    def __init__(self, alpha: float, size: int):
        self.alpha = alpha
        self.size  = size
        self.prev  = None  # 첫 프레임에는 prev가 없으므로 None으로 초기화

    def filter(self, q_new: np.ndarray) -> np.ndarray:
        if self.prev is None:
            # 첫 호출: 이전값 없으므로 그대로 통과 + 저장
            self.prev = q_new.copy()
            return q_new.copy()
        self.prev = self.alpha * q_new + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self, q: np.ndarray):
        """
        SYNCING 진입 시 / 트래킹 복귀 시 호출.
        이전값을 현재 실제 로봇 자세로 초기화해서 필터 워밍업 없이 바로 올바른 값을 출력하게 함.
        초기화 없이 재시작하면 이전 텔레옵 값이 남아서 시작할 때 이상하게 움직일 수 있음.
        """
        self.prev = q.copy()


def make_filters() -> dict:
    """
    텔레옵 루프에서 쓰는 EMA 필터 인스턴스 6개를 dict로 반환합니다.

    teleop_ik.py의 초기화 블록에서 arm_filter, wrist_filter_l 등으로 꺼내 쓰면 됩니다.
    alpha 값은 모두 config.py에서 가져오므로 config.py에서 한 번에 조정 가능합니다.

    Returns (key: 용도)
    -------------------
    'arm'         : 팔 IK 관절각 스무딩 (어깨 3개 + 팔꿈치 1개)
    'wrist_l'     : 왼손목 yaw 스무딩
    'wrist_r'     : 오른손목 yaw 스무딩
    'quest_pos_l' : 왼손 Quest 입력 위치 스무딩 (IK 계산 전 전처리)
    'quest_pos_r' : 오른손 Quest 입력 위치 스무딩
    'neck'        : 목 yaw/pitch 스무딩
    """
    return {
        'arm':         EMAFilter(config.EMA_ARM,       size=1),
        'wrist_l':     EMAFilter(config.EMA_WRIST,     size=1),
        'wrist_r':     EMAFilter(config.EMA_WRIST,     size=1),
        'quest_pos_l': EMAFilter(config.EMA_QUEST_POS, size=3),
        'quest_pos_r': EMAFilter(config.EMA_QUEST_POS, size=3),
        'neck':        EMAFilter(config.EMA_NECK,      size=2),
    }


# ══════════════════════════════════════════════════════════════
# 알림음
# ══════════════════════════════════════════════════════════════

def beep(kind: str = 'warn'):
    """
    상태 전환 시 소리로 알림을 줍니다.
    paplay를 백그라운드(&)로 실행하므로 메인 루프를 블로킹하지 않음.

    kind 값과 재생 시점:
      'teleop_start' → 텔레옵 첫 시작 (SYNCING → TELEOP, 최초 1회)
      'warn'         → 트래킹 소실 / 손 위치 점프 / IK 발산 경고
      'sync_done'    → 소실 복귀 후 재싱크 완료 (2번째 이후 TELEOP 진입)
      'calib_start'  → CALIBRATING 상태 진입
      'calib_done'   → 캘리브레이션 완료 (팔/손가락 각각)
    """
    path = config.SOUND.get(kind)
    if path:
        os.system(f'paplay {path} &')


# ══════════════════════════════════════════════════════════════
# 부드러운 자세 이동
# ══════════════════════════════════════════════════════════════

def publish_smooth_move(pub, target_vals: list, current_vals: list = None,
                        duration: float = 2.0, label: str = "이동"):
    """
    현재 자세(current_vals)에서 목표 자세(target_vals)로 선형 보간해 부드럽게 이동합니다.

    왜 필요한가?
    ────────────
    로봇에 목표 관절각을 갑자기 바꿔 publish하면 관절이 순간적으로 점프함.
    50Hz로 current → target을 조금씩 보간해 publish하면 부드럽게 이동함.
    캘리브 시작, 손가락 캘리브 시작, 종료 시에 사용됨.

    Parameters
    ----------
    pub          : rospy.Publisher (Float64MultiArray)
    target_vals  : 목표 관절각 리스트 (12개)
    current_vals : 시작 관절각 리스트. None이면 보간 없이 즉시 target 전송.
    duration     : 이동 소요 시간 [s]
    label        : 로그 출력용 이름
    """
    if current_vals is None:
        # 현재 자세를 모를 때는 즉시 전송 (보간 불가)
        cmd      = Float64MultiArray()
        cmd.data = target_vals
        pub.publish(cmd)
        return

    hz    = config.CONTROL_HZ
    steps = int(duration * hz)
    print(f"🎬 {duration}초 동안 [{label}] 자세로 부드럽게 이동합니다...")
    for i in range(1, steps + 1):
        t     = i / steps  # 0~1 선형 비율
        interp = [c + (g - c) * t for c, g in zip(current_vals, target_vals)]
        cmd      = Float64MultiArray()
        cmd.data = interp
        pub.publish(cmd)
        time.sleep(1.0 / hz)   # rospy.Rate 대신 time.sleep 사용


def publish_init(pub, current_vals=None):
    """
    초기 자세(앞으로 나란히, CALIB_POS)로 부드럽게 이동합니다.
    캘리브레이션 시작 시 로봇을 기준 자세로 세우기 위해 호출됨.
    """
    publish_smooth_move(pub, config.CALIB_POS, current_vals,
                        duration=2.0, label="캘리브레이션(초기)")


def publish_fin(pub, current_vals=None, duration=2.5):
    """
    종료 자세(INIT_POS)로 부드럽게 이동합니다.
    Ctrl+C 종료 시 finally 블록에서 호출되어 로봇을 안전 자세로 복귀시킴.
    """
    publish_smooth_move(pub, config.INIT_POS, current_vals,
                        duration=duration, label="종료")
