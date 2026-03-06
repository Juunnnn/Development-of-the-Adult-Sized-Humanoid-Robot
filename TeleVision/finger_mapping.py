"""
finger_mapping.py
─────────────────────────────────────────────────────────────
Amazing Hand 손가락 텔레오퍼레이션 모듈

로봇 손가락 구성:
  1 = 엄지 (thumb)   → Quest landmark  1~4
  2 = 검지 (index)   → Quest landmark  5~8
  3 = 중지 (middle)  → Quest landmark  9~12
  4 = 약지 (ring)    → Quest landmark 13~16

관절:
  AA  (Abduction/Adduction) : 좌우 벌림, ±20° (±0.349 rad)
  FE  (Flexion/Extension)   : 굽힘,      0 ~ -86° (0 ~ -1.501 rad)
  FE_follower               : FE 따라서 자동 움직임 → 제어 불필요

ROS topic: /hand_controller/command  (Float64MultiArray, 16개)
  cmd 인덱스:
    L: [L_AA_1, L_FE_1, L_AA_2, L_FE_2, L_AA_3, L_FE_3, L_AA_4, L_FE_4]   (0~7)
    R: [R_AA_1, R_FE_1, R_AA_2, R_FE_2, R_AA_3, R_FE_3, R_AA_4, R_FE_4]   (8~15)

Quest landmark 좌표계 (WebXR):
  - 0번 = 손목
  - 각 손가락 4개 관절: MCP, PIP, DIP, Tip 순
  - 손등 법선을 구해 FE 굽힘각 계산
  - 인접 손가락 간 벡터 각도로 AA 계산
"""

import numpy as np
from enum import Enum, auto

# ── 관절 한계 ──────────────────────────────────────────────
AA_MIN = -0.349   # rad  (-20°)
AA_MAX =  0.349   # rad  (+20°)
FE_MIN = -1.501   # rad  (-86°)  → 완전히 구부러진 상태
FE_MAX =  0.000   # rad  (  0°)  → 완전히 펼쳐진 상태

# ── Quest landmark 인덱스 정의 ──────────────────────────────
# 각 손가락: [MCP, PIP, DIP, TIP]
THUMB_IDX  = [1,  2,  3,  4]
INDEX_IDX  = [5,  6,  7,  8]
MIDDLE_IDX = [9,  10, 11, 12]
RING_IDX   = [13, 14, 15, 16]

# 로봇 1~4번 ↔ Quest 손가락 매핑
# robot finger i → quest landmark indices
FINGER_MAP = {
    1: THUMB_IDX,   # 로봇 1(엄지) ← Quest 엄지
    2: INDEX_IDX,   # 로봇 2(검지) ← Quest 검지
    3: MIDDLE_IDX,  # 로봇 3(중지) ← Quest 중지
    4: RING_IDX,    # 로봇 4(약지) ← Quest 약지
}

# AA: 인접 손가락 쌍 (로봇 번호 기준)
# (기준손가락, 비교손가락) → 둘 사이 벌어진 각도를 기준 손가락의 AA에 반영
AA_PAIRS = {
    1: (1, 2),   # 엄지 AA: 엄지-검지 벌어짐
    2: (2, 3),   # 검지 AA: 검지-중지 벌어짐
    3: (3, 4),   # 중지 AA: 중지-약지 벌어짐
    4: (3, 4),   # 약지 AA: 중지-약지 벌어짐 (같은 쌍, 부호 반전)
}

# ── EMA 필터 (손가락용) ──────────────────────────────────────
class FingerEMAFilter:
    """
    손가락 16채널 EMA 필터
    alpha: 높을수록 빠른 반응 (낮을수록 부드러움)
    """
    def __init__(self, alpha=0.4, n=16):
        self.alpha = alpha
        self.n = n
        self.prev = None

    def filter(self, cmd: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = cmd.copy()
            return cmd.copy()
        self.prev = self.alpha * cmd + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self, cmd: np.ndarray = None):
        if cmd is None:
            self.prev = None
        else:
            self.prev = cmd.copy()


# ── 핵심 계산 함수 ───────────────────────────────────────────

def _get_palm_normal(lm: np.ndarray) -> np.ndarray:
    """
    손바닥 법선 벡터 계산.
    손목(0), 검지MCP(5), 소지MCP(17)으로 평면 구성.
    법선이 손등 방향을 향하도록 설정.
    lm: (25, 3) landmark 배열
    """
    wrist   = lm[0]
    idx_mcp = lm[5]
    ring_mcp = lm[13]

    v1 = idx_mcp - wrist
    v2 = ring_mcp - wrist
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def calc_fe(lm: np.ndarray, finger_idx: int) -> float:
    """
    FE(굽힘) 각도 계산.

    방법:
      MCP → Tip 벡터와 손바닥 법선 사이 각도로 굽힘 정도(0~1) 추정
      - 손가락이 완전히 펴져 있으면 법선과 거의 수직 → flex_ratio ≈ 0
      - 손가락이 완전히 구부러지면 법선과 거의 평행 → flex_ratio ≈ 1

    반환: FE 관절각 (rad), 범위 [FE_MIN, FE_MAX]
    """
    idxs = FINGER_MAP[finger_idx]
    mcp = lm[idxs[0]]
    tip = lm[idxs[3]]

    finger_vec = tip - mcp
    fn = np.linalg.norm(finger_vec)
    if fn < 1e-6:
        return 0.0

    finger_vec = finger_vec / fn
    palm_normal = _get_palm_normal(lm)

    # 법선과 손가락 벡터의 dot product = cos(각도)
    # 완전히 펼쳤을 때: finger_vec ⊥ normal → dot ≈ 0
    # 구부렸을 때: finger_vec ∥ normal → |dot| ≈ 1
    dot = np.clip(np.dot(finger_vec, palm_normal), -1.0, 1.0)
    flex_ratio = np.abs(dot)   # 0(펼침) ~ 1(구부림)

    # FE 각도로 변환 (FE_MIN이 음수 = 구부림)
    fe_angle = FE_MIN * flex_ratio
    return float(np.clip(fe_angle, FE_MIN, FE_MAX))


def calc_aa(lm: np.ndarray, finger_idx: int, is_left: bool) -> float:
    """
    AA(벌림) 각도 계산.

    방법:
      인접 두 손가락의 MCP 위치 차이 벡터를 손바닥 평면에 투영한 후
      손가락 기준 방향과의 각도로 AA 계산.

    반환: AA 관절각 (rad), 범위 [AA_MIN, AA_MAX]
    """
    a_finger, b_finger = AA_PAIRS[finger_idx]
    a_mcp = lm[FINGER_MAP[a_finger][0]]
    b_mcp = lm[FINGER_MAP[b_finger][0]]

    # 두 MCP를 잇는 벡터 (손바닥 가로 방향)
    sep_vec = b_mcp - a_mcp
    sep_norm = np.linalg.norm(sep_vec)
    if sep_norm < 1e-6:
        return 0.0

    # 손바닥 법선
    palm_normal = _get_palm_normal(lm)

    # 손바닥 평면에 투영
    sep_proj = sep_vec - np.dot(sep_vec, palm_normal) * palm_normal
    proj_norm = np.linalg.norm(sep_proj)
    if proj_norm < 1e-6:
        return 0.0
    sep_proj = sep_proj / proj_norm

    # 기준 방향: 손목→검지MCP 방향 (손바닥 평면 내 기준축)
    wrist = lm[0]
    idx_mcp = lm[FINGER_MAP[2][0]]   # 검지 MCP
    ref_vec = idx_mcp - wrist
    ref_proj = ref_vec - np.dot(ref_vec, palm_normal) * palm_normal
    ref_norm = np.linalg.norm(ref_proj)
    if ref_norm < 1e-6:
        return 0.0
    ref_proj = ref_proj / ref_norm

    # 두 벡터 간 부호 있는 각도
    cross = np.cross(ref_proj, sep_proj)
    sign = np.sign(np.dot(cross, palm_normal))
    cos_angle = np.clip(np.dot(ref_proj, sep_proj), -1.0, 1.0)
    angle = sign * np.arccos(cos_angle)

    # 약지(4번)는 중지-약지 쌍이지만 약지 입장에서 반대 부호
    if finger_idx == 4:
        angle = -angle

    # 왼손/오른손 부호 대칭
    if not is_left:
        angle = -angle

    return float(np.clip(angle, AA_MIN, AA_MAX))


def landmarks_to_finger_cmd(lm: np.ndarray, is_left: bool,
                              calib_lm: np.ndarray = None) -> np.ndarray:
    """
    Quest landmark (25, 3) → 손가락 관절각 8개 배열

    출력 순서 (한 손):
      [AA_1, FE_1, AA_2, FE_2, AA_3, FE_3, AA_4, FE_4]

    calib_lm: 캘리브레이션 시점 landmark (현재 미사용, 추후 오프셋 보정용)
    """
    cmd = np.zeros(8)

    for robot_finger in range(1, 5):   # 1~4
        aa_idx = (robot_finger - 1) * 2       # 0, 2, 4, 6
        fe_idx = (robot_finger - 1) * 2 + 1  # 1, 3, 5, 7

        cmd[aa_idx] = calc_aa(lm, robot_finger, is_left)
        cmd[fe_idx] = calc_fe(lm, robot_finger)

    return cmd


def build_hand_cmd(left_lm: np.ndarray, right_lm: np.ndarray,
                   calib_left_lm: np.ndarray = None,
                   calib_right_lm: np.ndarray = None) -> np.ndarray:
    """
    양손 landmark → 16개 관절각 배열 (ROS publish용)

    출력: [L_AA_1, L_FE_1, ..., L_AA_4, L_FE_4,
           R_AA_1, R_FE_1, ..., R_AA_4, R_FE_4]
    """
    left_cmd  = landmarks_to_finger_cmd(left_lm,  is_left=True,  calib_lm=calib_left_lm)
    right_cmd = landmarks_to_finger_cmd(right_lm, is_left=False, calib_lm=calib_right_lm)
    return np.concatenate([left_cmd, right_cmd])


# ── landmark 유효성 검사 ─────────────────────────────────────

def is_landmark_valid(lm: np.ndarray, threshold: float = 1e-6) -> bool:
    """
    landmark가 모두 0이거나 NaN이면 유효하지 않음.
    """
    if lm is None:
        return False
    if np.any(np.isnan(lm)):
        return False
    if np.allclose(lm, 0, atol=threshold):
        return False
    return True
