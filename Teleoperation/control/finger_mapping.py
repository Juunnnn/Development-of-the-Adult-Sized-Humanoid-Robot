"""
finger_mapping.py  — AmazingHand 방식 (3-point 각도 → 선형 매핑)
─────────────────────────────────────────────────────────────────────
[리타게팅 원리]
  1. 각 손가락 세 점 p1(proximal), p2(vertex=MCP), p3(tip)로
     꼭짓점 각도(angle at vertex) 계산
  2. 굽힘각 = 180° − angle_at_vertex
       → 0°:  완전 펼침 (collinear)
       → 160°: 완전 구부림
  3. 선형 매핑: 굽힘각 → FE 관절각 (라디안)
       ANGLE_OPEN  → FE_MAX = 0 rad (완전 펼침)
       ANGLE_CLOSE → FE_MIN = −1.501 rad (완전 구부림)
  4. AA는 0으로 고정 (USE_AA = True 로 바꾸면 실험적 AA 추가)

[Quest landmark 인덱스 — WebXR 25-joint (vuer Hands 컴포넌트 기준)]
  ※ Meta Quest 3S + vuer는 MediaPipe(21점)가 아닌 WebXR 25관절 체계를 사용함

   0: wrist
   1: thumb-metacarpal
   2: thumb-phalanx-proximal  (= MCP)
   3: thumb-phalanx-distal    (= IP)
   4: thumb-tip
   5: index-finger-metacarpal
   6: index-finger-phalanx-proximal   (= MCP, 너클)
   7: index-finger-phalanx-intermediate (= PIP)
   8: index-finger-phalanx-distal      (= DIP)
   9: index-finger-tip
  10: middle-finger-metacarpal
  11: middle-finger-phalanx-proximal
  12: middle-finger-phalanx-intermediate
  13: middle-finger-phalanx-distal
  14: middle-finger-tip
  15: ring-finger-metacarpal
  16: ring-finger-phalanx-proximal
  17: ring-finger-phalanx-intermediate
  18: ring-finger-phalanx-distal
  19: ring-finger-tip
  20: pinky-finger-metacarpal
  21: pinky-finger-phalanx-proximal    (= MCP)
  22: pinky-finger-phalanx-intermediate
  23: pinky-finger-phalanx-distal
  24: pinky-finger-tip

[출력 배열 — 16개 관절각, 단위: rad]
  [L_AA_1, L_FE_1,  L_AA_2, L_FE_2,  L_AA_3, L_FE_3,  L_AA_4, L_FE_4,
   R_AA_1, R_FE_1,  R_AA_2, R_FE_2,  R_AA_3, R_FE_3,  R_AA_4, R_FE_4]
  손가락 번호: 1=엄지, 2=검지, 3=중지, 4=약지(Quest 소지 랜드마크 사용)
  AA는 항상 0
"""

from __future__ import annotations
import numpy as np

# ══════════════════════════════════════════════════════════════════
# §0  튜닝 파라미터
# ══════════════════════════════════════════════════════════════════

# ── 손가락별 굽힘각 범위 (실측값 기반) ───────────────────────────
#   ANGLE_OPEN  이하 → FE = 0 rad    (완전 펼침)
#   ANGLE_CLOSE 이상 → FE = FE_MIN   (완전 구부림)
#
#   튜닝 가이드:
#     손이 펼쳐져도 로봇 손가락이 구부러져 있으면 → ANGLE_OPEN ↓
#     손을 쥐었는데 로봇이 덜 구부러지면         → ANGLE_CLOSE ↓
#     FINGER_DEBUG = True 로 실시간 각도 확인 가능
ANGLE_OPEN: dict[int, float] = {
    1: 14.0,   # 엄지  (펼침 실측 ≈ 14°)
    2:  7.0,   # 검지  (펼침 실측 ≈  7°)
    3:  7.0,   # 중지  (펼침 실측 ≈  7°)
    4:  9.0,   # 약지←소지  (펼침 실측 ≈ 9°)
}
ANGLE_CLOSE: dict[int, float] = {
    1:  53.0,  # 엄지  (구부림 실측 ≈  53°)
    2: 130.0,  # 검지  (구부림 실측 ≈ 130°)
    3: 139.0,  # 중지  (구부림 실측 ≈ 139°)
    4: 147.0,  # 약지←소지  (구부림 실측 ≈ 147°)
}

# ── 로봇 FE/AA 관절 한계 (URDF 실측) ─────────────────────────────
FE_MIN = -1.501   # -86° (완전 구부림)
FE_MAX =  0.000   #  0°  (완전 펼침)
AA_MIN = -0.349   # -20°
AA_MAX =  0.349   # +20°

# ── AA 활성화 (False = AmazingHand 원본 동작, True = 실험적) ──────
USE_AA = False
AA_XY_THRESH = 0.15   # USE_AA=True 시: 이 미만이면 AA=0


# ══════════════════════════════════════════════════════════════════
# §1  Quest landmark 인덱스 정의 (WebXR 25-joint 체계)
# ══════════════════════════════════════════════════════════════════

# 로봇 손가락 번호 → (p1, p2_vertex, p3) 인덱스
# 굽힘각 = 180° − angle_at(p2) between vec(p1→p2) and vec(p3→p2)
FINGER_ANGLE_POINTS: dict[int, tuple[int, int, int]] = {
    1: ( 0,  2,  4),   # 엄지:       Wrist(0) → ThumbMCP(2)  → ThumbTip(4)
    2: ( 0,  6,  9),   # 검지:       Wrist(0) → IndexMCP(6)  → IndexTip(9)
    3: ( 0, 11, 14),   # 중지:       Wrist(0) → MiddleMCP(11) → MiddleTip(14)
    4: ( 0, 21, 24),   # 약지←소지: Wrist(0) → PinkyMCP(21) → PinkyTip(24)
}

# AA 계산용 손바닥 프레임 landmark 인덱스
WRIST_IDX  =  0
MIDDLE_MCP = 11
INDEX_MCP  =  6
PINKY_MCP  = 21


# ══════════════════════════════════════════════════════════════════
# §2  핵심 계산 함수
# ══════════════════════════════════════════════════════════════════

def _angle_at_vertex(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """꼭짓점 p2에서의 각도 [deg] 계산."""
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-7 or n2 < 1e-7:
        return 180.0
    cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _flex_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """굽힘각 = 180° − angle_at_vertex [deg]. 펼침≈0°, 구부림≈160°."""
    return 180.0 - _angle_at_vertex(p1, p2, p3)


def _map_value(x: float, in_min: float, in_max: float,
               out_min: float, out_max: float) -> float:
    """선형 보간 + 범위 클램핑."""
    x = float(np.clip(x, min(in_min, in_max), max(in_min, in_max)))
    if abs(in_max - in_min) < 1e-9:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def _flex_to_fe(flex_deg: float, f: int) -> float:
    """굽힘각(deg) → FE 관절각(rad). 손가락 번호 f별 범위 적용."""
    fe = _map_value(flex_deg, ANGLE_OPEN[f], ANGLE_CLOSE[f], FE_MAX, FE_MIN)
    return float(np.clip(fe, FE_MIN, FE_MAX))


# ══════════════════════════════════════════════════════════════════
# §3  AA 계산 (USE_AA=True 시 활성화)
# ══════════════════════════════════════════════════════════════════

def _palm_frame(lm: np.ndarray) -> np.ndarray:
    """손바닥 로컬 좌표계 회전행렬 R_palm (3×3) 반환."""
    y_vec = lm[MIDDLE_MCP] - lm[WRIST_IDX]
    pl = np.linalg.norm(y_vec)
    if pl < 1e-6:
        return np.eye(3)
    y_axis = y_vec / pl

    v1, v2 = lm[INDEX_MCP] - lm[WRIST_IDX], lm[PINKY_MCP] - lm[WRIST_IDX]
    z_raw  = np.cross(v1, v2)
    zn     = np.linalg.norm(z_raw)
    z_axis = z_raw / zn if zn > 1e-6 else np.array([0., 0., 1.])

    x_raw  = np.cross(y_axis, z_axis)
    xn     = np.linalg.norm(x_raw)
    x_axis = x_raw / xn if xn > 1e-6 else np.array([1., 0., 0.])
    z_axis = np.cross(x_axis, y_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


def _compute_aa(lm: np.ndarray, f: int, R_palm: np.ndarray, is_left: bool) -> float:
    """손바닥 로컬 프레임에서 MCP→Tip 벡터의 가로 성분으로 AA 계산."""
    mcp_i, tip_i = FINGER_ANGLE_POINTS[f][1], FINGER_ANGLE_POINTS[f][2]
    vec = lm[tip_i] - lm[mcp_i]
    d = np.linalg.norm(vec)
    if d < 1e-6:
        return 0.0
    tip_local = R_palm.T @ (vec / d)
    lx, ly = float(tip_local[0]), float(tip_local[1])
    aa = float(np.arctan2(lx, ly)) if np.hypot(lx, ly) > AA_XY_THRESH else 0.0
    if not is_left:
        aa = -aa
    return float(np.clip(aa, AA_MIN, AA_MAX))


# ══════════════════════════════════════════════════════════════════
# §4  한 손 리타게팅
# ══════════════════════════════════════════════════════════════════

def _retarget_hand(lm: np.ndarray, is_left: bool) -> np.ndarray:
    """Quest landmark (25,3) → 8개 관절각 [AA_1, FE_1, ..., AA_4, FE_4]."""
    R_palm = _palm_frame(lm) if USE_AA else None
    cmd = np.zeros(8)
    for f in range(1, 5):
        p1_i, p2_i, p3_i = FINGER_ANGLE_POINTS[f]
        cmd[(f-1)*2 + 1] = _flex_to_fe(_flex_angle(lm[p1_i], lm[p2_i], lm[p3_i]), f)
        if USE_AA and R_palm is not None:
            cmd[(f-1)*2] = _compute_aa(lm, f, R_palm, is_left)
    return cmd


# ══════════════════════════════════════════════════════════════════
# §5  공개 인터페이스
# ══════════════════════════════════════════════════════════════════

def is_landmark_valid(lm: object) -> bool:
    """Quest 핸드 트래킹 데이터 유효성 검사."""
    return (lm is not None
            and isinstance(lm, np.ndarray)
            and lm.shape == (25, 3)
            and not np.any(np.isnan(lm))
            and not np.allclose(lm, 0, atol=1e-6))


class FingerEMAFilter:
    """16채널 지수이동평균(EMA) 필터."""

    def __init__(self, alpha: float = 0.4, n: int = 16) -> None:
        self.alpha = alpha
        self.n     = n
        self._prev: np.ndarray | None = None

    def filter(self, cmd: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = cmd.copy()
            return cmd.copy()
        self._prev = self.alpha * cmd + (1.0 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self, cmd: np.ndarray | None = None) -> None:
        self._prev = None if cmd is None else cmd.copy()


def build_hand_cmd(
    left_lm:  np.ndarray | None,
    right_lm: np.ndarray | None,
) -> np.ndarray:
    """
    양손 Quest landmark → 16개 관절각.

    Returns
    -------
    cmd : ndarray (16,)
        [L_AA_1, L_FE_1, L_AA_2, L_FE_2, L_AA_3, L_FE_3, L_AA_4, L_FE_4,
         R_AA_1, R_FE_1, R_AA_2, R_FE_2, R_AA_3, R_FE_3, R_AA_4, R_FE_4]
    """
    l_cmd = _retarget_hand(left_lm,  is_left=True)  if is_landmark_valid(left_lm)  else np.zeros(8)
    r_cmd = _retarget_hand(right_lm, is_left=False) if is_landmark_valid(right_lm) else np.zeros(8)
    return np.concatenate([l_cmd, r_cmd])