"""
finger_mapping.py  — AmazingHand 방식 (3-point 각도 → 선형 매핑)
─────────────────────────────────────────────────────────────────────
[리타게팅 원리]
  AmazingHand 원본 방식 그대로:
  1. 각 손가락 세 점 p1(proximal), p2(vertex=MCP), p3(tip)로
     꼭짓점 각도(angle at vertex) 계산
  2. 굽힘각 = 180° − angle_at_vertex
       → 0°:  완전 펼침 (collinear)
       → 160°: 완전 구부림
  3. 선형 매핑: 굽힘각 → FE 관절각 (라디안)
       ANGLE_OPEN  (20°)  → FE_MAX = 0 rad (완전 펼침)
       ANGLE_CLOSE (160°) → FE_MIN = −1.501 rad (완전 구부림)
  4. AA는 AmazingHand 원본과 같이 0으로 고정
     (USE_AA = True 로 바꾸면 손바닥 로컬 프레임 기반 AA 추가 가능)

[Quest landmark 인덱스 — MediaPipe 21-point 기준]
  0 = Wrist
  1 = Thumb CMC,  2 = Thumb MCP,  3 = Thumb IP,   4 = Thumb Tip
  5 = Index MCP,  6 = Index PIP,  7 = Index DIP,  8 = Index Tip
  9 = Middle MCP, 10= Middle PIP, 11= Middle DIP, 12= Middle Tip
  13= Ring MCP,   14= Ring PIP,   15= Ring DIP,   16= Ring Tip
  17= Pinky MCP,  18= Pinky PIP,  19= Pinky DIP,  20= Pinky Tip
  (vuer로부터 받은 25×3 배열에서 상위 21점이 MediaPipe 순서)

[출력 배열 — 16개 관절각, 단위: rad]
  [L_AA_1, L_FE_1,  L_AA_2, L_FE_2,  L_AA_3, L_FE_3,  L_AA_4, L_FE_4,
   R_AA_1, R_FE_1,  R_AA_2, R_FE_2,  R_AA_3, R_FE_3,  R_AA_4, R_FE_4]
  손가락 번호: 1=엄지, 2=검지, 3=중지, 4=약지
  AA는 항상 0 (AmazingHand 원본 동작)
"""

from __future__ import annotations
import numpy as np

# ══════════════════════════════════════════════════════════════════
# §0  튜닝 파라미터
# ══════════════════════════════════════════════════════════════════

# ── 굽힘각 범위 (AmazingHand 원본값) ─────────────────────────────
#   굽힘각 = 180° − 꼭짓점 각도
ANGLE_OPEN  = 60.0    # [deg] 이 이하 → 완전 펼침 (FE = 0)
ANGLE_CLOSE = 160.0   # [deg] 이 이상 → 완전 구부림 (FE = FE_MIN)

# ── 로봇 FE/AA 관절 한계 (URDF 실측) ─────────────────────────────
FE_MIN = -1.501    # -86°  (완전 구부림)
FE_MAX =  0.000    # 0°    (완전 펼침)
AA_MIN = -0.349    # -20°
AA_MAX =  0.349    # +20°

# ── AA 활성화 플래그 ──────────────────────────────────────────────
#   False: AmazingHand 원본과 동일 (AA=0)
#   True:  손바닥 로컬 프레임 기반 AA 추가 (실험적)
USE_AA = False

# ── AA 계산 임계값 (USE_AA=True 시 사용) ──────────────────────────
AA_XY_THRESH = 0.15   # 손바닥 평면 투영 벡터 크기가 이 미만이면 AA=0


# ══════════════════════════════════════════════════════════════════
# §1  Quest landmark 인덱스 정의
# ══════════════════════════════════════════════════════════════════

# 로봇 손가락 번호 → (p1, p2_vertex, p3) 인덱스
#   굽힘각 = 180° − angle_at(p2) between vec(p1→p2) and vec(p3→p2)
FINGER_ANGLE_POINTS: dict[int, tuple[int, int, int]] = {
    1: ( 2,  3,  4),   # 엄지: Thumb-CMC → Thumb-MCP → Thumb-Tip
    2: ( 0,  5,  8),   # 검지: Wrist → Index-MCP → Index-Tip
    3: ( 0,  9, 12),   # 중지: Wrist → Middle-MCP → Middle-Tip
    4: ( 0, 13, 16),   # 약지: Wrist → Ring-MCP → Ring-Tip
}

# AA 계산용 손바닥 프레임 landmark 인덱스
WRIST_IDX  =  0
MIDDLE_MCP =  9
INDEX_MCP  =  5
PINKY_MCP  = 17


# ══════════════════════════════════════════════════════════════════
# §2  핵심 계산 함수
# ══════════════════════════════════════════════════════════════════

def _angle_at_vertex(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    꼭짓점 p2에서의 각도 [deg] 계산.
    vec1 = p1 − p2,  vec2 = p3 − p2
    반환: arccos(dot(v1,v2) / (|v1||v2|)) in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-7 or n2 < 1e-7:
        return 180.0   # 길이 0이면 직선으로 간주 → 굽힘 없음
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _flex_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    굽힘각 = 180° − angle_at_vertex(p1, p2, p3) [deg].
    - 펼침: ≈ 0°  (collinear → vertex angle ≈ 180°)
    - 구부림: ≈ 160°
    """
    return 180.0 - _angle_at_vertex(p1, p2, p3)


def _map_value(x: float, in_min: float, in_max: float,
               out_min: float, out_max: float) -> float:
    """
    AmazingHand 원본 map_value: 선형 보간 + 범위 클램핑.
    """
    x = float(np.clip(x, min(in_min, in_max), max(in_min, in_max)))
    if abs(in_max - in_min) < 1e-9:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def _flex_to_fe(flex_deg: float) -> float:
    """
    굽힘각(deg) → FE 관절각(rad).
    ANGLE_OPEN  → FE_MAX (0 rad, 펼침)
    ANGLE_CLOSE → FE_MIN (−1.501 rad, 구부림)
    """
    fe = _map_value(flex_deg, ANGLE_OPEN, ANGLE_CLOSE, FE_MAX, FE_MIN)
    return float(np.clip(fe, FE_MIN, FE_MAX))


# ══════════════════════════════════════════════════════════════════
# §3  AA 계산 (USE_AA=True 시 활성화)
# ══════════════════════════════════════════════════════════════════

def _palm_frame(lm: np.ndarray) -> np.ndarray:
    """
    손바닥 로컬 좌표계 회전행렬 R_palm (3×3) 반환.
    열: [x=가로, y=손 길이방향, z=손바닥 법선]
    """
    y_vec = lm[MIDDLE_MCP] - lm[WRIST_IDX]
    pl    = np.linalg.norm(y_vec)
    if pl < 1e-6:
        return np.eye(3)
    y_axis = y_vec / pl

    v1    = lm[INDEX_MCP] - lm[WRIST_IDX]
    v2    = lm[PINKY_MCP] - lm[WRIST_IDX]
    z_raw = np.cross(v1, v2)
    zn    = np.linalg.norm(z_raw)
    z_axis = z_raw / zn if zn > 1e-6 else np.array([0., 0., 1.])

    x_raw  = np.cross(y_axis, z_axis)
    xn     = np.linalg.norm(x_raw)
    x_axis = x_raw / xn if xn > 1e-6 else np.array([1., 0., 0.])
    z_axis = np.cross(x_axis, y_axis)   # 재직교화

    return np.column_stack([x_axis, y_axis, z_axis])


def _compute_aa(lm: np.ndarray, f: int, R_palm: np.ndarray,
                is_left: bool) -> float:
    """
    손바닥 로컬 프레임에서 MCP→Tip 벡터의 가로(x) 성분으로 AA 계산.
    오른손은 좌우 반전.
    """
    mcp_i = FINGER_ANGLE_POINTS[f][1]
    tip_i = FINGER_ANGLE_POINTS[f][2]
    vec   = lm[tip_i] - lm[mcp_i]
    d     = np.linalg.norm(vec)
    if d < 1e-6:
        return 0.0
    tip_unit  = vec / d
    tip_local = R_palm.T @ tip_unit   # [x=가로, y=앞, z=법선]
    lx = float(tip_local[0])
    ly = float(tip_local[1])
    if np.hypot(lx, ly) > AA_XY_THRESH:
        aa = float(np.arctan2(lx, ly))
    else:
        aa = 0.0
    if not is_left:
        aa = -aa   # 오른손 부호 반전
    return float(np.clip(aa, AA_MIN, AA_MAX))


# ══════════════════════════════════════════════════════════════════
# §4  한 손 리타게팅
# ══════════════════════════════════════════════════════════════════

def _retarget_hand(lm: np.ndarray, is_left: bool) -> np.ndarray:
    """
    Quest landmark (25,3) → 8개 관절각 배열.

    반환: [AA_1, FE_1, AA_2, FE_2, AA_3, FE_3, AA_4, FE_4]
    손가락 1=엄지, 2=검지, 3=중지, 4=약지
    """
    R_palm = _palm_frame(lm) if USE_AA else None

    cmd = np.zeros(8)
    for f in range(1, 5):
        p1_i, p2_i, p3_i = FINGER_ANGLE_POINTS[f]
        p1, p2, p3       = lm[p1_i], lm[p2_i], lm[p3_i]

        # ── FE: AmazingHand 3-point 각도 방식 ─────────────────
        flex_deg      = _flex_angle(p1, p2, p3)
        fe            = _flex_to_fe(flex_deg)
        cmd[(f-1)*2+1] = fe

        # ── AA: 원본 AmazingHand=0, USE_AA=True면 계산 ────────
        if USE_AA and R_palm is not None:
            cmd[(f-1)*2] = _compute_aa(lm, f, R_palm, is_left)
        # else: cmd[(f-1)*2] = 0.0 (기본값)

    return cmd


# ══════════════════════════════════════════════════════════════════
# §5  공개 인터페이스 (teleop_ik.py에서 호출, API 호환)
# ══════════════════════════════════════════════════════════════════

def is_landmark_valid(lm: object) -> bool:
    """Quest 핸드 트래킹 데이터 유효성 검사."""
    if lm is None:
        return False
    if not isinstance(lm, np.ndarray):
        return False
    if lm.shape != (25, 3):
        return False
    if np.any(np.isnan(lm)):
        return False
    if np.allclose(lm, 0, atol=1e-6):
        return False
    return True


class FingerEMAFilter:
    """16채널 지수이동평균(EMA) 필터. teleop_ik.py에서 호출."""

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
    # 하위 호환 인자 (사용 안 함)
    L_calib_left:  np.ndarray | None = None,
    L_calib_right: np.ndarray | None = None,
) -> np.ndarray:
    """
    양손 Quest landmark → 16개 관절각.

    Parameters
    ----------
    left_lm  : (25,3) or None
    right_lm : (25,3) or None

    Returns
    -------
    cmd : ndarray (16,)
        [L_AA_1, L_FE_1, L_AA_2, L_FE_2, L_AA_3, L_FE_3, L_AA_4, L_FE_4,
         R_AA_1, R_FE_1, R_AA_2, R_FE_2, R_AA_3, R_FE_3, R_AA_4, R_FE_4]

    Notes
    -----
    FE_follower 관절은 여기서 계산하지 않음.
    ros_interface.py에서 FE × 0.93 적용 필요 (mimic_fe_follower.py와 동일).
    """
    l_cmd = (_retarget_hand(left_lm,  is_left=True)
             if is_landmark_valid(left_lm)
             else np.zeros(8))
    r_cmd = (_retarget_hand(right_lm, is_left=False)
             if is_landmark_valid(right_lm)
             else np.zeros(8))
    return np.concatenate([l_cmd, r_cmd])


def compute_finger_calib(lm: np.ndarray) -> np.ndarray:
    """하위 호환용 더미. AmazingHand 방식은 캘리브 불필요."""
    return np.zeros(4)


# ══════════════════════════════════════════════════════════════════
# §6  단독 실행 검증
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("  finger_mapping — AmazingHand 방식 검증")
    print(f"  ANGLE_OPEN={ANGLE_OPEN}° → FE=0 rad")
    print(f"  ANGLE_CLOSE={ANGLE_CLOSE}° → FE={FE_MIN} rad ({math.degrees(FE_MIN):.1f}°)")
    print(f"  USE_AA = {USE_AA}")
    print("=" * 60)

    def _make_lm(palm: float = 0.09) -> np.ndarray:
        lm = np.zeros((25, 3))
        lm[0]  = [0, 0, 0]                         # wrist
        lm[2]  = [ 0.02, palm * 0.35, 0]           # thumb CMC
        lm[3]  = [ 0.025, palm * 0.50, 0]          # thumb MCP
        lm[5]  = [ 0.02,  palm, 0]                 # index MCP
        lm[9]  = [ 0.005, palm, 0]                 # middle MCP
        lm[13] = [-0.015, palm * 0.97, 0]          # ring MCP
        lm[17] = [-0.030, palm * 0.93, 0]          # pinky MCP
        return lm

    # ── 검증 1: 완전 펼침 → FE ≈ 0 ──────────────────────────────
    print("\n[1] 완전 펼침: 굽힘각≈0° → FE≈0 예상")
    lm = _make_lm()
    PALM = 0.09
    EXT = 0.08   # 손가락 길이

    # 각 손가락 tip을 MCP에서 y방향(손 길이방향)으로 배치
    tip_positions = {1: (3, 4), 2: (5, 8), 3: (9, 12), 4: (13, 16)}
    for f, (mcp_i, tip_i) in tip_positions.items():
        lm[tip_i] = lm[mcp_i] + np.array([0, EXT, 0])   # straight along y

    cmd = build_hand_cmd(lm, lm)[:8]
    for f in range(1, 5):
        aa = math.degrees(cmd[(f-1)*2])
        fe = math.degrees(cmd[(f-1)*2+1])
        flex = _flex_angle(lm[FINGER_ANGLE_POINTS[f][0]],
                           lm[FINGER_ANGLE_POINTS[f][1]],
                           lm[FINGER_ANGLE_POINTS[f][2]])
        ok = "✓" if abs(fe) < 3 else "✗"
        print(f"  f{f}: flex={flex:.1f}°  AA={aa:+.1f}°  FE={fe:+.1f}°  {ok}")

    # ── 검증 2: 완전 구부림 → FE ≈ -86° ─────────────────────────
    print("\n[2] 완전 구부림: 굽힘각≈160° → FE≈-86° 예상")
    lm2 = _make_lm()
    # tip을 MCP에서 -y 방향(손목 방향)으로 배치 → vertex angle ≈ 20° → flex ≈ 160°
    for f, (mcp_i, tip_i) in tip_positions.items():
        lm2[tip_i] = lm2[mcp_i] + np.array([0, -EXT * 0.5, 0])

    cmd2 = build_hand_cmd(lm2, lm2)[:8]
    for f in range(1, 5):
        fe = math.degrees(cmd2[(f-1)*2+1])
        flex = _flex_angle(lm2[FINGER_ANGLE_POINTS[f][0]],
                           lm2[FINGER_ANGLE_POINTS[f][1]],
                           lm2[FINGER_ANGLE_POINTS[f][2]])
        ok = "✓" if fe < -50 else "✗"
        print(f"  f{f}: flex={flex:.1f}°  FE={fe:+.1f}°  {ok}")

    # ── 검증 3: 출력 인덱스 확인 ─────────────────────────────────
    print("\n[3] 출력 배열 인덱스 (16개, 절반 구부림)")
    lm3 = _make_lm()
    for f, (mcp_i, tip_i) in tip_positions.items():
        lm3[tip_i] = lm3[mcp_i] + np.array([0, EXT * 0.2, EXT * 0.4])  # 45° 구부림
    full = build_hand_cmd(lm3, lm3)
    names = []
    for side in ["L", "R"]:
        for fi in range(1, 5):
            names += [f"{side}_AA_{fi}", f"{side}_FE_{fi}"]
    for i, (name, val) in enumerate(zip(names, full)):
        print(f"  [{i:2d}] {name:12s} = {math.degrees(val):+7.2f}°")

    print("\n✅ 검증 완료")
    print()
    print("[튜닝 가이드]")
    print(f"  ANGLE_OPEN  = {ANGLE_OPEN}°   : 이 각도 이하면 FE=0 (완전 펼침)")
    print(f"  ANGLE_CLOSE = {ANGLE_CLOSE}° : 이 각도 이상이면 FE=FE_MIN (완전 구부림)")
    print(f"  → 펼쳤는데 손이 아직 구부러져 있으면 ANGLE_OPEN ↑")
    print(f"  → 구부렸는데 FE가 너무 작으면 ANGLE_CLOSE ↓")
    print(f"  USE_AA      = {USE_AA}   : True로 바꾸면 AA 추가 (실험적)")