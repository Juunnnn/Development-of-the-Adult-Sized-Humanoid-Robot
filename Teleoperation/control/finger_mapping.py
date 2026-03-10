"""
finger_mapping.py  v3
─────────────────────────────────────────────────────────────
Quest Hand Tracking landmark → 로봇 손가락 관절각 변환 모듈.

Quest는 손의 25개 landmark(3D 좌표)를 제공함.
이 모듈은 그 좌표들로부터 로봇 손가락의 AA(벌림/모음)와 FE(굽힘/펴짐) 관절각을 계산함.

변경사항 (v2 → v3):
  - AA 계산: 캘리브 기준 상대값 → 절대값 방식으로 변경
    (손바닥 방향이 어느 쪽을 향하든 정확하게 계산됨)
  - calib_lm 파라미터 제거 (L_calib 거리값만 있으면 됨)
  - calc_fe에 clip 추가
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# 로봇 손가락 물리 파라미터
# ══════════════════════════════════════════════════════════════

# 손가락 링크 길이 [m].
# L1: 근위지절(MCP~PIP), L2: 원위지절(PIP~Tip).
L1      = 0.052
L2      = 0.039
L_TOTAL = L1 + L2   # 전체 손가락 길이 91mm. 캘리브 거리 → 로봇 거리 스케일링에 사용.

# FE_follower 관절의 mimic 비율.
# 로봇 구조상 FE 관절이 움직이면 follower 관절이 FE * 0.93으로 따라움직임 (mimic_fe_follower.py 담당).
MIMIC   = 0.93


# ══════════════════════════════════════════════════════════════
# 관절 한계
# ══════════════════════════════════════════════════════════════

AA_MIN = -0.349   # -20°  손가락 모음 한계
AA_MAX =  0.349   # +20°  손가락 벌림 한계
FE_MIN = -1.501   # -86°  최대 굽힘 한계
FE_MAX =  0.000   #   0°  완전 펼침 (굽힘 없음)


# ══════════════════════════════════════════════════════════════
# Quest landmark 인덱스
# Quest Hand Tracking API의 25개 landmark 중 필요한 것만 사용.
# ══════════════════════════════════════════════════════════════

# 각 손가락별 (MCP 인덱스, Tip 인덱스).
# FE 계산에 사용: MCP~Tip 거리가 손가락 굽힘 정도를 나타냄.
FINGER_QUEST = {
    1: (1,  4),   # 엄지  MCP=1,  Tip=4
    2: (5,  8),   # 검지  MCP=5,  Tip=8
    3: (9,  12),  # 중지  MCP=9,  Tip=12
    4: (13, 16),  # 약지  MCP=13, Tip=16
}

# 각 손가락 MCP 인덱스만 따로 정리 (AA 계산에서 인접 손가락 간 각도 계산에 사용).
MCP_IDX = {1: 1, 2: 5, 3: 9, 4: 13}


# ══════════════════════════════════════════════════════════════
# LUT (Look-Up Table): MCP~Tip 거리 ↔ FE 각도 변환
# ══════════════════════════════════════════════════════════════

def _tip_dist(fe: float) -> float:
    """
    FE 관절각 → MCP~Tip 거리 [m] 순방향 계산 (기구학).

    FE 관절이 굽혀지면(음수 방향) Tip이 MCP에 가까워짐.
    FE_follower도 함께 움직이므로 (fe + fe*MIMIC) 형태로 두 링크 끝점을 계산함.
    """
    ff = fe * MIMIC    # follower 관절이 따라 움직이는 양
    x  = L1 * np.cos(fe) + L2 * np.cos(fe + ff)
    y  = L1 * np.sin(fe) + L2 * np.sin(fe + ff)
    return float(np.sqrt(x * x + y * y))

def _build_lut(n: int = 400):
    """
    FE_MIN~FE_MAX 범위를 400등분해서 (거리, FE) 쌍 배열을 만듦.
    이걸 미리 만들어두면 매 프레임 역방향 계산을 빠르게 할 수 있음(interpolation).
    """
    fe_arr   = np.linspace(FE_MIN, FE_MAX, n)
    dist_arr = np.array([_tip_dist(fe) for fe in fe_arr])
    return dist_arr, fe_arr

# 모듈 로드 시 한 번만 계산해서 전역에 저장
_LUT_DIST, _LUT_FE = _build_lut()

def dist_to_fe(d: float) -> float:
    """
    MCP~Tip 거리 → FE 각도 [rad] 역방향 계산.
    LUT 범위를 벗어난 값은 클리핑 후 보간.
    """
    d_c = np.clip(d, _LUT_DIST[0], _LUT_DIST[-1])
    return float(np.interp(d_c, _LUT_DIST, _LUT_FE))


# ══════════════════════════════════════════════════════════════
# 손가락 EMA 필터
# ══════════════════════════════════════════════════════════════

class FingerEMAFilter:
    """
    손가락 16개 관절각 전용 EMA 필터.
    motion_utils.py의 EMAFilter와 동일한 구조지만 손가락 전용으로 분리됨.

    alpha=0.4: 팔 필터(0.6)보다 낮게 설정 → 손가락 떨림이 심하므로 더 강하게 스무딩.
    n=16: 양손 손가락 관절 총 16개 [L×8, R×8]
    """

    def __init__(self, alpha: float = 0.4, n: int = 16):
        self.alpha = alpha
        self.n     = n
        self.prev  = None

    def filter(self, cmd: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = cmd.copy()
            return cmd.copy()
        self.prev = self.alpha * cmd + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self, cmd: np.ndarray = None):
        """None을 넘기면 완전 초기화, 값을 넘기면 그 값으로 초기화."""
        self.prev = None if cmd is None else cmd.copy()


# ══════════════════════════════════════════════════════════════
# 손바닥 법선 벡터 계산
# ══════════════════════════════════════════════════════════════

def _palm_normal(lm: np.ndarray) -> np.ndarray:
    """
    손목(0), 검지MCP(5), 약지MCP(13) 세 점으로 손바닥 평면의 법선 벡터를 계산합니다.

    왜 필요한가?
    ────────────
    AA(손가락 벌림 각도) 계산 시 손바닥 평면을 기준으로 해야 함.
    Quest 좌표계 기준이 아니라 현재 손 자세 기준으로 계산해야
    손이 어느 방향을 향하든 정확하게 벌림 각도가 나옴.
    """
    v1   = lm[5]  - lm[0]   # 손목 → 검지MCP
    v2   = lm[13] - lm[0]   # 손목 → 약지MCP
    n    = np.cross(v1, v2)  # 두 벡터의 외적 = 손바닥 평면의 법선
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-6 else np.array([0., 0., 1.])


# ══════════════════════════════════════════════════════════════
# FE (굽힘/펴짐) 관절각 계산
# ══════════════════════════════════════════════════════════════

def calc_fe(lm: np.ndarray, finger_idx: int, L_calib: float) -> float:
    """
    Quest landmark에서 MCP~Tip 거리를 재고, LUT로 FE 관절각을 역산합니다.

    핵심 아이디어
    ─────────────
    손가락을 구부리면 Tip이 MCP에 가까워짐 (거리 감소).
    캘리브 시 측정한 L_calib(펼쳤을 때의 MCP~Tip 거리)를 기준으로
    현재 거리의 비율을 구해 로봇 링크 길이로 스케일링한 뒤 LUT로 FE 각도를 역산함.

    Parameters
    ----------
    lm        : landmark 배열 (25, 3)
    finger_idx: 손가락 번호 (1=엄지, 2=검지, 3=중지, 4=약지)
    L_calib   : 손 펼쳤을 때 이 손가락의 MCP~Tip 거리 [m] (캘리브에서 측정)

    Returns
    -------
    FE 각도 [rad], 범위 [FE_MIN, FE_MAX]
    """
    mcp_i, tip_i = FINGER_QUEST[finger_idx]
    d_quest = float(np.linalg.norm(lm[tip_i] - lm[mcp_i]))  # 현재 MCP~Tip 거리

    if d_quest < 1e-6 or L_calib < 1e-6:
        return 0.0  # landmark 이상하면 0 반환

    # Quest 거리 비율 → 로봇 링크 거리로 스케일링
    d_robot = (d_quest / L_calib) * L_TOTAL

    return float(np.clip(dist_to_fe(d_robot), FE_MIN, FE_MAX))


# ══════════════════════════════════════════════════════════════
# AA (벌림/모음) 관절각 계산
# ══════════════════════════════════════════════════════════════

def calc_aa(lm: np.ndarray, finger_idx: int, is_left: bool) -> float:
    """
    인접 손가락 MCP 간 절대 각도를 계산해 AA 관절각으로 변환합니다.

    핵심 아이디어 (v3 절대값 방식)
    ────────────────────────────────
    손가락 벌어짐 = 인접 MCP들 사이의 벡터가 손 길이 방향(손목→검지MCP)과
    이루는 각도로 표현할 수 있음.

    계산 순서:
      1. 손바닥 평면 법선(normal) 계산
      2. 손 길이 방향 벡터(ref_proj)를 손바닥 평면에 투영
      3. 인접 MCP 간 벡터(sep_proj)를 손바닥 평면에 투영
      4. ref_proj에 수직인 가로 방향(perp)에 sep_proj를 투영 → 벌어진 정도
      5. arcsin으로 각도 계산

    캘리브 자세와 무관 → 손이 어느 방향을 향하든 정확함 (v2 대비 개선점).

    손가락별 인접 쌍 및 부호 보정:
      손가락1(엄지): 검지MCP와 엄지MCP 간 각도, 부호 반전
      손가락2(검지): 검지MCP와 중지MCP 간 각도
      손가락3(중지): 중지MCP와 약지MCP 간 각도
      손가락4(약지): 중지MCP와 약지MCP 간 각도, 부호 반전
      오른손: 전체 부호 반전 (거울 대칭)
    """
    # 각 손가락이 참조하는 인접 MCP 쌍 (a_f→b_f 방향 벡터 기준)
    AA_PAIRS = {
        1: (2, 1),   # 검지→엄지 방향 (엄지가 검지로부터 벌어지는 방향)
        2: (2, 3),   # 검지→중지 방향
        3: (3, 4),   # 중지→약지 방향
        4: (3, 4),   # 중지→약지 방향 (약지는 반대 부호로 보정)
    }
    a_f, b_f = AA_PAIRS[finger_idx]
    a_i, b_i = MCP_IDX[a_f], MCP_IDX[b_f]

    # 손바닥 평면 법선
    normal   = _palm_normal(lm)

    # 손 길이 방향 기준 벡터: 손목(0) → 검지MCP(5)를 손바닥 평면에 투영
    ref_vec  = lm[MCP_IDX[2]] - lm[0]
    ref_proj = ref_vec - np.dot(ref_vec, normal) * normal
    ref_norm = np.linalg.norm(ref_proj)
    if ref_norm < 1e-6:
        return 0.0
    ref_proj /= ref_norm

    # 인접 MCP 간 벡터를 손바닥 평면에 투영
    sep_vec  = lm[b_i] - lm[a_i]
    sep_proj = sep_vec - np.dot(sep_vec, normal) * normal
    sep_norm = np.linalg.norm(sep_proj)
    if sep_norm < 1e-6:
        return 0.0
    sep_proj /= sep_norm

    # ref_proj에 수직인 가로 방향 (손가락 벌림 방향)
    perp   = np.cross(normal, ref_proj)

    # sep_proj의 perp 성분 = 손가락이 가로 방향으로 벌어진 정도
    spread = np.dot(sep_proj, perp)
    angle  = np.arcsin(np.clip(spread, -1.0, 1.0))

    # 손가락별/손 방향별 부호 보정
    if finger_idx == 1:   # 엄지: 인접 쌍이 반대 방향이라 부호 반전
        angle = -angle
    if finger_idx == 4:   # 약지: 중지와 약지 쌍을 공유하지만 반대 방향
        angle = -angle
    if not is_left:        # 오른손: 좌우 대칭이라 전체 부호 반전
        angle = -angle

    return float(np.clip(angle, AA_MIN, AA_MAX))


# ══════════════════════════════════════════════════════════════
# 한 손 전체 landmark → 8개 관절각
# ══════════════════════════════════════════════════════════════

def landmarks_to_finger_cmd(lm: np.ndarray, is_left: bool,
                             L_calib: np.ndarray) -> np.ndarray:
    """
    한 손의 landmark → [AA_1, FE_1, AA_2, FE_2, AA_3, FE_3, AA_4, FE_4] 8개 관절각.

    4개 손가락 각각에 대해 AA, FE를 계산해서 순서대로 배열함.
    (손가락1=엄지, 2=검지, 3=중지, 4=약지)

    Parameters
    ----------
    lm      : landmark 배열 (25, 3)
    is_left : True=왼손, False=오른손 (AA 부호 보정에 사용)
    L_calib : shape (4,) 각 손가락의 캘리브 MCP~Tip 거리 [m]
    """
    cmd = np.zeros(8)
    for f in range(1, 5):
        cmd[(f - 1) * 2]     = calc_aa(lm, f, is_left)
        cmd[(f - 1) * 2 + 1] = calc_fe(lm, f, L_calib[f - 1])
    return cmd


def build_hand_cmd(left_lm: np.ndarray, right_lm: np.ndarray,
                   L_calib_left: np.ndarray,
                   L_calib_right: np.ndarray) -> np.ndarray:
    """
    양손 landmark → 16개 관절각 배열.

    반환 형식: [L_AA_1, L_FE_1, ..., L_AA_4, L_FE_4,
                R_AA_1, R_FE_1, ..., R_AA_4, R_FE_4]

    teleop_ik.py의 TELEOP 상태에서 매 프레임 호출됨.
    결과는 finger_filter(EMA)를 거친 뒤 pub_hand로 publish됨.
    """
    l = landmarks_to_finger_cmd(left_lm,  True,  L_calib_left)
    r = landmarks_to_finger_cmd(right_lm, False, L_calib_right)
    return np.concatenate([l, r])


# ══════════════════════════════════════════════════════════════
# 캘리브레이션
# ══════════════════════════════════════════════════════════════

def compute_finger_calib(lm: np.ndarray) -> np.ndarray:
    """
    손 펼친 상태의 landmark에서 각 손가락 MCP~Tip 거리를 측정합니다.

    이 값(L_calib)이 FE 계산의 기준이 됨.
    같은 손가락을 구부렸을 때 현재 거리 / L_calib = 굽힘 비율.

    CALIBRATING_FINGERS 상태에서 50프레임 평균 landmark로 호출됨.

    Returns
    -------
    L_calib : shape (4,) [엄지, 검지, 중지, 약지] MCP~Tip 거리 [m]
    """
    L = np.zeros(4)
    for f, (mcp_i, tip_i) in FINGER_QUEST.items():
        L[f - 1] = np.linalg.norm(lm[tip_i] - lm[mcp_i])
    return L


# ══════════════════════════════════════════════════════════════
# 유효성 검사
# ══════════════════════════════════════════════════════════════

def is_landmark_valid(lm: np.ndarray) -> bool:
    """
    landmark 배열이 사용 가능한 상태인지 확인합니다.

    Quest Hand Tracking이 손을 못 잡으면 전부 0 또는 NaN으로 채운 배열을 반환함.
    이걸 그대로 계산에 쓰면 이상한 관절각이 나오므로 먼저 검사함.

    False를 반환하는 경우:
      - lm이 None
      - NaN 포함
      - 모든 값이 0에 가까움 (트래킹 미감지 상태)
    """
    if lm is None:
        return False
    if np.any(np.isnan(lm)):
        return False
    if np.allclose(lm, 0, atol=1e-6):
        return False
    return True


# ══════════════════════════════════════════════════════════════
# 단독 테스트
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=== LUT 검증: FE → 거리 → FE 역산 오차 확인 ===')
    for fe_deg in [0, -20, -45, -60, -86]:
        fe   = np.radians(fe_deg)
        d    = _tip_dist(fe)
        fe_r = dist_to_fe(d)
        print(f'  FE={fe_deg:4d}°  dist={d*1000:.2f}mm  역산={np.degrees(fe_r):.2f}°')

    print('\n=== AA 절대값 계산 테스트: 나란히 자세 → 0°에 가까워야 ===')
    lm = np.zeros((25, 3))
    lm[0]  = [0, 0, 0]       # 손목
    lm[5]  = [0, 0.08, 0]    # 검지MCP (Y방향 = 손 길이 방향)
    lm[13] = [0, 0.07, 0]    # 약지MCP
    for f, (mcp_i, _) in FINGER_QUEST.items():
        lm[mcp_i] = [0, 0.07, 0]   # 모든 MCP를 나란히 배치
    for f in range(1, 5):
        aa = calc_aa(lm, f, is_left=True)
        print(f'  손가락{f} AA={np.degrees(aa):.1f}°')
    print('✅ 완료')
