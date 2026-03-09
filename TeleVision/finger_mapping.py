"""
finger_mapping.py  v3
─────────────────────────────────────────────────────────────
변경사항 (v2 → v3):
  - AA 계산을 캘리브 기준 상대값 → 절대값으로 변경
    (손가락 사이 각도를 직접 계산 → 손바닥 방향 무관)
  - calib_lm 파라미터 제거 (L_calib만 필요)
  - calc_fe에 명시적 clip 추가
"""

import numpy as np

# ── 로봇 손가락 물리 파라미터 ──────────────────────────────
L1      = 0.052
L2      = 0.039
L_TOTAL = L1 + L2   # 91mm
MIMIC   = 0.93

# ── 관절 한계 ──────────────────────────────────────────────
AA_MIN = -0.349   # -20°
AA_MAX =  0.349   # +20°
FE_MIN = -1.501   # -86°
FE_MAX =  0.000   #   0°

# ── Quest landmark 인덱스 (MCP, Tip) ──────────────────────
# 필요하면 여기서 3↔4 교체
FINGER_QUEST = {
    1: (1,  4),   # 엄지  MCP=1,  Tip=4
    2: (5,  8),   # 검지  MCP=5,  Tip=8
    3: (9,  12),  # 중지  MCP=9,  Tip=12
    4: (13, 16),  # 약지  MCP=13, Tip=16
}
MCP_IDX = {1: 1, 2: 5, 3: 9, 4: 13}

# ── LUT ───────────────────────────────────────────────────
def _tip_dist(fe: float) -> float:
    ff = fe * MIMIC
    x  = L1 * np.cos(fe) + L2 * np.cos(fe + ff)
    y  = L1 * np.sin(fe) + L2 * np.sin(fe + ff)
    return float(np.sqrt(x*x + y*y))

def _build_lut(n=400):
    fe_arr   = np.linspace(FE_MIN, FE_MAX, n)
    dist_arr = np.array([_tip_dist(fe) for fe in fe_arr])
    return dist_arr, fe_arr

_LUT_DIST, _LUT_FE = _build_lut()

def dist_to_fe(d: float) -> float:
    d_c = np.clip(d, _LUT_DIST[0], _LUT_DIST[-1])
    return float(np.interp(d_c, _LUT_DIST, _LUT_FE))


# ── EMA 필터 ──────────────────────────────────────────────
class FingerEMAFilter:
    def __init__(self, alpha=0.4, n=16):
        self.alpha = alpha
        self.n     = n
        self.prev  = None

    def filter(self, cmd: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = cmd.copy()
            return cmd.copy()
        self.prev = self.alpha * cmd + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self, cmd=None):
        self.prev = None if cmd is None else cmd.copy()


# ── 손바닥 법선 (매 프레임 현재 landmark로 계산) ───────────
def _palm_normal(lm: np.ndarray) -> np.ndarray:
    """
    손목(0), 검지MCP(5), 약지MCP(13)으로 손바닥 평면 법선.
    캘리브와 무관하게 현재 자세 기준.
    """
    v1 = lm[5]  - lm[0]
    v2 = lm[13] - lm[0]
    n  = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-6 else np.array([0., 0., 1.])


# ── FE 계산 ────────────────────────────────────────────────
def calc_fe(lm: np.ndarray, finger_idx: int, L_calib: float) -> float:
    """
    MCP~Tip 거리 → FE 각도 (rad)
    L_calib: 캘리브 시 MCP~Tip 거리 (m)
    """
    mcp_i, tip_i = FINGER_QUEST[finger_idx]
    d_quest = float(np.linalg.norm(lm[tip_i] - lm[mcp_i]))
    if d_quest < 1e-6 or L_calib < 1e-6:
        return 0.0
    d_robot = (d_quest / L_calib) * L_TOTAL
    fe = dist_to_fe(d_robot)
    return float(np.clip(fe, FE_MIN, FE_MAX))


# ── AA 계산 (절대값 방식, 캘리브 무관) ─────────────────────
def calc_aa(lm: np.ndarray, finger_idx: int, is_left: bool) -> float:
    """
    인접 손가락 MCP 간 절대 각도 → AA 관절각
    
    손바닥 평면에 투영한 인접 MCP 벡터의 각도를 직접 계산.
    캘리브 자세와 무관 → 손바닥 방향이 달라도 정확.

    중립(손가락 나란히) = 0°
    벌어짐 = 양수 또는 음수
    """
    # 인접 MCP 쌍 (a→b 벡터 기준)
    AA_PAIRS = {
        1: (2, 1),   # 검지→엄지 (엄지가 벌어지는 방향)
        2: (2, 3),   # 검지→중지
        3: (3, 4),   # 중지→약지
        4: (3, 4),   # 중지→약지 (약지는 반대 부호)
    }
    a_f, b_f = AA_PAIRS[finger_idx]
    a_i, b_i = MCP_IDX[a_f], MCP_IDX[b_f]

    normal = _palm_normal(lm)

    # 손바닥 기준 방향: 손목→검지MCP (손 길이 방향)
    ref_vec = lm[MCP_IDX[2]] - lm[0]
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

    # ref_proj에 수직인 방향 (손가락 벌림 방향)
    perp = np.cross(normal, ref_proj)  # 손바닥 평면 내 가로 방향

    # sep_proj의 perp 성분 = 벌어진 정도
    spread = np.dot(sep_proj, perp)
    # arcsin으로 각도 계산 (작은 각도에서 선형적)
    angle = np.arcsin(np.clip(spread, -1.0, 1.0))

    # 손가락별 부호 보정
    if finger_idx == 1:   # 엄지: 반대 방향
        angle = -angle
    if finger_idx == 4:   # 약지: 반대 부호
        angle = -angle
    if not is_left:       # 오른손 대칭
        angle = -angle

    return float(np.clip(angle, AA_MIN, AA_MAX))


# ── 한 손 전체 ─────────────────────────────────────────────
def landmarks_to_finger_cmd(lm: np.ndarray, is_left: bool,
                             L_calib: np.ndarray) -> np.ndarray:
    """
    → [AA_1, FE_1, AA_2, FE_2, AA_3, FE_3, AA_4, FE_4]
    L_calib: shape (4,) 각 손가락 캘리브 MCP~Tip 거리
    calib_lm 파라미터 제거됨 (v3)
    """
    cmd = np.zeros(8)
    for f in range(1, 5):
        cmd[(f-1)*2]   = calc_aa(lm, f, is_left)
        cmd[(f-1)*2+1] = calc_fe(lm, f, L_calib[f-1])
    return cmd


def build_hand_cmd(left_lm, right_lm,
                   L_calib_left, L_calib_right) -> np.ndarray:
    """양손 → 16개 관절각 [L×8, R×8]"""
    l = landmarks_to_finger_cmd(left_lm,  True,  L_calib_left)
    r = landmarks_to_finger_cmd(right_lm, False, L_calib_right)
    return np.concatenate([l, r])


# ── 캘리브레이션 ───────────────────────────────────────────
def compute_finger_calib(lm: np.ndarray) -> np.ndarray:
    """
    손 펼친 상태 landmark → 각 손가락 MCP~Tip 거리 (4,)
    L_calib만 저장하면 됨. calib_lm은 불필요 (v3).
    """
    L = np.zeros(4)
    for f, (mcp_i, tip_i) in FINGER_QUEST.items():
        L[f-1] = np.linalg.norm(lm[tip_i] - lm[mcp_i])
    return L


# ── 유효성 검사 ────────────────────────────────────────────
def is_landmark_valid(lm: np.ndarray) -> bool:
    if lm is None:                     return False
    if np.any(np.isnan(lm)):           return False
    if np.allclose(lm, 0, atol=1e-6):  return False
    return True


if __name__ == '__main__':
    print('=== LUT 검증 ===')
    for fe_deg in [0, -20, -45, -60, -86]:
        fe   = np.radians(fe_deg)
        d    = _tip_dist(fe)
        fe_r = dist_to_fe(d)
        print(f'  FE={fe_deg:4d}°  dist={d*1000:.2f}mm  역산={np.degrees(fe_r):.2f}°')

    print()
    print('=== AA 절대값 계산 테스트 ===')
    lm = np.zeros((25, 3))
    lm[0]  = [0, 0, 0]       # 손목
    lm[5]  = [0, 0.08, 0]    # 검지MCP (손 길이 방향 = Y)
    lm[13] = [0, 0.07, 0]    # 약지MCP
    # 손바닥 법선 = Z축
    # 손가락 나란히 (AA=0)
    for f, (mcp_i, _) in FINGER_QUEST.items():
        lm[mcp_i] = [0, 0.07, 0]  # 모두 Y방향으로 나란히
    print('  나란히 자세:')
    for f in range(1, 5):
        aa = calc_aa(lm, f, is_left=True)
        print(f'    손가락{f} AA={np.degrees(aa):.1f}° (0에 가까워야)')
    print('✅ 완료')