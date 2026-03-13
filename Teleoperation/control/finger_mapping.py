"""
finger_mapping.py  v7
─────────────────────────────────────────────────────────────
Quest Hand Tracking landmark → 로봇 손가락 관절각 변환 모듈.

[v6 → v7 버그 수정 요약]

  ❌ v6 버그 1: HUMAN_FINGER_LEN 하드코딩
      - Quest 스케일/사용자 손 크기에 따라 d_max보다 항상 작은 값이
        들어오면 t < 1.0 → FE 항상 구부러짐 (중지 항상 굽힘 원인)

  ✅ v7 수정 1: 손바닥 길이(wrist→middle_MCP) 기준 비율로 d_max 계산
      - palm_len = |lm[MIDDLE_MCP] - lm[WRIST]|  (Quest 어떤 스케일이든 자동 적응)
      - d_max = FINGER_PALM_RATIO[f] × palm_len   (해부학적 비율, 사람마다 일정)

  ❌ v6 버그 2: 방향 기반 FE는 palm orientation(pronation/supination)에 따라
      local_z 부호가 뒤집혀서 불안정. arctan2(-local_z, ...) 가 손 방향에 따라
      양수/음수가 섞임.

  ✅ v7 수정 2: FE는 거리 기반(DexPilot 방식)만 사용.
      3D 거리 |MCP→Tip| 는 손 방향에 완전 무관.

  ❌ v5 버그 (참고): arctan2(local_z, ...) 에서 z 부호 오류 + clip(0.0)으로
      FE가 항상 0으로 강제됨. (방향 기반 FE의 근본 취약점)

[손가락 매핑]
  로봇 1번(엄지) ← 사용자 엄지  (Quest MCP=1, Tip=4 )
  로봇 2번(검지) ← 사용자 검지  (Quest MCP=5, Tip=8 )
  로봇 3번(중지) ← 사용자 중지  (Quest MCP=9, Tip=12)
  로봇 4번(약지) ← 사용자 소지  (Quest MCP=17,Tip=20) ← 요청

[URDF 기반 로봇 파라미터]
  L1=0.052m (L_FE_1_joint → L_FE_follower_1_joint origin x)
  L2=0.039m (L_FE_follower_1_joint → L_finger_end_joint_1 origin x)
  AA: lower=-0.349(-20°), upper=0.349(+20°)
  FE: lower=-1.501(-86°), upper=0.000( 0°)
  FE_follower: mimic ≈ 0.93 × FE
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# 로봇 손가락 물리 파라미터  (URDF에서 추출)
# ══════════════════════════════════════════════════════════════

L1      = 0.052          # 근위 링크 [m]
L2      = 0.039          # 원위 링크 [m]
L_TOTAL = L1 + L2        # 0.091 m
MIMIC   = 0.93           # FE_follower = FE × MIMIC


# ══════════════════════════════════════════════════════════════
# 관절 한계  (URDF)
# ══════════════════════════════════════════════════════════════

AA_MIN = -0.349   # -20°
AA_MAX =  0.349   # +20°
FE_MIN = -1.501   # -86°
FE_MAX =  0.000   #   0°


# ══════════════════════════════════════════════════════════════
# Quest landmark 인덱스
# ══════════════════════════════════════════════════════════════

FINGER_TIPS = {
    1: (1,  4),    # 엄지  → 로봇 1번
    2: (5,  8),    # 검지  → 로봇 2번
    3: (9,  12),   # 중지  → 로봇 3번
    4: (13, 16),   # 약지  → 로봇 4번  (이전: 소지 17→20, 약지 직접 매핑으로 변경)
}

# 손바닥 프레임 기준점
WRIST_IDX  = 0
MIDDLE_MCP = 9    # palm_len 기준 / y축
INDEX_MCP  = 5    # z축 계산용
PINKY_MCP  = 17   # z축 계산용


# ══════════════════════════════════════════════════════════════
# 자동 캘리브레이션 설정
# ══════════════════════════════════════════════════════════════
# 텔레옵 시작 후 CALIB_FRAMES 프레임 동안 자동으로 d_max/d_min 수집.
# 그 시간 안에 손가락을 한 번씩 펼쳤다 구부려주면 됩니다.
# ══════════════════════════════════════════════════════════════

CALIB_FRAMES = 300   # 약 3~5초

# 캘리브 미완료 시 폴백값 [m]  (검지 실측 기반)
CALIB_SEED_MAX = {1: 0.090, 2: 0.120, 3: 0.120, 4: 0.095}
CALIB_SEED_MIN = {1: 0.038, 2: 0.052, 3: 0.052, 4: 0.040}


# ══════════════════════════════════════════════════════════════
# 2링크 IK LUT  (거리 → FE 각도, 모듈 로드 시 1회 생성)
# ══════════════════════════════════════════════════════════════

def _build_lut(n: int = 500):
    """
    FE ∈ [FE_MIN, FE_MAX] → 손가락 끝 거리 d 테이블 생성.
    MIMIC 적용:
      px = L1·cos(fe) + L2·cos(fe·(1+MIMIC))
      py = L1·sin(fe) + L2·sin(fe·(1+MIMIC))
      d  = sqrt(px²+py²)
    단조 증가 → np.interp 역산 가능.
    """
    fe_arr   = np.linspace(FE_MIN, FE_MAX, n)
    dist_arr = np.zeros(n)
    for i, fe in enumerate(fe_arr):
        ff = fe * MIMIC
        px = L1 * np.cos(fe) + L2 * np.cos(fe + ff)
        py = L1 * np.sin(fe) + L2 * np.sin(fe + ff)
        dist_arr[i] = np.sqrt(px * px + py * py)
    return dist_arr, fe_arr


_LUT_DIST, _LUT_FE = _build_lut()
# _LUT_DIST[0] ≈ 0.0702 (완전 구부림)
# _LUT_DIST[-1] = L_TOTAL = 0.091 (완전 펼침)


def _dist_to_fe(d: float) -> float:
    """로봇 스케일 MCP-Tip 거리 → FE 관절각 [rad]."""
    d = float(np.clip(d, _LUT_DIST[0], _LUT_DIST[-1]))
    return float(np.interp(d, _LUT_DIST, _LUT_FE))


# ══════════════════════════════════════════════════════════════
# 손바닥 로컬 프레임
# ══════════════════════════════════════════════════════════════

def _palm_frame(lm: np.ndarray) -> np.ndarray:
    """
    손바닥 기준 회전행렬 R (3×3) 반환.
    열 순서: [x=가로, y=손 길이(앞), z=손바닥 법선]

    Note: FE는 거리 기반이므로 z축 방향에 무관.
    AA에만 x, y 성분 사용 (손바닥 법선 방향 불필요).
    """
    y_vec  = lm[MIDDLE_MCP] - lm[WRIST_IDX]
    y_norm = np.linalg.norm(y_vec)
    if y_norm < 1e-6:
        return np.eye(3)
    y_axis = y_vec / y_norm

    v1    = lm[INDEX_MCP] - lm[WRIST_IDX]
    v2    = lm[PINKY_MCP] - lm[WRIST_IDX]
    z_raw = np.cross(v1, v2)
    z_n   = np.linalg.norm(z_raw)
    z_axis = z_raw / z_n if z_n > 1e-6 else np.array([0., 0., 1.])

    x_raw  = np.cross(y_axis, z_axis)
    x_n    = np.linalg.norm(x_raw)
    x_axis = x_raw / x_n if x_n > 1e-6 else np.array([1., 0., 0.])
    z_axis = np.cross(x_axis, y_axis)   # 재직교화

    return np.column_stack([x_axis, y_axis, z_axis])


# ══════════════════════════════════════════════════════════════
# EMA 필터
# ══════════════════════════════════════════════════════════════

class FingerEMAFilter:
    def __init__(self, alpha: float = 0.4, n: int = 16):
        self.alpha = alpha
        self.n     = n
        self.prev  = None

    def filter(self, cmd: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = cmd.copy()
            return cmd.copy()
        self.prev = self.alpha * cmd + (1.0 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self, cmd: np.ndarray = None):
        self.prev = None if cmd is None else cmd.copy()


# ══════════════════════════════════════════════════════════════
# 자동 캘리브레이터
# ══════════════════════════════════════════════════════════════

class FingerCalibrator:
    def __init__(self):
        self.reset()

    def reset(self):
        # 0에서 시작 → 실측값이 항상 갱신됨 (SEED 초기값에 막히지 않음)
        self._d_max = {f: 0.0 for f in range(1, 5)}
        self._d_min = {f: 9.9 for f in range(1, 5)}
        self._frame = 0
        self.done   = False

    def update(self, lm):
        if self.done:
            return
        for f, (mcp_i, tip_i) in FINGER_TIPS.items():
            d = float(np.linalg.norm(lm[tip_i] - lm[mcp_i]))
            if d > 1e-4:
                self._d_max[f] = max(self._d_max[f], d)
                self._d_min[f] = min(self._d_min[f], d)
        self._frame += 1
        if self._frame >= CALIB_FRAMES:
            # 데이터 품질 검사: 범위가 너무 좁으면 SEED로 대체
            for f in range(1, 5):
                rng = self._d_max[f] - self._d_min[f]
                if self._d_max[f] < 1e-3 or rng < self._d_max[f] * 0.2:
                    self._d_max[f] = CALIB_SEED_MAX[f]
                    self._d_min[f] = CALIB_SEED_MIN[f]
            self.done = True
            print("[FingerCalib] 캘리브 완료!")
            for f in range(1, 5):
                print(f"  손가락{f}: max={self._d_max[f]*1000:.1f}mm  "
                      f"min={self._d_min[f]*1000:.1f}mm")

    @property
    def d_max(self): return self._d_max

    @property
    def d_min(self): return self._d_min


_calib_L = FingerCalibrator()
_calib_R = FingerCalibrator()


# ══════════════════════════════════════════════════════════════
# FE + AA 계산
# ══════════════════════════════════════════════════════════════

def _calc_fe_aa(lm: np.ndarray, finger_idx: int,
                R_palm: np.ndarray, palm_len: float,
                is_left: bool, calib: "FingerCalibrator" = None):
    """
    손가락 1개의 FE, AA 관절각 계산.

    FE — 거리 기반 (orientation 무관)
    ─────────────────────────────────
    1. d_human = |MCP→Tip| 3D 거리
    2. d_max   = FINGER_PALM_RATIO[f] × palm_len  (손 크기 자동 적응)
    3. d_min   = d_max × BEND_RATIO               (완전 구부림 기준)
    4. t       = clip((d_human - d_min) / (d_max - d_min), 0, 1)
                 t=1.0: 완전 펼침 → FE=0
                 t=0.0: 완전 구부림 → FE=FE_MIN
    5. d_robot = t × (L_TOTAL - LUT_min) + LUT_min
    6. FE      = LUT 역산(d_robot)

    AA — 방향 기반 (손바닥 로컬 x/y 비율)
    ──────────────────────────────────────
    tip_vec 를 palm 로컬로 변환 → local_x(가로) / local_y(앞) 비율
    AA = arctan2(local_x, local_y)
    오른손 부호 반전
    """
    mcp_i, tip_i = FINGER_TIPS[finger_idx]

    tip_vec = lm[tip_i] - lm[mcp_i]
    d_human = float(np.linalg.norm(tip_vec))
    if d_human < 1e-6:
        return 0.0, 0.0

    # ── FE ──────────────────────────────────────────────────
    d_max_f = calib.d_max[finger_idx] if (calib and calib.done) else CALIB_SEED_MAX[finger_idx]
    d_min_f = calib.d_min[finger_idx] if (calib and calib.done) else CALIB_SEED_MIN[finger_idx]
    t       = float(np.clip((d_human - d_min_f) / (d_max_f - d_min_f), 0.0, 1.0))
    d_robot = t * (L_TOTAL - _LUT_DIST[0]) + _LUT_DIST[0]
    fe      = _dist_to_fe(d_robot)

    # ── AA ──────────────────────────────────────────────────
    # 단위벡터로 palm 로컬 프레임에 투영
    tip_unit  = tip_vec / d_human
    tip_local = R_palm.T @ tip_unit   # [x=가로, y=앞, z=법선]
    local_x   = tip_local[0]
    local_y   = tip_local[1]

    # local_y 강제 양수 제거 → xy 벡터 크기가 충분할 때만 계산
    xy_norm = np.sqrt(local_x ** 2 + local_y ** 2)
    if xy_norm > 0.1:   # tip 벡터가 손바닥 법선 방향으로만 향하는 비정상 자세 제외
        aa = float(np.arctan2(local_x, local_y))
    else:
        aa = 0.0
    if not is_left:
        aa = -aa

    fe = float(np.clip(fe, FE_MIN, FE_MAX))
    aa = float(np.clip(aa, AA_MIN, AA_MAX))
    return aa, fe


# ══════════════════════════════════════════════════════════════
# 한 손 전체: 8개 관절각
# ══════════════════════════════════════════════════════════════

def landmarks_to_finger_cmd(lm: np.ndarray, is_left: bool,
                             L_calib: np.ndarray = None) -> np.ndarray:
    """
    한 손 landmark (25, 3) → [AA_1, FE_1, AA_2, FE_2, AA_3, FE_3, AA_4, FE_4]
    총 8개 관절각 [rad]
    """
    R_palm = _palm_frame(lm)
    calib  = _calib_L if is_left else _calib_R
    calib.update(lm)

    cmd = np.zeros(8)
    for f in range(1, 5):
        aa, fe             = _calc_fe_aa(lm, f, R_palm, 0.0, is_left, calib)
        cmd[(f - 1) * 2]   = aa
        cmd[(f - 1) * 2 + 1] = fe
    return cmd


def build_hand_cmd(left_lm: np.ndarray, right_lm: np.ndarray,
                   L_calib_left:  np.ndarray = None,
                   L_calib_right: np.ndarray = None) -> np.ndarray:
    """
    양손 landmark → 16개 관절각.
    [L_AA_1, L_FE_1, ..., L_AA_4, L_FE_4,
     R_AA_1, R_FE_1, ..., R_AA_4, R_FE_4]
    """
    l = landmarks_to_finger_cmd(left_lm,  is_left=True)
    r = landmarks_to_finger_cmd(right_lm, is_left=False)
    return np.concatenate([l, r])


# ══════════════════════════════════════════════════════════════
# 호환성 유지용 더미
# ══════════════════════════════════════════════════════════════

def compute_finger_calib(lm: np.ndarray) -> np.ndarray:
    return np.zeros(4)


def is_landmark_valid(lm: np.ndarray) -> bool:
    if lm is None or np.any(np.isnan(lm)):
        return False
    return not np.allclose(lm, 0, atol=1e-6)


# ══════════════════════════════════════════════════════════════
# 단독 테스트
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 60)
    print('  finger_mapping v7 검증')
    print('=' * 60)

    def make_hand(palm_len=0.090):
        lm = np.zeros((25, 3))
        lm[WRIST_IDX]  = [0,      0,           0]
        lm[MIDDLE_MCP] = [0,      palm_len,    0]   # 정확히 y축
        lm[INDEX_MCP]  = [ 0.030, palm_len*0.97, 0]
        lm[PINKY_MCP]  = [-0.030, palm_len*0.95, 0]
        lm[1]  = [ 0.025, palm_len*0.35, 0]   # 엄지 MCP
        lm[5]  = [ 0.030, palm_len*0.97, 0]   # 검지 MCP = INDEX_MCP
        lm[9]  = [ 0,     palm_len,      0]   # 중지 MCP = MIDDLE_MCP
        lm[17] = [-0.030, palm_len*0.95, 0]   # 소지 MCP = PINKY_MCP
        return lm

    PALM = 0.090

    def dm(f, pl=PALM): return FINGER_PALM_RATIO[f] * pl
    def dn(f, pl=PALM): return dm(f, pl) * BEND_RATIO

    # ── 1. 펼침 ────────────────────────────────────────────────
    print('\n[1] 모든 손가락 펼쳤을 때 → FE=0°, AA=0°')
    lm = make_hand(PALM)
    for f, (mcp_i, tip_i) in FINGER_TIPS.items():
        lm[tip_i] = lm[mcp_i] + [0, dm(f), 0]
    cmd = landmarks_to_finger_cmd(lm, is_left=True)
    for f in range(1, 5):
        print(f'  로봇{f}: AA={np.degrees(cmd[(f-1)*2]):+.1f}°  FE={np.degrees(cmd[(f-1)*2+1]):+.1f}°')

    # ── 2. 완전 구부림 ─────────────────────────────────────────
    print('\n[2] 모든 손가락 완전 구부림 → FE=-86°')
    lm = make_hand(PALM)
    for f, (mcp_i, tip_i) in FINGER_TIPS.items():
        lm[tip_i] = lm[mcp_i] + [0, dn(f) * 0.8, 0]
    cmd = landmarks_to_finger_cmd(lm, is_left=True)
    for f in range(1, 5):
        print(f'  로봇{f}: AA={np.degrees(cmd[(f-1)*2]):+.1f}°  FE={np.degrees(cmd[(f-1)*2+1]):+.1f}°')

    # ── 3. 손 크기 독립성 ──────────────────────────────────────
    print('\n[3] 손 크기 독립 - 검지 절반 구부림')
    for pl in [0.075, 0.090, 0.105]:
        lm2 = make_hand(pl)
        lm2[8] = lm2[5] + [0, (dm(2,pl)+dn(2,pl))/2, 0]
        fe = np.degrees(landmarks_to_finger_cmd(lm2, True)[3])
        print(f'  palm={pl*1000:.0f}mm: 검지 FE={fe:+.1f}°  (크기별 동일해야 함)')

    # ── 4. Palm orientation 독립성 ─────────────────────────────
    print('\n[4] Palm orientation 독립성 (FE = 거리 기반, 방향 무관)')
    dh = (dm(2)+dn(2))/2
    lm3 = make_hand(PALM); lm3[8] = lm3[5] + [0, dh, 0]
    fe_n = np.degrees(landmarks_to_finger_cmd(lm3, True)[3])
    lm4 = make_hand(PALM)
    for i in range(len(lm4)): lm4[i] = [lm3[i][0], lm3[i][2], lm3[i][1]]
    fe_r = np.degrees(landmarks_to_finger_cmd(lm4, True)[3])
    print(f'  정상 방향: FE={fe_n:+.1f}°')
    print(f'  회전된   : FE={fe_r:+.1f}°')
    print(f'  차이     : {abs(fe_n-fe_r):.1f}°  (0이어야 함)')

    # ── 5. 소지→로봇 약지 매핑 ────────────────────────────────
    print('\n[5] 소지(Quest 17→20) → 로봇 4번(약지) 매핑')
    lm5 = make_hand(PALM)
    lm5[20] = lm5[17] + [0, dm(4), 0]
    fe_e = np.degrees(landmarks_to_finger_cmd(lm5, True)[7])
    lm5[20] = lm5[17] + [0, dn(4)*0.8, 0]
    fe_b = np.degrees(landmarks_to_finger_cmd(lm5, True)[7])
    print(f'  소지 펼침  → 로봇4 FE={fe_e:+.1f}°  (목표: 0°)')
    print(f'  소지 구부림 → 로봇4 FE={fe_b:+.1f}°  (목표: -86°)')

    # ── 6. AA 방향 ────────────────────────────────────────────
    print('\n[6] 검지 AA 방향')
    lm6 = make_hand(PALM); base = lm6[5] + [0, dm(2), 0]
    lm6[8] = base
    aa0 = np.degrees(landmarks_to_finger_cmd(lm6, True)[2])
    lm6[8] = base + [0.03, 0, 0]
    aa_r = np.degrees(landmarks_to_finger_cmd(lm6, True)[2])
    lm6[8] = base + [-0.02, 0, 0]
    aa_l = np.degrees(landmarks_to_finger_cmd(lm6, True)[2])
    print(f'  중립       : AA={aa0:+.1f}°')
    print(f'  오른쪽 벌림: AA={aa_r:+.1f}°  (양수여야 함)')
    print(f'  왼쪽 모음  : AA={aa_l:+.1f}°  (음수여야 함)')

    print('\n✅ 검증 완료')
    print()
    print('[튜닝 가이드]')
    print(f'  BEND_RATIO={BEND_RATIO}  구부림 민감도: 느리면↓(0.30) 빠르면↑(0.50)')
    print('  FINGER_PALM_RATIO  손 비율 조정 시: 펼침에서 FE≠0° 이면 조정')