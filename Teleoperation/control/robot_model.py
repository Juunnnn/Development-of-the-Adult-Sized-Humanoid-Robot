"""
robot_model.py
─────────────────────────────────────────────────────────────
로봇 수학 계산을 담당합니다.

- 피노키오 모델 초기화 (URDF 로드 + 손바닥 가상 프레임 등록)
- IK (역기구학): 손바닥 목표 위치 → 관절각 계산
- 손목 yaw 추출: Quest 손목 rotation → wrist_yaw 관절각
- 좌표 변환: Quest 좌표계 → 로봇 좌표계
"""

import numpy as np
import pinocchio as pin
from pinocchio import SE3
from scipy.spatial.transform import Rotation

import config


# ══════════════════════════════════════════════════════════════
# 피노키오 모델 초기화
# ══════════════════════════════════════════════════════════════

def build_robot_model():
    """
    URDF를 읽어 피노키오 모델을 구성하고,
    손바닥 가상 프레임(L_palm, R_palm)을 추가한 뒤
    초기 자세(q_init)와 그 자세에서의 손바닥 위치를 계산해 반환합니다.

    Returns
    -------
    model       : pin.Model   피노키오 관절 모델
    data        : pin.Data    FK/IK 계산용 데이터 버퍼
    ids         : dict        자주 쓰는 인덱스 묶음 (아래 참조)
    q_init      : np.ndarray  초기 자세 q 벡터 (앞으로 나란히)
    robot_init  : dict        초기 자세에서의 손바닥 위치 {'L': ..., 'R': ...}

    ids 내용
    --------
    'L_palm'      : L_palm 프레임 인덱스 (IK 목표 프레임)
    'R_palm'      : R_palm 프레임 인덱스
    'joint_ids'   : JOINT_ORDER 순서대로 pinocchio q 벡터 내 인덱스 12개
    'L_joint_mask': IK에서 왼팔만 제어할 때 넘기는 인덱스 (어깨 3개 + 팔꿈치 1개)
    'R_joint_mask': IK에서 오른팔만 제어할 때 넘기는 인덱스
    """
    # URDF에서 관절 구조/한계 로드
    model = pin.buildModelFromUrdf(config.URDF_PATH)
    data  = model.createData()

    # 가상 손바닥 프레임 추가
    # wrist_yaw 프레임에서 z축으로 PALM_Z_OFFSET만큼 떨어진 가상 프레임을 만든다.
    # IK 목표를 손목 관절이 아니라 실제 손바닥 중심 위치 기준으로 맞추기 위함.
    palm_offset = SE3(np.eye(3), np.array([0, 0, config.PALM_Z_OFFSET]))
    for side in ('L', 'R'):
        wrist_id  = model.getFrameId(f'{side}_wrist_yaw')
        parent    = model.frames[wrist_id].parentJoint
        placement = model.frames[wrist_id].placement * palm_offset
        model.addFrame(pin.Frame(
            f'{side}_palm', parent, wrist_id, placement, pin.FrameType.OP_FRAME
        ))
    # 프레임 추가 후 data 재생성 (추가된 프레임이 반영되도록)
    data = model.createData()

    L_palm_id = model.getFrameId('L_palm')
    R_palm_id = model.getFrameId('R_palm')

    # 초기 자세(앞으로 나란히) q 벡터 구성
    # pin.neutral()은 모든 관절이 0인 q를 반환. 여기에 INIT_VALS를 덮어씀.
    q_init    = pin.neutral(model)
    joint_ids = []
    for name, val in zip(config.JOINT_ORDER, config.CALIB_POS):
        jid = model.getJointId(name)
        idx = model.joints[jid].idx_q   # pinocchio q 벡터 내 해당 관절의 인덱스
        q_init[idx] = val
        joint_ids.append(idx)

    # 초기 자세에서 손바닥 위치 계산 (순기구학 FK 실행)
    # 이 위치가 텔레옵 중 Quest 좌표→로봇 좌표 변환의 기준점이 됨.
    pin.forwardKinematics(model, data, q_init)
    pin.updateFramePlacements(model, data)
    robot_init = {
        'L': data.oMf[L_palm_id].translation.copy(),
        'R': data.oMf[R_palm_id].translation.copy(),
    }
    print(f"로봇 초기 L_palm: {robot_init['L'].round(3)}")
    print(f"로봇 초기 R_palm: {robot_init['R'].round(3)}")

    # Waist_joint(torso yaw) idx 추적
    # URDF에서 revolute로 변경된 경우 pinocchio q 벡터에 포함됨
    torso_idx = None
    if model.existJointName('Waist_joint'):
        torso_jid = model.getJointId('Waist_joint')
        torso_idx = model.joints[torso_jid].idx_q

    ids = {
        'L_palm':       L_palm_id,
        'R_palm':       R_palm_id,
        'joint_ids':    joint_ids,
        'L_joint_mask': joint_ids[0:4],   # 왼팔: L_shoulder(3개) + L_elbow(1개)
        'R_joint_mask': joint_ids[5:9],   # 오른팔: R_shoulder(3개) + R_elbow(1개)
        'torso_idx':    torso_idx,        # Waist_joint q 인덱스 (None이면 fixed)
        'L_roll_idx':   joint_ids[1],     # L_shoulder_roll q 인덱스
        'R_roll_idx':   joint_ids[6],     # R_shoulder_roll q 인덱스
    }
    return model, data, ids, q_init, robot_init


# ══════════════════════════════════════════════════════════════
# IK (역기구학)
# ══════════════════════════════════════════════════════════════

def compute_ik(model, data, frame_id, target_pos, q_cur,
               q_ref=None, q_init=None,
               target_forearm_dir=None,
               max_iter=50, eps=1e-3,
               null_weight=0.3, orient_weight=0.5, joint_mask=None,
               orient_min_iter=5, orient_eps=0.02, orient_max_delta=0.05,
               orient_limit_margin=0.15):
    """
    Position-only IK + orientation 보조 과제: 손바닥(frame_id)이 target_pos에 가도록
    관절각을 반복 계산합니다. target_forearm_dir이 주어지면, position을 절대 해치지
    않는 선(null space)에서 전완 방향도 최대한 그쪽으로 맞춥니다.

    핵심 알고리즘: DLS pseudo-inverse + null-space 제어
    ───────────────────────────────────────────────────
    - DLS(Damped Least Squares): J†= Jᵀ(JJᵀ + λI)⁻¹
      일반 역행렬 대신 λ 항을 추가해 특이점(팔이 완전히 뻗은 상태 등)에서도 안정적임.
    - Null-space 제어: (I - J†J) @ q_null
      손바닥 위치를 유지하면서 남는 자유도로 두 가지를 절충함:
        1) 자세 복원 (q_ref) - 이게 없으면 팔꿈치가 이상한 방향으로 꺾일 수 있음
        2) 전완 방향 정렬 (target_forearm_dir) - shoulder_yaw가 이 자유도의
           대부분을 차지하므로, 이 항을 넣으면 yaw가 유의미하게 움직이기 시작함
      두 항 모두 같은 1차원 null space 위에 투영되므로, orient_weight를
      null_weight보다 충분히 크게 잡으면 orientation이 우선하고
      (target_forearm_dir이 없거나 신호가 불안정할 때는) posture 복원이 fallback 역할을 함.

    안전장치 (4가지)
    ─────────────────
    1. adaptive λ: 위치 에러가 클수록 λ도 크게 → 발산 방지
    2. step size clamping: 한 번에 0.3rad 이상 이동 금지 → overshooting 방지
    3. 조기 종료 (발산): 에러가 이전보다 커지면 중단
    4. 조기 종료 (클램핑): 관절 한계에 5번 연속 막히면 도달 불가로 판단해 중단

    Parameters
    ----------
    frame_id   : IK 목표 프레임 인덱스 (L_palm_id 또는 R_palm_id)
    target_pos : 목표 손바닥 위치 [m] (3,)
    q_cur      : 현재 관절각 벡터 (IK 반복 계산의 시작점)
    q_ref      : null-space 기준 자세. 주로 트래킹 시작 시점의 실제 로봇 자세.
    q_init     : q_ref가 None일 때의 fallback 자세
    target_forearm_dir : 목표 전완 방향 단위벡터 (로봇 world 좌표계, (3,)).
                 None이면 orientation 항은 완전히 비활성화되고 기존 동작과 100% 동일.
    orient_weight : orientation 정렬 gain. null_weight보다 충분히 크게 잡아야
                 orientation이 posture 복원보다 우선함 (권장: null_weight의 2~5배).
    orient_min_iter : target_forearm_dir이 있을 때, position이 eps 이내로
                 이미 수렴해도 최소 이만큼은 더 반복함. 50Hz 연속 추종 중에는
                 위치 오차가 첫 iteration부터 eps 밑으로 나오는 경우가 많아서,
                 이게 없으면 orientation 보정이 실행될 기회 자체를 못 얻음.
    orient_eps  : orientation 정렬 오차(rad, |cross product| 기준)가 이 이하면
                 orient_min_iter를 다 안 채워도 조기 종료.
    orient_max_delta : orientation 보정이 한 iteration에 낼 수 있는 최대 관절
                 변화량(rad). 전체 dq의 0.3rad 클램프와 별개로, orientation
                 항만 따로 작게 제한함 - 정렬오차가 클 때(예: 90°) 한 번에
                 확 튀는 걸 막아서 위치 정확도가 흔들리는 걸 방지함.
    orient_limit_margin : 관절이 자기 하드리밋으로부터 이 거리(rad) 이내로
                 들어오면, orientation 스텝 중 "그 방향으로 더 미는" 성분을
                 부드럽게 0까지 감쇠시킴. 이게 없으면 매 프레임 조금씩 계속
                 같은 방향으로 밀리다 결국 하드리밋에 눌러붙는 문제가 생김.
    joint_mask : 제어할 관절 인덱스 리스트.
                 왼팔 IK면 L_joint_mask를 넘겨서 오른팔 관절은 건드리지 않음.
    """
    q = q_cur.copy()
    if q_ref is None:
        q_ref = q_init.copy() if q_init is not None else q_cur.copy()

    prev_err    = np.inf
    clamp_count = 0
    CLAMP_LIMIT = 5  # 관절 한계에 이 횟수 이상 막히면 도달 불가로 판단

    for i in range(max_iter):
        # 현재 q에서 FK 실행 → 손바닥 현재 위치 계산
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pos_error = target_pos - data.oMf[frame_id].translation
        err_norm  = np.linalg.norm(pos_error)

        # orientation 오차를 미리 계산 (수렴판단 + 아래 dq_orient에서 재사용, 중복계산 없음)
        e_orient    = None
        orient_done = True
        if target_forearm_dir is not None:
            d_current   = data.oMf[frame_id].rotation[:, 2]
            e_orient    = np.cross(target_forearm_dir, d_current)  # 정렬되면 0벡터
            orient_done = np.linalg.norm(e_orient) < orient_eps

        # 수렴 조건: 위치 오차가 eps(1mm) 이하 이고,
        #           (orientation 타겟이 없거나) orientation도 충분히 정렬됐거나 orient_min_iter를 채웠으면 완료
        # 주의: target_forearm_dir이 있을 때 위치만 보고 바로 break하면 안 됨 -
        # 50Hz 연속 추종 중엔 위치가 첫 iteration부터 eps 밑인 경우가 대부분이라,
        # 이 가드가 없으면 orientation 보정이 실행될 기회 자체를 못 얻음.
        if err_norm < eps and (orient_done or i >= orient_min_iter):
            break

        # 안전장치 3: 에러가 이전보다 커지면 발산 중 → 중단
        if i > 5 and err_norm > prev_err * 1.01:
            break
        prev_err = err_norm

        # 야코비안 계산 (손바닥 프레임, 월드 정렬 기준)
        # LOCAL_WORLD_ALIGNED라서 J[3:6,:]는 이미 world 좌표계의 각속도 Jacobian.
        # 원래부터 계산되던 값인데 여태 버려지고 있었음 - orientation 항 추가에
        # 별도 FK/Jacobian 재계산이 전혀 필요 없는 이유가 이것.
        J     = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        J_pos = J[:3, :]  # 위치 관련 행 (3×n)
        J_ang = J[3:6, :]  # orientation(각속도) 관련 행 (3×n)

        # joint_mask: 해당 팔 관절 열만 활성화, 나머지는 0으로 마스킹
        # 왼팔 IK 중에 오른팔 관절이 움직이는 걸 방지함
        if joint_mask is not None:
            mask         = np.zeros(J_pos.shape[1])
            mask[joint_mask] = 1.0
            J_pos        = J_pos * mask
            J_ang        = J_ang * mask

        # adaptive λ: 에러가 클수록 λ도 크게 (더 보수적으로)
        # 에러 0.001m → λ≈1e-4(정밀 추적), 에러 0.1m → λ≈1e-2(안전 우선)
        lam   = np.clip(err_norm * 0.1, 1e-4, 5e-2)

        # DLS pseudo-inverse
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lam * np.eye(3))

        # null-space projector: 위치 달성 후 남은 자유도
        N     = np.eye(len(q)) - J_dls @ J_pos

        # null-space 2차 과제 ① : 자연 자세 복원 (기존과 동일)
        dq_posture = -null_weight * (q - q_ref)

        # null-space 2차 과제 ② : 전완 방향 정렬 (신규)
        # d_current/e_orient는 위에서 이미 계산해뒀음 (재계산 없음).
        # wrist_yaw는 자기 축(Z) 중심 회전이라 이 방향 자체엔 영향 없음 (twist와 완전히 분리됨).
        #
        # 구현법: "Jacobian transpose" 방식(orient_weight*J_ang.T@e_orient) 대신
        # null space(joint_mask 4개 관절 - position 3제약 = 정확히 1차원) 안에서
        # 닫힌형(closed-form) 최적 스텝을 계산함.
        # position이 이미 수렴한 상태(err_norm 작음)라 λ가 최소값에 가까워서 N은
        # 거의 진짜 직교 투영행렬이므로, 아래 방식이 잘 성립함.
        #   1) seed 벡터를 N에 투영해서 null space 방향(n_hat, 단위벡터) 추출
        #   2) 그 방향으로 1rad 움직이면 orientation이 얼마나 바뀌는지(Jn) 계산
        #   3) e_orient를 Jn 방향으로 최소자승 투영한 스칼라(alpha)가 "최적 스텝 크기"
        # orient_weight는 이제 (0~1) 감쇠계수로 해석 - 1.0이면 선형화 기준 완전 정렬.
        dq_orient = np.zeros(len(q))
        if e_orient is not None:
            seed = np.zeros(len(q))
            if joint_mask is not None:
                seed[joint_mask] = 1.0
            else:
                seed[:] = 1.0
            n_raw  = N @ seed
            n_norm = np.linalg.norm(n_raw)
            if n_norm > 1e-8:
                n_hat = n_raw / n_norm
                Jn    = J_ang @ n_hat
                Jn_sq = np.dot(Jn, Jn)
                # Jn_sq(=null space 방향이 orientation에 갖는 leverage)가 작으면
                # alpha = dot(Jn,e_orient)/Jn_sq 가 나눗셈으로 폭주할 수 있음.
                # (실제로 겪은 문제: yaw가 몇 프레임 만에 수십 도씩 튀던 원인)
                if Jn_sq > 1e-4:   # 1e-8은 너무 관대해서 폭주를 못 막았음 - 기준 강화
                    alpha     = np.dot(Jn, e_orient) / Jn_sq
                    dq_orient = orient_weight * alpha * n_hat

            # orientation 전용 스텝 제한 - Jn_sq 기준 강화와 별개로 이중 안전장치.
            # 전체 dq의 0.3rad 클램프와는 별개(그건 position까지 합친 값이라
            # orientation 혼자서 너무 크게 튀는 걸 못 막았었음).
            dq_orient_norm = np.linalg.norm(dq_orient)
            if dq_orient_norm > orient_max_delta:
                dq_orient = dq_orient * (orient_max_delta / dq_orient_norm)

            # 관절 하드리밋 접근 감쇠 - 위 두 안전장치로도 못 막는 문제가 실측으로 확인됨:
            # 매 프레임 조금씩이라도 계속 같은 방향으로 밀면, 결국 관절 하드리밋(예: yaw ±90°)
            # 까지 가서 눌러붙어버림 (한 번 붙으면 매 프레임 "더 가고 싶은데 막힘"이 반복되며
            # 그 자리에서 안 떨어짐). NULL_WEIGHT 스프링만으로는 충분한 브레이크가 안 됨을 확인.
            # → 관절이 자기 리밋에 가까워질수록, "그 방향으로 더 미는" 성분을 부드럽게 0까지 감쇠.
            if joint_mask is not None:
                relevant_idx = joint_mask
            else:
                relevant_idx = range(len(q))
            for idx in relevant_idx:
                step = dq_orient[idx]
                if step == 0.0:
                    continue
                lo, hi = model.lowerPositionLimit[idx], model.upperPositionLimit[idx]
                margin = (hi - q[idx]) if step > 0 else (q[idx] - lo)
                if margin < orient_limit_margin:
                    scale = max(margin, 0.0) / orient_limit_margin
                    dq_orient[idx] *= scale

        # 최종 관절각 변화량: 위치 제어 + null-space(자세복원 + orientation 정렬)
        dq    = J_dls @ pos_error + N @ (dq_posture + dq_orient)

        # joint_mask: 해당 팔 관절만 q 업데이트
        if joint_mask is not None:
            dq_masked            = np.zeros_like(dq)
            dq_masked[joint_mask] = dq[joint_mask]
            dq                   = dq_masked

        # 안전장치 2: 한 번에 0.3rad 이상 이동 금지
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.3:
            dq = dq * (0.3 / dq_norm)

        q_new     = pin.integrate(model, q, dq)
        q_clipped = np.clip(q_new, model.lowerPositionLimit, model.upperPositionLimit)

        # 안전장치 4: q_new와 q_clipped 차이가 크면 관절 한계에 막힌 것
        if np.linalg.norm(q_new - q_clipped) > 0.01:
            clamp_count += 1
            if clamp_count >= CLAMP_LIMIT:
                break   # 타겟이 팔 길이 밖 → 조기 종료
        else:
            clamp_count = 0  # 클램핑 해소되면 카운터 리셋

        q = q_clipped
    return q


def apply_torso_compensation(model, data, q,
                              L_palm_id, R_palm_id,
                              l_target, r_target,
                              torso_idx, L_roll_idx, R_roll_idx):
    """
    shoulder_roll이 한계(0)에 막힌 경우 torso_yaw로 위치 오차를 보상합니다.

    개선 사항
    ─────────
    [1] 방향 판단: 가상 FK로 "roll을 열면 실제로 목표 쪽으로 가는가" 직접 확인
        - 단순 y축 부호 체크 대신 내적으로 판단 → 어떤 팔 자세에서도 정확
    [4] 스프링 복귀: return_step = max(MIN, |torso| × GAIN)
        - torso가 많이 돌아갔을수록 빠르게 복귀, 0 근처에서 자연스럽게 감속

    Returns
    -------
    q           : torso_yaw 갱신된 q 벡터
    torso_yaw   : 목표 torso_yaw 값 [rad] (EMA 전 raw값)
    compensated : bool (보상 중이면 True, 복귀 중이면 False)
    """
    if torso_idx is None:
        return q, 0.0, False, None

    ROLL_LIMIT_THRESH = 0.05   # roll이 이 값 이하면 limit에 붙어있다고 판단 [rad]

    current_torso = float(q[torso_idx])

    # ── Step 1: 현재 q로 FK → 손바닥 실제 위치 + 오차 계산 ──────
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    p_L = data.oMf[L_palm_id].translation.copy()
    p_R = data.oMf[R_palm_id].translation.copy()

    err_L = l_target - p_L
    err_R = r_target - p_R

    L_at_limit = q[L_roll_idx] < ROLL_LIMIT_THRESH
    R_at_limit = q[R_roll_idx] > -ROLL_LIMIT_THRESH

    # ── Step 2: 가상 FK — roll을 한계 너머로 조금 열었을 때 개선되는가 ──
    # roll이 limit에 붙어있을 때만 계산 (err 크기 무관)
    # 내적 > 0: roll을 열면 손바닥이 목표 방향으로 이동 → roll limit이 원인
    improvement_L = 0.0
    improvement_R = 0.0

    if L_at_limit:
        q_test = q.copy()
        q_test[L_roll_idx] -= config.TORSO_ROLL_TEST_DELTA
        pin.forwardKinematics(model, data, q_test)
        pin.updateFramePlacements(model, data)
        p_L_test      = data.oMf[L_palm_id].translation.copy()
        improvement_L = np.dot(p_L_test - p_L, err_L)

    if R_at_limit:
        q_test = q.copy()
        q_test[R_roll_idx] += config.TORSO_ROLL_TEST_DELTA
        pin.forwardKinematics(model, data, q_test)
        pin.updateFramePlacements(model, data)
        p_R_test      = data.oMf[R_palm_id].translation.copy()
        improvement_R = np.dot(p_R_test - p_R, err_R)

    # 복귀 조건: roll이 limit에서 벗어나면 (팔을 바깥으로 뺌) 복귀
    # err 조건 없음: 허리가 돌아간 상태에서 팔이 목표에 도달해도
    # roll이 여전히 limit에 붙어있으면 허리 유지
    L_needs = L_at_limit and improvement_L > 0
    R_needs = R_at_limit and improvement_R > 0

    # ── 우선순위 로직: 한쪽이 보상 중이면 반대쪽 차단 ──────────────
    # "양팔 동시 → 둘 다 취소" 방식은 L_needs가 역보정 부작용으로 True가 되면
    # 취소→복귀→재보상→반복의 진동 루프를 만들 수 있음.
    # 대신 한쪽이 먼저 needs이면 반대쪽은 평가하지 않는 우선순위 방식 적용.
    if L_needs and R_needs:
        L_needs = False   # 양팔 동시 → 방향 충돌 → 둘 다 취소
        R_needs = False
    elif R_needs:
        L_needs = False   # R팔 보상 중 → L팔 차단 (진동 방지)
    elif L_needs:
        R_needs = False   # L팔 보상 중 → R팔 차단 (진동 방지)

    # ── 보상 불필요: 스프링 방식으로 0으로 복귀 ──────────────────
    # [4] 개선: 고정 step → 스프링 (torso 각도 클수록 빠른 복귀)
    if not L_needs and not R_needs:
        if abs(current_torso) < config.TORSO_RETURN_MIN:
            new_torso = 0.0
        else:
            return_step = max(config.TORSO_RETURN_MIN,
                              abs(current_torso) * config.TORSO_RETURN_GAIN)
            new_torso   = current_torso - np.sign(current_torso) * return_step
        q = q.copy()
        q[torso_idx] = new_torso
        return q, new_torso, False, None

    # ── 보상 필요: Jacobian으로 delta 계산 ───────────────────────
    # 가상 FK 이후 data가 q_test 상태이므로 원래 q로 FK 복원
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p_torso = data.oMf[model.getFrameId('Torso')].translation.copy()

    Z = np.array([0.0, 0.0, 1.0])
    J_L = np.cross(Z, p_L - p_torso)
    J_R = np.cross(Z, p_R - p_torso)

    # 양팔 동시 요구는 위에서 이미 취소되므로 L_needs/R_needs 중 하나만 True
    delta_torso = 0.0
    if L_needs:
        J_L_sq = np.dot(J_L, J_L)
        if J_L_sq > 1e-6:
            delta_torso = np.dot(J_L, err_L) / J_L_sq
        else:
            return q, current_torso, False
    elif R_needs:
        J_R_sq = np.dot(J_R, J_R)
        if J_R_sq > 1e-6:
            delta_torso = np.dot(J_R, err_R) / J_R_sq
        else:
            return q, current_torso, False
    else:
        return q, current_torso, False

    # 프레임당 최대 변화량 제한
    delta_torso = float(np.clip(delta_torso,
                                -config.TORSO_MAX_DELTA_PER_FRAME,
                                 config.TORSO_MAX_DELTA_PER_FRAME))

    new_torso = float(np.clip(
        current_torso + delta_torso,
        model.lowerPositionLimit[torso_idx],
        model.upperPositionLimit[torso_idx],
    ))

    q = q.copy()
    q[torso_idx] = new_torso
    compensated_arm = 'R' if R_needs else 'L'
    return q, new_torso, True, compensated_arm
# ══════════════════════════════════════════════════════════════
# 손목 yaw 추출
# ══════════════════════════════════════════════════════════════

def extract_wrist_twist_z(rot_mat: np.ndarray, init_rot_mat: np.ndarray) -> float:
    """
    Quest에서 받은 손목 rotation 행렬에서 Z축 비틀림(twist)만 추출해 wrist_yaw 관절각으로 변환합니다.

    왜 이렇게 하나?
    ───────────────
    Quest 손목 rotation에는 손목 비틀림(yaw)뿐 아니라 손목 꺾임도 섞여 있음.
    단순히 오일러각 Z를 쓰면 다른 축 회전과 섞여서 부정확하고 불연속 문제도 생김.
    → 캘리브 시점 rotation(init_rot_mat)과의 상대 회전을 구한 뒤,
      쿼터니언에서 Z축 성분만 분리해 arctan2로 계산.

    Parameters
    ----------
    rot_mat      : 현재 손목 rotation 행렬 (3×3)
    init_rot_mat : 캘리브 시점 손목 rotation 행렬 (기준값)

    Returns
    -------
    angle : 캘리브 시점 대비 Z축 회전량 [rad], 범위 [-π, +π]
    """
    R_rel = init_rot_mat.T @ rot_mat              # 캘리브 대비 상대 rotation
    qx, qy, qz, qw = Rotation.from_matrix(R_rel).as_quat()
    angle = 2.0 * np.arctan2(qz, qw)             # Z축 twist만 추출
    return float((angle + np.pi) % (2 * np.pi) - np.pi)  # [-π, π]로 wrap


# ══════════════════════════════════════════════════════════════
# 좌표 변환: Quest → 로봇
# ══════════════════════════════════════════════════════════════

def quest_to_robot(pos: np.ndarray, quest_init: np.ndarray,
                   robot_init: np.ndarray) -> np.ndarray:
    """
    Quest 좌표계의 손 위치를 로봇 좌표계 목표 위치로 변환합니다.

    변환 방식
    ─────────
    Quest와 로봇의 좌표 축이 다르므로 축 재배치가 필요함:
      Quest  → 로봇
      -Z     → X (앞뒤)
      -X     → Y (좌우)
      +Y     → Z (상하)

    캘리브 시점의 Quest 손 위치(quest_init)를 기준으로 delta를 계산해
    로봇 초기 손바닥 위치(robot_init)에 더함.
    → 사람이 캘리브 자세에서 손을 앞으로 10cm 내밀면 로봇도 똑같이 10cm 내밈.

    원본에서 quest_to_robot_L / quest_to_robot_R 두 함수가 완전히 동일해서 하나로 통합함.
    """
    delta = pos - quest_init
    return robot_init + np.array([-delta[2], -delta[0], delta[1]])


def quest_dir_to_robot(quest_rot: np.ndarray, quest_local_axis: np.ndarray) -> np.ndarray:
    """
    Quest 손 rotation 행렬에서 특정 로컬 축(예: 전완이 향하는 방향)을 뽑아
    로봇 world 좌표계의 방향 단위벡터로 변환합니다.

    quest_to_robot()의 위치 축변환([-dz,-dx,dy])과 동일한 재배치를
    방향벡터에도 그대로 적용합니다 (선형 변환이라 delta든 방향이든 동일 행렬 사용 가능).

    quest_local_axis: Quest 손 트래킹 좌표계 기준, "전완이 몸 바깥을 향하는" 방향에
    해당하는 로컬 단위축. 이 값 자체(+X/-X/+Y/-Y/+Z/-Z 중 어느 것인지)는
    Quest 핸드트래킹 SDK 컨벤션에 달려있어 실측으로 확인 필요.
    (차렷 자세에서 전완을 몸 안/밖으로 돌려보며, 로봇이 기대한 방향으로
    yaw를 움직이는 축을 찾으면 됩니다 - R/L 부호 확인 때와 같은 절차.)

    Parameters
    ----------
    quest_rot        : 현재 Quest 손 rotation 행렬 (3×3)
    quest_local_axis : 전완 방향에 해당하는 로컬 단위축, 예: np.array([0,0,1])

    Returns
    -------
    로봇 world 좌표계 방향 단위벡터 (3,)
    """
    v_quest = quest_rot @ quest_local_axis
    v_robot = np.array([-v_quest[2], -v_quest[0], v_quest[1]])
    norm = np.linalg.norm(v_robot)
    return v_robot / norm if norm > 1e-9 else v_robot


def calc_target_from_calib(l_raw, r_raw,
                            quest_L_init, quest_R_init,
                            robot_L_init, robot_R_init):
    """
    매 프레임 Quest에서 받은 양손 위치를 로봇 목표 위치로 변환합니다.

    SYNCING 상태와 TELEOP 상태 모두에서 호출됨.
    캘리브레이션으로 저장된 기준점(quest_L/R_init, robot_L/R_init)을 인자로 받음.
    """
    l_target = quest_to_robot(l_raw, quest_L_init, robot_L_init)
    r_target = quest_to_robot(r_raw, quest_R_init, robot_R_init)
    return l_target, r_target

# ══════════════════════════════════════════════════════════════
# 역방향 좌표 변환: 로봇 → Quest (구체 오버레이용)
# ══════════════════════════════════════════════════════════════
# 아래 세 함수는 원래 HR_teleop.py 최상단에 있던 것들입니다.
# 로봇 모델/좌표계 관련 계산이므로 robot_model.py로 이동했습니다.

def robot_to_quest(robot_pos: np.ndarray,
                   robot_init: np.ndarray,
                   quest_init: np.ndarray) -> np.ndarray:
    """
    로봇 좌표계의 손바닥 위치를 Quest 좌표계로 역변환합니다.
    Quest 구체 오버레이에서 '로봇 손바닥이 지금 어디에 있는지'를
    Quest 공간에 표시할 때 사용합니다.

    pinocchio FK는 world 좌표계 기준으로 위치를 반환하므로
    torso가 돌아도 l_actual_vis는 이미 올바른 world 좌표임.
    별도의 torso 역보정 없이 단순 축 변환만 수행.

    quest_to_robot()의 역연산:
      robot delta = pos - robot_init
      quest = quest_init + [-delta[1], delta[2], -delta[0]]
    """
    delta = robot_pos - robot_init
    return quest_init + np.array([-delta[1], delta[2], -delta[0]])


def robot_dir_to_quest(rot_matrix: np.ndarray, side: str) -> np.ndarray:
    """
    로봇 손바닥 법선 벡터를 Quest 좌표계로 변환합니다.
    Quest 구체 오버레이의 손바닥 방향 화살표에 사용합니다.

    URDF 기준:
      오른손 손바닥 법선 → wrist_yaw 프레임의 +Y축
      왼손  손바닥 법선 → wrist_yaw 프레임의 -Y축

    Parameters
    ----------
    rot_matrix : 손바닥 프레임 rotation 행렬 (FK 결과)
    side       : 'L' 또는 'R'
    """
    palm_normal = rot_matrix[:, 1] if side == 'R' else -rot_matrix[:, 1]
    return np.array([-palm_normal[1], palm_normal[2], -palm_normal[0]])


def fk_palm_pose(model, data, frame_id: int,
                 q_vals: list) -> tuple:
    """
    특정 관절각(q_vals)에서 지정 프레임의 rotation·translation을 계산합니다.
    초기화 시 CALIB_POS/INIT_POS에서의 손바닥 pose를 사전계산하는 데 사용합니다.

    Parameters
    ----------
    q_vals : config.JOINT_ORDER 순서의 관절각 리스트 (12개)

    Returns
    -------
    (rotation, translation) : (3×3 ndarray, (3,) ndarray)
    """
    q_tmp = pin.neutral(model)
    for name, val in zip(config.JOINT_ORDER, q_vals):
        if model.existJointName(name):
            jid = model.getJointId(name)
            q_tmp[model.joints[jid].idx_q] = val
    pin.forwardKinematics(model, data, q_tmp)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id].rotation.copy(), data.oMf[frame_id].translation.copy()