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
    for name, val in zip(config.JOINT_ORDER, config.INIT_VALS):
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

    ids = {
        'L_palm':       L_palm_id,
        'R_palm':       R_palm_id,
        'joint_ids':    joint_ids,
        'L_joint_mask': joint_ids[0:4],   # 왼팔: L_shoulder(3개) + L_elbow(1개)
        'R_joint_mask': joint_ids[5:9],   # 오른팔: R_shoulder(3개) + R_elbow(1개)
    }
    return model, data, ids, q_init, robot_init


# ══════════════════════════════════════════════════════════════
# IK (역기구학)
# ══════════════════════════════════════════════════════════════

def compute_ik(model, data, frame_id, target_pos, q_cur,
               q_ref=None, q_init=None,
               max_iter=50, eps=1e-3,
               null_weight=0.3, joint_mask=None):
    """
    Position-only IK: 손바닥(frame_id)이 target_pos에 가도록 관절각을 반복 계산합니다.

    핵심 알고리즘: DLS pseudo-inverse + null-space 제어
    ───────────────────────────────────────────────────
    - DLS(Damped Least Squares): J†= Jᵀ(JJᵀ + λI)⁻¹
      일반 역행렬 대신 λ 항을 추가해 특이점(팔이 완전히 뻗은 상태 등)에서도 안정적임.
    - Null-space 제어: (I - J†J) @ q_null
      손바닥 위치를 유지하면서 팔꿈치가 자연스러운 자세(q_ref)로 돌아가게 함.
      이게 없으면 팔꿈치가 이상한 방향으로 꺾일 수 있음.

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

        # 수렴 조건: 오차가 eps(1mm) 이하면 완료
        if err_norm < eps:
            break

        # 안전장치 3: 에러가 이전보다 커지면 발산 중 → 중단
        if i > 5 and err_norm > prev_err * 1.01:
            break
        prev_err = err_norm

        # 야코비안 계산 (손바닥 프레임, 월드 정렬 기준)
        J     = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        J_pos = J[:3, :]  # 위치 관련 행만 사용 (3×n)

        # joint_mask: 해당 팔 관절 열만 활성화, 나머지는 0으로 마스킹
        # 왼팔 IK 중에 오른팔 관절이 움직이는 걸 방지함
        if joint_mask is not None:
            mask         = np.zeros(J_pos.shape[1])
            mask[joint_mask] = 1.0
            J_pos        = J_pos * mask

        # adaptive λ: 에러가 클수록 λ도 크게 (더 보수적으로)
        # 에러 0.001m → λ≈1e-4(정밀 추적), 에러 0.1m → λ≈1e-2(안전 우선)
        lam   = np.clip(err_norm * 0.1, 1e-4, 5e-2)

        # DLS pseudo-inverse
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lam * np.eye(3))

        # null-space projector: 위치 달성 후 남은 자유도
        N     = np.eye(len(q)) - J_dls @ J_pos

        # 최종 관절각 변화량: 위치 제어 + null-space에서 자연 자세 복원
        dq    = J_dls @ pos_error + N @ (-null_weight * (q - q_ref))

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
