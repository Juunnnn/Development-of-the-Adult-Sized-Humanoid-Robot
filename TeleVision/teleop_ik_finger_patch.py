"""
teleop_ik_finger_patch.py
─────────────────────────────────────────────────────────────
teleop_ik.py에 손가락 제어를 추가하기 위한 통합 가이드 및 패치 코드.

이 파일은 teleop_ik.py와 같은 폴더에 두고, 아래 설명대로 해당 위치에 코드를 삽입하세요.
총 4군데만 수정하면 됩니다.
"""

# ══════════════════════════════════════════════════════════════════
# [PATCH 1] import 섹션 상단에 추가
# 위치: from TeleVision import OpenTeleVision  바로 아래
# ══════════════════════════════════════════════════════════════════
PATCH_1 = """
from finger_mapping import (
    build_hand_cmd, is_landmark_valid, FingerEMAFilter
)
"""

# ══════════════════════════════════════════════════════════════════
# [PATCH 2] ROS Publisher 선언부에 추가
# 위치: pub = rospy.Publisher('/arm_controller/command', ...) 바로 아래
# ══════════════════════════════════════════════════════════════════
PATCH_2 = """
# 손가락 전용 publisher
# Float64MultiArray 16개: [L_AA_1, L_FE_1, ..., L_AA_4, L_FE_4,
#                          R_AA_1, R_FE_1, ..., R_AA_4, R_FE_4]
pub_hand = rospy.Publisher('/hand_controller/command', Float64MultiArray, queue_size=10)

# 손가락 EMA 필터 (alpha 낮을수록 부드러움 / 높을수록 빠른 반응)
finger_filter = FingerEMAFilter(alpha=0.4, n=16)

# 손가락 중립 자세 (모두 펼침)
FINGER_NEUTRAL = [0.0] * 16

# 캘리브 시점 landmark 저장용
calib_left_lm  = None
calib_right_lm = None
"""

# ══════════════════════════════════════════════════════════════════
# [PATCH 3] CALIBRATING 상태 완료 블록에 추가
# 위치: quest_R_wrist_rot_init = calib_samples_R_rot[-1]  바로 아래
# 캘리브 시점 landmark 저장
# ══════════════════════════════════════════════════════════════════
PATCH_3 = """
                # 캘리브 시점 손가락 landmark 저장 (AA 기준점으로 활용 가능)
                calib_left_lm  = tv.left_landmarks.copy()
                calib_right_lm = tv.right_landmarks.copy()
"""

# ══════════════════════════════════════════════════════════════════
# [PATCH 4] TELEOP 상태 메인 루프에서 pub.publish(cmd) 바로 아래에 추가
# 위치: pub.publish(cmd) 다음 줄
# ══════════════════════════════════════════════════════════════════
PATCH_4 = """
        # ── 손가락 제어 ──────────────────────────────────────────
        left_lm  = tv.left_landmarks   # (25, 3)
        right_lm = tv.right_landmarks  # (25, 3)

        if is_landmark_valid(left_lm) and is_landmark_valid(right_lm):
            raw_finger_cmd = build_hand_cmd(
                left_lm, right_lm,
                calib_left_lm=calib_left_lm,
                calib_right_lm=calib_right_lm
            )
            finger_cmd = finger_filter.filter(raw_finger_cmd)

            if not (np.any(np.isnan(finger_cmd)) or np.any(np.isinf(finger_cmd))):
                hand_msg = Float64MultiArray()
                hand_msg.data = finger_cmd.tolist()
                pub_hand.publish(hand_msg)
                print(f"[finger] L={[round(v,2) for v in finger_cmd[:8]]} "
                      f"R={[round(v,2) for v in finger_cmd[8:]]}")
        else:
            # landmark 없으면 중립(펼침) 자세 유지
            hand_msg = Float64MultiArray()
            hand_msg.data = FINGER_NEUTRAL
            pub_hand.publish(hand_msg)
        # ─────────────────────────────────────────────────────────
"""

# ══════════════════════════════════════════════════════════════════
# SYNCING 상태에서도 손가락 중립 publish (선택사항)
# 위치: SYNCING 블록의 pub.publish(cmd) 바로 아래
# ══════════════════════════════════════════════════════════════════
PATCH_4b = """
            # 싱크 중 손가락은 펼침 유지
            hand_msg = Float64MultiArray()
            hand_msg.data = FINGER_NEUTRAL
            pub_hand.publish(hand_msg)
"""

# ══════════════════════════════════════════════════════════════════
# publish_fin 수정 (종료 시 손가락도 펼침)
# 위치: publish_fin 함수 내 pub.publish(cmd) 바로 아래 (publish_smooth_move 내부)
# 또는 publish_fin 호출 직후에 아래를 추가
# ══════════════════════════════════════════════════════════════════
PATCH_SHUTDOWN = """
    # 종료 시 손가락 펼침
    hand_msg = Float64MultiArray()
    hand_msg.data = FINGER_NEUTRAL
    pub_hand.publish(hand_msg)
"""

# ══════════════════════════════════════════════════════════════════
# 테스트용 단독 실행 코드
# python3 finger_mapping.py 로 단독 테스트 가능
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from finger_mapping import build_hand_cmd, is_landmark_valid

    print("=== 손가락 매핑 단독 테스트 ===")

    # 가상 landmark: 손가락을 완전히 펼친 상태
    lm_open = np.zeros((25, 3))
    # 손목
    lm_open[0] = [0, 0, 0]
    # 손바닥 평면 기준점 (검지MCP, 약지MCP)
    lm_open[5]  = [0.03,  0.08, 0]   # 검지 MCP
    lm_open[13] = [-0.04, 0.06, 0]   # 약지 MCP

    # 각 손가락 MCP→Tip (손바닥 평면에서 앞으로 뻗음 → FE 적음)
    for i, base in enumerate([1, 5, 9, 13]):
        for j in range(4):
            lm_open[base+j] = [0.03*(i-1.5), 0.07 + j*0.02, 0.005*j]

    print("\n[펼침 상태]")
    cmd = build_hand_cmd(lm_open, lm_open)
    for i in range(4):
        print(f"  손가락{i+1}: AA={cmd[i*2]:.3f}rad  FE={cmd[i*2+1]:.3f}rad")
    print(f"  (R쪽도 동일하므로 생략)")

    # 가상 landmark: 손가락을 구부린 상태 (Tip이 손바닥 방향으로)
    lm_closed = lm_open.copy()
    palm_normal = np.array([0, 0, 1])  # 손등이 +Z
    for i, base in enumerate([1, 5, 9, 13]):
        mcp = lm_open[base]
        for j in range(1, 4):
            # Tip을 palm_normal 방향으로 구부림
            lm_closed[base+j] = mcp + palm_normal * (j * 0.025)

    print("\n[구부림 상태]")
    cmd_c = build_hand_cmd(lm_closed, lm_closed)
    for i in range(4):
        print(f"  손가락{i+1}: AA={cmd_c[i*2]:.3f}rad  FE={cmd_c[i*2+1]:.3f}rad")

    print("\n✅ 테스트 완료")
