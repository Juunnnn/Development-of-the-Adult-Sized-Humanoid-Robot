"""
finger_test.py
─────────────────────────────────────────────────────────────
Quest 손가락 트래킹 독립 테스트 스크립트.

실행 방법:
    conda activate tv
    cd ~/Development of the Adult Sized Humanoid Robot/Teleoperation/control
    python finger_test.py

기능:
  - Quest landmark 25개 실시간 수신
  - 손가락별 MCP~Tip 거리 출력
  - 손가락 구부림 비율 (0.0=펼침 ~ 1.0=완전구부림) 출력
  - 각도 방식도 함께 출력 (비교용)
  - 'c' 입력: 현재 상태로 캘리브 (L_calib 업데이트)
  - 'q' 입력: 종료
"""

import sys
import os
import time
import numpy as np
from multiprocessing import shared_memory, Queue, Event

# 경로 설정 (teleop_ik.py와 같은 폴더에서 실행 가정)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import config
sys.path.insert(0, os.path.join(config.TELEVISION_DIR, 'teleop'))
from TeleVision import OpenTeleVision

# ── Quest landmark 인덱스 ──────────────────────────────────
# (MCP, Tip) 쌍
FINGERS = {
    '엄지': (1,  4),
    '검지': (5,  8),
    '중지': (9,  12),
    '약지': (13, 16),
    '소지': (17, 20),
}

# 손바닥 참조점 (법선 계산용)
WRIST_IDX   = 0
IDX_FOREFINGER_MCP = 5
IDX_PINKY_MCP      = 17


def is_valid(lm):
    if lm is None: return False
    if np.any(np.isnan(lm)): return False
    if np.allclose(lm, 0, atol=1e-6): return False
    return True


def palm_normal(lm):
    v1 = lm[IDX_FOREFINGER_MCP] - lm[WRIST_IDX]
    v2 = lm[IDX_PINKY_MCP]      - lm[WRIST_IDX]
    n  = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-6 else np.array([0., 0., 1.])


def finger_angle(lm, mcp_i, tip_i):
    """MCP~Tip 벡터와 손 길이 방향의 각도 (굽힘 각도 추정)"""
    normal   = palm_normal(lm)
    ref_vec  = lm[IDX_FOREFINGER_MCP] - lm[WRIST_IDX]
    ref_proj = ref_vec - np.dot(ref_vec, normal) * normal
    ref_norm = np.linalg.norm(ref_proj)
    if ref_norm < 1e-6:
        return 0.0
    ref_proj /= ref_norm

    finger_vec  = lm[tip_i] - lm[mcp_i]
    finger_proj = finger_vec - np.dot(finger_vec, normal) * normal
    fn = np.linalg.norm(finger_proj)
    if fn < 1e-6:
        return 0.0
    finger_proj /= fn

    cos_a = np.clip(np.dot(ref_proj, finger_proj), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def bar(ratio, width=20):
    """0.0~1.0을 막대그래프로 표현"""
    ratio = np.clip(ratio, 0.0, 1.0)
    filled = int(ratio * width)
    return '[' + '█' * filled + '░' * (width - filled) + f'] {ratio*100:.0f}%'


def print_hand(label, lm, L_calib):
    """한 손 출력"""
    print(f"\n  {'─'*55}")
    print(f"  {label}")
    print(f"  {'─'*55}")
    print(f"  {'손가락':6s}  {'거리(mm)':>8s}  {'캘리브(mm)':>10s}  {'비율':>6s}  막대")
    print(f"  {'─'*55}")

    for i, (name, (mcp_i, tip_i)) in enumerate(FINGERS.items()):
        d = np.linalg.norm(lm[tip_i] - lm[mcp_i]) * 1000  # mm

        if L_calib is not None and i < 4:
            calib_mm = L_calib[i] * 1000
            # 비율: 캘리브(펼침)=1.0, 완전구부림 추정=캘리브*0.55
            min_d = calib_mm * 0.55
            ratio = (d - min_d) / (calib_mm - min_d)
            ratio = 1.0 - np.clip(ratio, 0.0, 1.0)  # 반전: 구부릴수록 1.0
            print(f"  {name:6s}  {d:8.1f}  {calib_mm:10.1f}  {bar(ratio)}")
        else:
            print(f"  {name:6s}  {d:8.1f}  {'(캘리브없음)':>10s}")

    # 추가: 각도 방식
    print(f"\n  [각도 방식]")
    for name, (mcp_i, tip_i) in FINGERS.items():
        ang = finger_angle(lm, mcp_i, tip_i)
        ratio = np.clip(ang / 90.0, 0.0, 1.0)
        print(f"  {name:6s}  {ang:5.1f}°  {bar(ratio, 15)}")


def main():
    print("=" * 60)
    print("  Quest 손가락 트래킹 테스트")
    print("=" * 60)
    print("  Quest 접속 후 손가락을 움직여보세요.")
    print("  명령: c=캘리브  q=종료")
    print("=" * 60)

    # TeleVision 초기화
    RES = (720, 1280)
    shm = shared_memory.SharedMemory(create=True, size=RES[0] * RES[1] * 3 * 2)
    image_array = np.ndarray((RES[0], RES[1] * 2, 3), dtype=np.uint8, buffer=shm.buf)
    image_array[:] = 128
    image_queue      = Queue()
    toggle_streaming = Event()

    tv = OpenTeleVision(RES, shm.name, image_queue, toggle_streaming,
                        stream_mode="image",
                        cert_file=config.CERT_FILE,
                        key_file=config.KEY_FILE,
                        ngrok=False)

    # 캘리브 값 (초기엔 None)
    L_calib_left  = None
    L_calib_right = None

    # finger_calib.json 있으면 로드
    import json
    if os.path.exists(config.FINGER_CALIB_PATH):
        with open(config.FINGER_CALIB_PATH) as f:
            fc = json.load(f)
        L_calib_left  = np.array(fc['L_calib_left'])
        L_calib_right = np.array(fc['L_calib_right'])
        print(f"✅ 캘리브 로드: L={( L_calib_left*1000).round(1)}mm  R={(L_calib_right*1000).round(1)}mm")
    else:
        print("⚠️  캘리브 파일 없음. 'c'로 캘리브하세요.")

    # 비블로킹 입력
    import threading
    cmd = {'val': ''}
    def input_thread():
        while True:
            c = input()
            cmd['val'] = c.strip().lower()
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    print("\n⏳ Quest 접속 대기 중...")

    try:
        while True:
            left_lm  = tv.left_landmarks
            right_lm = tv.right_landmarks

            # RAW 디버그: 무조건 출력
            ltype = type(left_lm).__name__
            lnone = left_lm is None
            if not lnone:
                try:
                    lshape = left_lm.shape
                    lsum   = float(np.sum(np.abs(left_lm)))
                except:
                    lshape = '?'
                    lsum   = '?'
            else:
                lshape, lsum = 'None', 0
            print(f"\r[raw] type={ltype} None={lnone} shape={lshape} sum={lsum:.3f}    ", end='', flush=True)

            time.sleep(0.2)
            continue

            if not l_valid and not r_valid:
                print("\r⏳ Quest 대기 중... (손을 Quest 카메라 앞에 보여주세요)", end='')
                time.sleep(0.1)
                continue

            # 명령 처리
            if cmd['val'] == 'q':
                print("\n종료합니다.")
                break
            elif cmd['val'] == 'c':
                if is_valid(left_lm) and is_valid(right_lm):
                    L_calib_left  = np.array([
                        np.linalg.norm(left_lm[tip] - left_lm[mcp])
                        for _, (mcp, tip) in list(FINGERS.items())[:4]
                    ])
                    L_calib_right = np.array([
                        np.linalg.norm(right_lm[tip] - right_lm[mcp])
                        for _, (mcp, tip) in list(FINGERS.items())[:4]
                    ])
                    with open(config.FINGER_CALIB_PATH, 'w') as f:
                        json.dump({
                            'L_calib_left':  L_calib_left.tolist(),
                            'L_calib_right': L_calib_right.tolist(),
                        }, f)
                    print(f"\n✅ 캘리브 완료!")
                    print(f"   L={( L_calib_left*1000).round(1)}mm")
                    print(f"   R={(L_calib_right*1000).round(1)}mm")
                    print(f"   finger_calib.json 저장됨")
                else:
                    print("\n⚠️  landmark 없음. Quest 연결 확인!")
                cmd['val'] = ''

            # 화면 출력
            if l_valid or r_valid:
                os.system('clear')
                print("=" * 60)
                print("  Quest 손가락 트래킹 테스트  |  c=캘리브  q=종료")
                print("=" * 60)

                if is_valid(left_lm):
                    print_hand("👈 왼손", left_lm, L_calib_left)
                else:
                    print("\n  👈 왼손: 트래킹 없음")

                if is_valid(right_lm):
                    print_hand("👉 오른손", right_lm, L_calib_right)
                else:
                    print("\n  👉 오른손: 트래킹 없음")

                print("\n  명령: c=캘리브(손 펼친 상태에서)  q=종료")

            time.sleep(0.1)  # 10Hz로 출력

    finally:
        shm.close()
        shm.unlink()
        print("정리 완료.")


if __name__ == '__main__':
    main()