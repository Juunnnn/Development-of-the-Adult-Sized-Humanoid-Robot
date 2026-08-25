"""
ros_interface.py
─────────────────────────────────────────────────────────────
ROS 노드 초기화, Publisher/Subscriber 생성,
카메라 콜백, 관절값 읽기를 담당합니다.

대체 함수/블록:
  - rospy.init_node / Publisher 선언부
  - camera_callback()
  - joint_state_callback(), get_current_q()
"""

import time
import threading
import numpy as np
import cv2
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CompressedImage, JointState
import pinocchio as pin

import config

from std_msgs.msg import Float64MultiArray, Float32MultiArray, Bool  # Float32MultiArray, Bool 추가

# 파일 상단, import 근처에 추가
JOINT_NAME_REMAP = {
    'L_elbow_joint': 'L_elbow_pitch_joint',
    'R_elbow_joint': 'R_elbow_pitch_joint',
}

# ── 눈 하나당 버퍼 해상도 (HR_teleop.py의 RES와 일치해야 함) ──
EYE_W = 1280
EYE_H = 720

# ── 카메라 토픽 방식 선택 ─────────────────────────────
# True  : /camera/color/image_raw/compressed (JPEG, ~3MB/s)  ← 대역폭 절감
# False : /camera/color/image_raw            (무압축, ~55MB/s) ← 기존 동작
# 압축 토픽에서 화면이 안 나오면 False로 바꿔 즉시 기존 동작으로 복귀 가능.
USE_COMPRESSED_CAMERA = True

# config.py에 아래 두 줄 추가하고 여기서 import해서 씀
# JOINT_SANITY_RANGE = {name: (lo, hi) for name, lo, hi in zip(...)}  ← 4번 항목에서 설명

class RosInterface:
    """
    텔레옵에 필요한 ROS 인터페이스를 캡슐화합니다.

    Attributes
    ----------
    pub_arm  : Publisher  → /arm_controller/command
    pub_hand : Publisher  → /finger_controller/command
    """

    # 상태별 오버레이 색상 (BGR)
    _STATE_COLORS = {
        'WAITING_QUEST':       (140, 140, 140),
        'CALIBRATING':         (0, 165, 255),
        'CALIBRATING_FINGERS': (0, 165, 255),
        'SYNCING':             (0, 220, 220),
        'TELEOP':              (60, 220, 60),
        'FREEZE':              (50,  50, 255),
    }

    def __init__(self, model, q_init, image_array: np.ndarray):
        """
        Parameters
        ----------
        model       : pinocchio.Model  (관절명 → idx 변환에 사용)
        q_init      : np.ndarray       (joint_states 미수신 시 fallback)
        image_array : np.ndarray       (카메라 → Quest 스트리밍용 공유 배열)
        """
        self.model       = model
        self.q_init      = q_init
        self.image_array = image_array

        self._current_joint_state = None
        self._draw_buffer         = None
        self._last_camera_time    = 0.0
        self._decode_fail         = 0      # JPEG 디코딩 실패 누적
        self._frame_ok            = False  # 첫 프레임 수신 여부
        self._cam_warned          = False  # 무프레임 경고 1회 출력용
        self._start_time          = time.time()
        self._fit_geom            = None   # 레터박스 배치 캐시 ((h,w), nw, nh, ox, oy)
        self.video_playing        = False   # True이면 카메라 콜백·오버레이 루프 억제
        self._buf_lock            = threading.Lock()  # _draw_buffer / image_array 동시 접근 방지

        # ── 오버레이 공유 데이터 ───────────────────────────
        self.overlay: dict = {
            'state':            'WAITING_QUEST',
            'hz':               0.0,
            'countdown':        -1,
            'calib_wait':       0.0,   # > 0 이면 수집 전 대기 카운트다운
            'calib_n':          0,
            'calib_total':      50,
            'finger_n':         0,
            'finger_total':     50,
            'sync_elapsed':     0.0,
            'sync_timeout':     10.0,
            'freeze_remaining': 0.0,
            'l_err':            None,
            'r_err':            None,
            'l_joints':         [],
            'r_joints':         [],
            'neck_warn':        '',   # 'YAW' | 'PITCH' | 'BOTH' | ''
            'torso_yaw':        0.0,  # 현재 torso yaw 값 [rad]
            'torso_state':      '',   # 'COMP' | 'RETURN' | ''
            'grasp_state':      False,  # 젯슨 그리퍼 grasp 판정 결과
        }

        # rospy.init_node('teleop_ik', disable_signals=True)
        rospy.init_node('teleop_ik', anonymous=True, disable_signals=True)

        self.pub_arm  = rospy.Publisher(config.TOPIC_ARM,   Float64MultiArray, queue_size=10)
        self.pub_hand = rospy.Publisher(config.TOPIC_HAND,  Float64MultiArray, queue_size=10)
        self.pub_torso = rospy.Publisher(config.TOPIC_TORSO, Float64MultiArray, queue_size=1)

        self.pub_amazing_hand = rospy.Publisher('/amazing_hand/finger_angles', Float32MultiArray, queue_size=10)

        # 목 전용 publisher (젯슨 neck_dynamixel_node 수신용)
        self.pub_neck = rospy.Publisher('/neck_controller/command', Float64MultiArray, queue_size=1)

        # 그리퍼(오른손) 전용 publisher — 목과 같은 젯슨 노드(neck_dynamixel_node)가
        # 같은 U2D2 버스에서 함께 처리 (아래 publish_gripper 참고)
        self.pub_gripper = rospy.Publisher('/gripper_controller/command', Float64MultiArray, queue_size=1)

        if USE_COMPRESSED_CAMERA:
            _cam_topic = config.TOPIC_CAMERA.rstrip('/') + '/compressed'
            rospy.Subscriber(_cam_topic, CompressedImage, self._camera_callback)
        else:
            _cam_topic = config.TOPIC_CAMERA
            rospy.Subscriber(_cam_topic, Image, self._camera_callback_raw)
        print(f"[camera] 구독 토픽: {_cam_topic}  (compressed={USE_COMPRESSED_CAMERA})")
        rospy.Subscriber(config.TOPIC_JOINT_STATES, JointState, self._joint_state_callback)
        rospy.Subscriber(config.TOPIC_GRIPPER_STATUS, Bool,     self._grasp_status_callback)

        # ── 카메라 없을 때도 오버레이를 그리는 백그라운드 스레드 ──
        # 카메라가 꺼져 있으면 _camera_callback이 한 번도 호출되지 않아
        # image_array가 회색(128) 그대로 유지됨.
        # 이 스레드가 10Hz로 어두운 배경 위에 오버레이를 직접 렌더링해
        # 카메라 없이도 Quest 화면에 상태/손 위치 정보가 표시되게 함.
        self._overlay_thread = threading.Thread(
            target=self._overlay_loop, daemon=True
        )
        self._overlay_thread.start()

    # ── 오른쪽 정렬 텍스트 헬퍼 ───────────────────────────
    def _put_text_right(self, img, text, y, x0, W, font, scale, color, thickness=1):
        """오른쪽 끝(x0+W)에서 18px 여백으로 오른쪽 정렬 출력."""
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = int(x0 + W - tw - 18)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # ── 카메라 없을 때 오버레이 루프 ──────────────────────
    def _overlay_loop(self):
        """
        카메라 프레임이 0.5초 이상 안 들어오면
        어두운 배경(RGB 30) 위에 오버레이를 10Hz로 직접 렌더링.
        카메라가 들어오는 동안에는 _camera_callback이 담당하므로 이 루프는 스킵.
        """
        BG_COLOR = 30   # 어두운 회색 배경
        while True:
            time.sleep(0.1)
            if self.video_playing:
                continue   # 영상 재생 중에는 오버레이 루프 스킵
            if time.time() - self._last_camera_time < 0.5:
                continue

            # 시작 후 5초간 프레임이 한 번도 안 오면 1회 경고
            if (not self._frame_ok and not self._cam_warned
                    and time.time() - self._start_time > 5.0):
                self._cam_warned = True
                print("[camera] 5초간 프레임 없음. 아래를 확인하세요:\n"
                      "  rostopic hz   {0}/compressed\n"
                      "  rostopic list | grep compressed\n"
                      "  (토픽이 없으면 ros_interface.py의 "
                      "USE_COMPRESSED_CAMERA = False 로 변경)"
                      .format(config.TOPIC_CAMERA.rstrip('/')))

            # 버퍼 초기화 (최초 1회)
            if self._draw_buffer is None:
                self._draw_buffer = np.full_like(self.image_array, BG_COLOR)

            self._draw_buffer[:] = BG_COLOR
            self._draw_overlay(self._draw_buffer)
            with self._buf_lock:
                np.copyto(self.image_array, self._draw_buffer)

    # ── 그리퍼 grasp 상태 콜백 ───────────────────────────────
    def _grasp_status_callback(self, msg: Bool):
        """젯슨 neck_dynamixel_node의 grasp 판정 결과 수신 → overlay 갱신.
        디바운스는 이미 젯슨쪽에서 처리하고 오므로 여기선 그대로 반영만 함."""
        self.overlay['grasp_state'] = bool(msg.data)

    # ── 카메라 콜백 (압축 JPEG) ────────────────────────────
    def _camera_callback(self, msg: CompressedImage):
        if self.video_playing:
            return   # 영상 재생 중에는 카메라 프레임이 영상을 덮어쓰지 못하도록 차단
        try:
            arr   = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR로 디코딩
            if frame is None:
                # imdecode는 실패 시 예외가 아니라 None을 반환함.
                # 여기서 _last_camera_time을 갱신하지 않아야 오버레이 루프가
                # 대체 화면을 계속 그려줌 (완전 회색 화면 방지).
                self._decode_fail += 1
                if self._decode_fail in (1, 30, 300):
                    print(f"[camera] JPEG 디코딩 실패 {self._decode_fail}회 "
                          f"(data {len(msg.data)} bytes) — 압축 토픽 형식 확인 필요")
                return
            # 압축 토픽은 BGR로 디코딩되므로 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._push_frame(frame)
        except Exception as e:
            print(f"카메라 오류(compressed): {e}")

    # ── 카메라 콜백 (무압축) ───────────────────────────────
    def _camera_callback_raw(self, msg: Image):
        if self.video_playing:
            return
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self._push_frame(frame)
        except Exception as e:
            print(f"카메라 오류(raw): {e}")

    # ── 프레임 → 공유 버퍼 (두 콜백 공통) ──────────────────
    def _push_frame(self, frame):
        self._last_camera_time = time.time()   # 카메라 활성 시각 갱신
        if not self._frame_ok:
            self._frame_ok = True
            print(f"[camera] 첫 프레임 수신 OK — 원본 {frame.shape[1]}x{frame.shape[0]}")

        # 더블 버퍼 초기화 (최초 1회)
        if self._draw_buffer is None:
            self._draw_buffer = np.zeros_like(self.image_array)

        # ── 종횡비 유지 배치 (letterbox / pillarbox) ────────
        # 640x480(4:3)을 1280x720(16:9)에 그냥 늘리면 가로로 33% 왜곡됨.
        # 비율을 유지해 960x720으로 축소 배치하고 좌우는 검은 여백으로 둠.
        h, w = frame.shape[:2]
        if self._fit_geom is None or self._fit_geom[0] != (h, w):
            scale = min(EYE_W / w, EYE_H / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            ox, oy = (EYE_W - nw) // 2, (EYE_H - nh) // 2
            self._fit_geom = ((h, w), nw, nh, ox, oy)
            print(f"[camera] 배치: {w}x{h} → {nw}x{nh} "
                  f"(여백 좌우 {ox}px / 상하 {oy}px, 종횡비 유지)")

        _, nw, nh, ox, oy = self._fit_geom
        fitted = cv2.resize(frame, (nw, nh))

        # 여백 영역을 검게 유지 (오버레이 폴백 화면이 남아있을 수 있으므로 매 프레임)
        if ox > 0:
            self._draw_buffer[:, :ox]                       = 0
            self._draw_buffer[:, ox + nw:EYE_W + ox]         = 0
            self._draw_buffer[:, EYE_W + ox + nw:]           = 0
        if oy > 0:
            self._draw_buffer[:oy]                          = 0
            self._draw_buffer[oy + nh:]                      = 0

        # 버퍼에 프레임 + 오버레이를 완전히 그린 후 (양쪽 눈 동일)
        self._draw_buffer[oy:oy + nh, ox:ox + nw]                 = fitted
        self._draw_buffer[oy:oy + nh, EYE_W + ox:EYE_W + ox + nw] = fitted
        self._draw_overlay(self._draw_buffer)

        # 완성된 프레임을 한 번에 복사 → 깜박임 방지
        with self._buf_lock:
            np.copyto(self.image_array, self._draw_buffer)

    # ── 오버레이 그리기 ────────────────────────────────────
    def _draw_overlay(self, img):
        """카메라 영상 위에 상태·손 위치 오버레이를 그린다 (양쪽 눈 동일)."""
        ov    = self.overlay
        state = ov.get('state', 'WAITING_QUEST')
        color = self._STATE_COLORS.get(state, (200, 200, 200))
        W     = EYE_W   # 한쪽 눈 너비

        for x0 in (0, W):
            self._draw_eye(img, x0, W, state, color, ov)

    def _draw_eye(self, img, x0, W, state, color, ov):
        """한쪽 눈 영역에 오버레이 요소를 순서대로 그린다."""
        F = cv2.FONT_HERSHEY_SIMPLEX

        # ── 상단 반투명 바 ─────────────────────────────────
        img[0:72, x0:x0 + W] = (img[0:72, x0:x0 + W] * 0.45).astype(np.uint8)

        # 상태 이름 (오른쪽 정렬)
        self._put_text_right(img, state, 50, x0, W, F, 1.3, color, 2)

        # # Hz (상태 이름 왼쪽에)
        # hz_txt = f"{ov.get('hz', 0.0):.1f} Hz"
        # (state_w, _), _ = cv2.getTextSize(state, F, 1.3, 2)
        # (hz_w, _), _    = cv2.getTextSize(hz_txt, F, 1.0, 2)
        # hz_x = int(x0 + W - state_w - hz_w - 40)
        # cv2.putText(img, hz_txt, (hz_x, 50), F, 1.0, (190, 190, 190), 2, cv2.LINE_AA)

        # ── 상태별 추가 정보 (오른쪽 정렬) ────────────────
        if state == 'WAITING_QUEST':
            countdown = ov.get('countdown', -1)
            l_actual  = ov.get('l_actual')
            r_actual  = ov.get('r_actual')

            if countdown > 0:
                # 카운트다운 숫자를 화면 중앙에 크게 표시
                num_txt = str(countdown)
                (nw, nh), _ = cv2.getTextSize(num_txt, F, 8.0, 12)
                nx = x0 + (W - nw) // 2
                ny = 720 // 2 + nh // 2
                # 그림자 (가독성)
                cv2.putText(img, num_txt, (nx + 4, ny + 4), F, 8.0, (0, 0, 0), 16, cv2.LINE_AA)
                cv2.putText(img, num_txt, (nx, ny),         F, 8.0, (60, 220, 60), 12, cv2.LINE_AA)
                # 안내 텍스트
                guide = "Starting Teleoperation in..."
                (gw, _), _ = cv2.getTextSize(guide, F, 0.85, 2)
                cv2.putText(img, guide, (x0 + (W - gw) // 2, ny - nh - 20),
                            F, 0.85, (200, 200, 200), 2, cv2.LINE_AA)

                # 로봇 손바닥 초기 위치 (카운트다운과 함께 표시)
                if l_actual is not None or r_actual is not None:
                    img[590:, x0:x0 + W] = (img[590:, x0:x0 + W] * 0.45).astype(np.uint8)

                    header = "Robot palm position (align your hands here)"
                    (hw, _), _ = cv2.getTextSize(header, F, 0.65, 1)
                    cv2.putText(img, header, (x0 + (W - hw) // 2, 612),
                                F, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

                    if l_actual is not None:
                        ltxt = f"L  x:{l_actual[0]:+.2f}  y:{l_actual[1]:+.2f}  z:{l_actual[2]:+.2f} m"
                        (lw, _), _ = cv2.getTextSize(ltxt, F, 0.8, 2)
                        cv2.putText(img, ltxt, (x0 + (W - lw) // 2, 648),
                                    F, 0.8, (60, 220, 60), 2, cv2.LINE_AA)

                    if r_actual is not None:
                        rtxt = f"R  x:{r_actual[0]:+.2f}  y:{r_actual[1]:+.2f}  z:{r_actual[2]:+.2f} m"
                        (rw, _), _ = cv2.getTextSize(rtxt, F, 0.8, 2)
                        cv2.putText(img, rtxt, (x0 + (W - rw) // 2, 684),
                                    F, 0.8, (100, 180, 255), 2, cv2.LINE_AA)

                    hint = "Green = Left arm   Blue = Right arm"
                    (htw, _), _ = cv2.getTextSize(hint, F, 0.6, 1)
                    cv2.putText(img, hint, (x0 + (W - htw) // 2, 714),
                                F, 0.6, (130, 130, 130), 1, cv2.LINE_AA)
            else:
                guide = "Waiting for Quest connection..."
                (gw, _), _ = cv2.getTextSize(guide, F, 0.85, 2)
                cv2.putText(img, guide, (x0 + (W - gw) // 2, 370),
                            F, 0.85, (200, 200, 200), 2, cv2.LINE_AA)

        elif state == 'CALIBRATING':
            n          = ov.get('calib_n', 0)
            total      = ov.get('calib_total', 50)
            calib_wait = ov.get('calib_wait', 0.0)

            if calib_wait > 0:
                # ── 대기 단계: 큰 숫자 카운트다운 ──────────────
                num_txt = str(int(np.ceil(calib_wait)))
                (nw, nh), _ = cv2.getTextSize(num_txt, F, 6.0, 10)
                nx = x0 + (W - nw) // 2
                ny = 720 // 2 + nh // 2
                cv2.putText(img, num_txt, (nx+3, ny+3), F, 6.0, (0,0,0), 14, cv2.LINE_AA)
                cv2.putText(img, num_txt, (nx, ny),     F, 6.0, color, 10, cv2.LINE_AA)
                guide = "Position your hands on the spheres!"
                (gw, _), _ = cv2.getTextSize(guide, F, 0.85, 2)
                cv2.putText(img, guide, (x0 + (W-gw)//2, ny - nh - 20),
                            F, 0.85, (200,200,200), 2, cv2.LINE_AA)
            else:
                # ── 수집 단계: n/total 텍스트 + 프로그레스 바 ──
                count_txt = f"{n} / {total}"
                (cw, _), _ = cv2.getTextSize(count_txt, F, 1.2, 2)
                cv2.putText(img, count_txt, (x0 + (W-cw)//2, 110),
                            F, 1.2, color, 2, cv2.LINE_AA)
                self._draw_progress_bar(img, x0+18, 130, W-36, 18,
                                        n / max(total,1), color)
                self._put_text_right(img, "Stretch arms forward!",
                                     160, x0, W, F, 0.8, (200,200,200), 1)

        elif state == 'CALIBRATING_FINGERS':
            n     = ov.get('finger_n', 0)
            total = ov.get('finger_total', 50)
            self._put_text_right(img, f"Show palms to Quest!  ({n}/{total})",
                                 110, x0, W, F, 0.85, color, 2)
            self._draw_progress_bar(img, x0 + 18, 130, W - 36, 18,
                                    n / max(total, 1), color)

        elif state == 'SYNCING':
            elapsed = ov.get('sync_elapsed', 0.0)
            timeout = ov.get('sync_timeout', 10.0)
            frac    = min(elapsed / max(timeout, 0.001), 1.0)
            self._put_text_right(img, f"Syncing... ({elapsed:.1f}s / {timeout:.0f}s)",
                                 110, x0, W, F, 0.85, color, 2)
            self._draw_progress_bar(img, x0 + 18, 130, W - 36, 18,
                                    frac, color)

        elif state == 'FREEZE':
            rem = ov.get('freeze_remaining', 0.0)
            self._put_text_right(img, f"Tracking Lost  {rem:.1f}s after return",
                                 110, x0, W, F, 0.9, color, 2)

        elif state == 'TELEOP':
            self._draw_hand_info(img, x0, W, ov)

            # 그리퍼 grasp 인디케이터 — 쥐고 있는 동안 오른쪽 위, 상태 이름 바로 아래
            if ov.get('grasp_state', False):
                self._put_text_right(img, "GRASP", 82, x0, W, F, 0.8, (60, 220, 60), 2)

            # 목 한계 경고 (화면 정중앙)
            neck_warn = ov.get('neck_warn', '')
            if neck_warn:
                warn_txt = f"WARNING: Neck {neck_warn} Limit Reached"
                (tw, th), _ = cv2.getTextSize(warn_txt, F, 0.8, 2)
                cx = x0 + (W - tw) // 2
                cy = 370
                cv2.rectangle(img, (cx - 8, cy - th - 6), (cx + tw + 8, cy + 8), (20, 20, 20), -1)
                cv2.putText(img, warn_txt, (cx, cy),
                            F, 0.8, (255, 69, 0), 2, cv2.LINE_AA)

    def _draw_progress_bar(self, img, x, y, w, h, frac, color):
        """진행 바: 배경(회색) 위에 채워진 사각형."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
        filled = int(w * min(max(frac, 0.0), 1.0))
        if filled > 0:
            cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)

    def _draw_hand_info(self, img, x0, W, ov):
        l_err       = ov.get('l_err')
        r_err       = ov.get('r_err')
        l_joints    = ov.get('l_joints', [])
        r_joints    = ov.get('r_joints', [])
        torso_yaw   = ov.get('torso_yaw',   0.0)
        torso_state = ov.get('torso_state', '')
        F = cv2.FONT_HERSHEY_SIMPLEX

        def err_color(e):
            if e is None:  return (160, 160, 160)
            if e < 0.05:   return (60, 220, 60)
            if e < 0.15:   return (0, 165, 255)
            return             (50,  50, 255)

        img[550:, x0:x0 + W] = (img[550:, x0:x0 + W] * 0.45).astype(np.uint8)

        # ── Torso yaw 표시 (왼쪽, 상단) ────────────────────────────
        # state별 색상: COMP=주황, RETURN=하늘, 비활성=회색
        if torso_state == 'COMP':
            t_color = (0, 165, 255)    # 주황
            t_label = f"TORSO COMP  {torso_yaw:+.3f} rad"
        elif torso_state == 'RETURN':
            t_color = (255, 220, 60)   # 하늘
            t_label = f"TORSO RTN   {torso_yaw:+.3f} rad"
        else:
            t_color = (80, 80, 80)     # 회색 (비활성)
            t_label = None

        if t_label is not None:
            cv2.putText(img, t_label, (x0 + 18, 580), F, 0.75, t_color, 2, cv2.LINE_AA)

        # Torso 게이지 바 (±1.57 범위)
        BAR_X   = x0 + 18
        BAR_Y   = 590
        BAR_W   = 220
        BAR_H   = 10
        BAR_MAX = 1.57
        # 배경 바
        cv2.rectangle(img, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H),
                      (50, 50, 50), -1)
        # 중앙선
        mid_x = BAR_X + BAR_W // 2
        cv2.line(img, (mid_x, BAR_Y - 2), (mid_x, BAR_Y + BAR_H + 2),
                 (100, 100, 100), 1)
        # 현재값 마커
        if abs(torso_yaw) > 0.003:
            ratio    = float(np.clip(torso_yaw / BAR_MAX, -1.0, 1.0))
            marker_x = int(mid_x + ratio * (BAR_W // 2))
            fill_x0  = min(mid_x, marker_x)
            fill_x1  = max(mid_x, marker_x)
            bar_color = t_color if t_label else (80, 80, 80)
            cv2.rectangle(img, (fill_x0, BAR_Y), (fill_x1, BAR_Y + BAR_H),
                          bar_color, -1)
            cv2.circle(img, (marker_x, BAR_Y + BAR_H // 2), 5, bar_color, -1)

        # ── IK 오차 (오른쪽 정렬) ──────────────────────────────────
        lc  = err_color(l_err)
        ltx = f"L err: {l_err*100:.1f}cm" if l_err is not None else "L err: ---"
        self._put_text_right(img, ltx, 610, x0, W, F, 0.8, lc, 2)

        rc  = err_color(r_err)
        rtx = f"R err: {r_err*100:.1f}cm" if r_err is not None else "R err: ---"
        self._put_text_right(img, rtx, 640, x0, W, F, 0.8, rc, 2)

        # ── 관절각 (오른쪽 정렬) ───────────────────────────────────
        joint_names = ["SP", "SR", "SY", "EP", "WY"]
        if l_joints:
            header = "L:  " + "  ".join(f"{n:>6}" for n in joint_names)
            values = "    " + "  ".join(f"{v:>6.2f}" for v in l_joints)
            self._put_text_right(img, header, 665, x0, W, F, 0.55, (120, 120, 120), 1)
            self._put_text_right(img, values, 685, x0, W, F, 0.55, (180, 180, 180), 1)
        if r_joints:
            header = "R:  " + "  ".join(f"{n:>6}" for n in joint_names)
            values = "    " + "  ".join(f"{v:>6.2f}" for v in r_joints)
            self._put_text_right(img, header, 700, x0, W, F, 0.55, (120, 120, 120), 1)
            self._put_text_right(img, values, 715, x0, W, F, 0.55, (180, 180, 180), 1)

    # ── 관절 상태 콜백 ─────────────────────────────────────
    def _joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg

    # ── 현재 q 읽기 ────────────────────────────────────────
    def get_current_q(self) -> np.ndarray:
        """
        /joint_states에서 받은 실제 관절값으로 pinocchio q 벡터를 구성.
        수신 전이면 q_init으로 대체 (경고 출력).
        """
        if self._current_joint_state is None:
            print("⚠️  /joint_states 미수신 → q_init으로 대체 (로봇이 앞으로나란히 자세여야 안전)")
            return self.q_init.copy()

        q_cur = pin.neutral(self.model)
        unmatched, out_of_range = [], []

        for name, val in zip(self._current_joint_state.name,
                             self._current_joint_state.position):
            mapped = JOINT_NAME_REMAP.get(name, name)

            if not self.model.existJointName(mapped):
                unmatched.append(name)
                continue

            lo, hi = config.JOINT_SANITY_RANGE.get(mapped, (-np.pi, np.pi))
            if not (lo <= val <= hi):
                out_of_range.append((name, val))
                continue

            jid        = self.model.getJointId(mapped)
            idx        = self.model.joints[jid].idx_q
            q_cur[idx] = val

        if unmatched and not getattr(self, '_warned_unmatched', False):
            print(f"⚠️  /joint_states 이름 매칭 실패: {unmatched}")
            self._warned_unmatched = True
        if out_of_range:
            print(f"🚨 /joint_states 값 이상 (무시됨): {out_of_range}")

        return q_cur

    def wait_for_joint_states(self, timeout: float = None) -> np.ndarray:
        """
        /joint_states 수신까지 대기 후 현재 q 반환.
        timeout(s) 초과 시 q_init 반환.
        """
        if timeout is None:
            timeout = config.JOINT_STATE_TIMEOUT

        print(f"⏳ /joint_states 수신 대기 중... (최대 {timeout}s)")
        t0 = time.time()
        while self._current_joint_state is None and not rospy.is_shutdown():
            if time.time() - t0 > timeout:
                print("⚠️  /joint_states 수신 타임아웃 → q_init으로 대체")
                break
            time.sleep(0.05)

        return self.get_current_q()

    # ── publish 헬퍼 ───────────────────────────────────────
    def publish_arm(self, data: list):
        msg = Float64MultiArray()
        msg.data = data
        self.pub_arm.publish(msg)

        # # ── 임시 디버그: 실제 호출 빈도 + 호출 스택 확인 ──
        # import traceback
        # self._arm_pub_count = getattr(self, '_arm_pub_count', 0) + 1
        # if self._arm_pub_count <= 3:          # 처음 3번만 호출 스택 출력
        #     print(f"[DEBUG publish_arm #{self._arm_pub_count}] 호출 스택:")
        #     traceback.print_stack()
        # now = time.time()
        # if not hasattr(self, '_arm_pub_last_print'):
        #     self._arm_pub_last_print = now
        # if now - self._arm_pub_last_print >= 1.0:
        #     print(f"[DEBUG publish_arm] 최근 1초간 {self._arm_pub_count}회 호출")
        #     self._arm_pub_count = 0
        #     self._arm_pub_last_print = now

    def publish_neck(self, yaw: float, pitch: float):
        """목 관절각을 /neck_controller/command 토픽으로 전송 (젯슨 수신용)."""
        msg      = Float64MultiArray()
        msg.data = [float(yaw), float(pitch)]
        self.pub_neck.publish(msg)

    def publish_gripper(self, ratio: float):
        """그리퍼 개폐 비율(0=열림, 1=닫힘)을 /gripper_controller/command로 전송 (젯슨 수신용)."""
        msg      = Float64MultiArray()
        msg.data = [float(np.clip(ratio, 0.0, 1.0))]
        self.pub_gripper.publish(msg)

    def publish_torso(self, yaw: float):
        """허리 yaw 관절각을 /torso_controller/command 토픽으로 전송."""
        msg      = Float64MultiArray()
        msg.data = [float(yaw)]
        self.pub_torso.publish(msg)

    def publish_hand(self, data: list):
        msg      = Float64MultiArray()
        msg.data = data
        self.pub_hand.publish(msg)

        msg2      = Float32MultiArray()
        msg2.data = [float(x) for x in data]
        self.pub_amazing_hand.publish(msg2)