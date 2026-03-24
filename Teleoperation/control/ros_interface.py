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
from sensor_msgs.msg import Image, JointState
import pinocchio as pin

import config

from std_msgs.msg import Float64MultiArray, Float32MultiArray  # Float32MultiArray 추가


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
        }

        rospy.init_node('teleop_ik', disable_signals=True)

        self.pub_arm  = rospy.Publisher(config.TOPIC_ARM,  Float64MultiArray, queue_size=10)
        self.pub_hand = rospy.Publisher(config.TOPIC_HAND, Float64MultiArray, queue_size=10)

        self.pub_amazing_hand = rospy.Publisher('/amazing_hand/finger_angles', Float32MultiArray, queue_size=10)

        # 목 전용 publisher (젯슨 neck_dynamixel_node 수신용)
        self.pub_neck = rospy.Publisher('/neck_controller/command', Float64MultiArray, queue_size=1)

        rospy.Subscriber(config.TOPIC_CAMERA,       Image,      self._camera_callback)
        rospy.Subscriber(config.TOPIC_JOINT_STATES, JointState, self._joint_state_callback)

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

            # 버퍼 초기화 (최초 1회)
            if self._draw_buffer is None:
                self._draw_buffer = np.full_like(self.image_array, BG_COLOR)

            self._draw_buffer[:] = BG_COLOR
            self._draw_overlay(self._draw_buffer)
            with self._buf_lock:
                np.copyto(self.image_array, self._draw_buffer)

    # ── 카메라 콜백 ────────────────────────────────────────
    def _camera_callback(self, msg: Image):
        if self.video_playing:
            return   # 영상 재생 중에는 카메라 프레임이 영상을 덮어쓰지 못하도록 차단
        try:
            self._last_camera_time = time.time()   # 카메라 활성 시각 갱신
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            frame = cv2.resize(frame, (1280, 720))

            # 더블 버퍼 초기화 (최초 1회)
            if self._draw_buffer is None:
                self._draw_buffer = np.zeros_like(self.image_array)

            # 버퍼에 프레임 + 오버레이를 완전히 그린 후
            self._draw_buffer[:, :1280, :] = frame
            self._draw_buffer[:, 1280:, :] = frame
            self._draw_overlay(self._draw_buffer)

            # 완성된 프레임을 한 번에 복사 → 깜박임 방지
            with self._buf_lock:
                np.copyto(self.image_array, self._draw_buffer)

        except Exception as e:
            print(f"카메라 오류: {e}")

    # ── 오버레이 그리기 ────────────────────────────────────
    def _draw_overlay(self, img):
        """카메라 영상 위에 상태·손 위치 오버레이를 그린다 (양쪽 눈 동일)."""
        ov    = self.overlay
        state = ov.get('state', 'WAITING_QUEST')
        color = self._STATE_COLORS.get(state, (200, 200, 200))
        W     = 1280   # 한쪽 눈 너비

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

    def _draw_progress_bar(self, img, x, y, w, h, frac, color):
        """진행 바: 배경(회색) 위에 채워진 사각형."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
        filled = int(w * min(max(frac, 0.0), 1.0))
        if filled > 0:
            cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)

    def _draw_hand_info(self, img, x0, W, ov):
        l_err    = ov.get('l_err')
        r_err    = ov.get('r_err')
        l_joints = ov.get('l_joints', [])
        r_joints = ov.get('r_joints', [])
        F = cv2.FONT_HERSHEY_SIMPLEX

        def err_color(e):
            if e is None:  return (160, 160, 160)
            if e < 0.05:   return (60, 220, 60)
            if e < 0.15:   return (0, 165, 255)
            return             (50,  50, 255)

        img[550:, x0:x0 + W] = (img[550:, x0:x0 + W] * 0.45).astype(np.uint8)

        # IK 오차 (오른쪽 정렬)
        lc  = err_color(l_err)
        ltx = f"L err: {l_err*100:.1f}cm" if l_err is not None else "L err: ---"
        self._put_text_right(img, ltx, 580, x0, W, F, 0.8, lc, 2)

        rc  = err_color(r_err)
        rtx = f"R err: {r_err*100:.1f}cm" if r_err is not None else "R err: ---"
        self._put_text_right(img, rtx, 610, x0, W, F, 0.8, rc, 2)

        # 관절각 (오른쪽 정렬)
        joint_names = ["SP", "SR", "SY", "EP", "WY"]
        if l_joints:
            header = "L:  " + "  ".join(f"{n:>6}" for n in joint_names)
            values = "    " + "  ".join(f"{v:>6.2f}" for v in l_joints)
            self._put_text_right(img, header, 640, x0, W, F, 0.55, (120, 120, 120), 1)
            self._put_text_right(img, values, 660, x0, W, F, 0.55, (180, 180, 180), 1)
        if r_joints:
            header = "R:  " + "  ".join(f"{n:>6}" for n in joint_names)
            values = "    " + "  ".join(f"{v:>6.2f}" for v in r_joints)
            self._put_text_right(img, header, 690, x0, W, F, 0.55, (120, 120, 120), 1)
            self._put_text_right(img, values, 710, x0, W, F, 0.55, (180, 180, 180), 1)

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
        for name, val in zip(self._current_joint_state.name,
                             self._current_joint_state.position):
            if self.model.existJointName(name):
                jid        = self.model.getJointId(name)
                idx        = self.model.joints[jid].idx_q
                q_cur[idx] = val
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
        msg      = Float64MultiArray()
        msg.data = data
        self.pub_arm.publish(msg)

    def publish_neck(self, yaw: float, pitch: float):
        """목 관절각을 /neck_controller/command 토픽으로 전송 (젯슨 수신용)."""
        msg      = Float64MultiArray()
        msg.data = [float(yaw), float(pitch)]
        self.pub_neck.publish(msg)

    def publish_hand(self, data: list):
        msg      = Float64MultiArray()
        msg.data = data
        self.pub_hand.publish(msg)

        msg2      = Float32MultiArray()
        msg2.data = [float(x) for x in data]
        self.pub_amazing_hand.publish(msg2)