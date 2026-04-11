"""
HR_teleop.py  (메인)
─────────────────────────────────────────────────────────────
VR 텔레오퍼레이션 메인 상태 머신.
실행: python3 HR_teleop.py

상태 흐름:
  WAITING_QUEST → CALIBRATING → SYNCING → TELEOP
                                    ↑         ↓
                                 FREEZE ←────┘
"""
import sys, os, json, time, threading
from multiprocessing import shared_memory, Queue, Event
from enum import Enum, auto

import numpy as np
import pinocchio as pin
import rospy
from scipy.spatial.transform import Rotation as Rot

import config
sys.path.insert(0, os.path.join(config.TELEVISION_DIR, 'teleop'))
sys.path.insert(0, config.CURRENT_DIR)
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from TeleVision import OpenTeleVision

from robot_model  import (build_robot_model, compute_ik,
                           extract_wrist_twist_z, calc_target_from_calib,
                           robot_to_quest, robot_dir_to_quest, fk_palm_pose)
from motion_utils import (EMAFilter, make_filters, beep,
                           publish_smooth_move, publish_init, publish_fin)
from ros_interface import RosInterface
from finger_mapping import (build_hand_cmd, is_landmark_valid, FingerEMAFilter,
                            FINGER_ANGLE_POINTS, _flex_angle)
from quest_video import play_video_to_quest


class TeleopState(Enum):
    WAITING_QUEST = auto()
    CALIBRATING   = auto()
    SYNCING       = auto()
    FREEZE        = auto()
    TELEOP        = auto()


# ══════════════════════════════════════════════════════════════
# TeleopController
# ══════════════════════════════════════════════════════════════

class TeleopController:
    """
    VR 텔레오퍼레이션 전체 상태 머신을 캡슐화합니다.

    run() 을 호출하면 메인 루프가 시작됩니다.
    각 상태는 _handle_<state>() 메서드로 분리되어 있습니다:
      - _handle_waiting()      : WAITING_QUEST
      - _handle_calibrating()  : CALIBRATING
      - _handle_freeze()       : FREEZE
      - _handle_syncing()      : SYNCING
      - _handle_teleop()       : TELEOP (팔/목)
      - _handle_fingers()      : TELEOP (손가락, 항상 별도 호출)
    """

    def __init__(self):
        self._init_model()
        self._init_vision()
        self._init_ros()
        self._init_filters()
        self._load_calibration()
        self._init_state()
        self._print_header()
        self._init_intro_video()

    # ──────────────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────────────

    def _init_model(self):
        """피노키오 모델 + 손바닥 프레임 사전계산."""
        self.model, self.data, ids, self.q_init, robot_init = build_robot_model()
        self.joint_ids    = ids['joint_ids']
        self.L_palm_id    = ids['L_palm']
        self.R_palm_id    = ids['R_palm']
        self.L_joint_mask = ids['L_joint_mask']
        self.R_joint_mask = ids['R_joint_mask']
        self.robot_L_init = robot_init['L']
        self.robot_R_init = robot_init['R']

        # CALIB/INIT 자세에서 손바닥 pose 사전계산 (구체 오버레이용)
        self.robot_L_calib_rot, self.robot_L_calib_pos = fk_palm_pose(self.model, self.data, self.L_palm_id, config.CALIB_POS)
        self.robot_R_calib_rot, self.robot_R_calib_pos = fk_palm_pose(self.model, self.data, self.R_palm_id, config.CALIB_POS)
        self.robot_L_init_rot,  self.robot_L_init_pos  = fk_palm_pose(self.model, self.data, self.L_palm_id, config.INIT_POS)
        self.robot_R_init_rot,  self.robot_R_init_pos  = fk_palm_pose(self.model, self.data, self.R_palm_id, config.INIT_POS)

        # WAITING 구체 방향 (위치는 calib 로드 후 _update_waiting_spheres()에서 갱신)
        self.WAITING_L_POS = None
        self.WAITING_R_POS = None
        self.WAITING_L_DIR = robot_dir_to_quest(self.robot_L_init_rot, 'L')
        self.WAITING_R_DIR = robot_dir_to_quest(self.robot_R_init_rot, 'R')

        # CALIB 구체 위치·방향 (신체 자세 기반 하드코딩)
        self.CALIB_L_POS = np.array([-0.25, 0.9583, -0.50])
        self.CALIB_R_POS = np.array([ 0.25, 0.9583, -0.50])
        self.CALIB_L_DIR = np.array([ 0.9397, -0.3420,  0.0000])
        self.CALIB_R_DIR = np.array([-0.9397, -0.3420,  0.0000])

    def _update_waiting_spheres(self):
        """calib 로드 후 WAITING 구체 위치를 Quest 좌표계로 계산."""
        self.WAITING_L_POS = robot_to_quest(self.robot_L_init_pos, self.robot_L_calib_pos, self.quest_L_init)
        self.WAITING_R_POS = robot_to_quest(self.robot_R_init_pos, self.robot_R_calib_pos, self.quest_R_init)

    def _init_vision(self):
        """OpenTeleVision + 공유 메모리 초기화."""
        RES = (720, 1280)
        self.shm         = shared_memory.SharedMemory(create=True, size=RES[0] * RES[1] * 3 * 2)
        self.image_array = np.ndarray((RES[0], RES[1] * 2, 3), dtype=np.uint8, buffer=self.shm.buf)
        self.image_array[:] = 128
        self.image_queue      = Queue()
        self.toggle_streaming = Event()
        self.tv = OpenTeleVision(RES, self.shm.name, self.image_queue, self.toggle_streaming,
                                 stream_mode="image",
                                 cert_file=config.CERT_FILE,
                                 key_file=config.KEY_FILE,
                                 ngrok=False)

    def _init_ros(self):
        """ROS 인터페이스 초기화."""
        self.ros = RosInterface(self.model, self.q_init, self.image_array)

    def _init_filters(self):
        """EMA 필터 + 손가락 필터 초기화."""
        filters = make_filters()
        self.arm_filter         = filters['arm']
        self.wrist_filter_l     = filters['wrist_l']
        self.wrist_filter_r     = filters['wrist_r']
        self.quest_pos_filter_l = filters['quest_pos_l']
        self.quest_pos_filter_r = filters['quest_pos_r']
        self.neck_filter        = filters['neck']
        self.finger_filter      = FingerEMAFilter(alpha=config.EMA_FINGER, n=16)
        self.FINGER_NEUTRAL     = [0.0] * 16

    def _load_calibration(self):
        """calib.json 로드. 없으면 USE_ARM=False 시 zeros로 초기화."""
        self.quest_L_init           = None
        self.quest_R_init           = None
        self.quest_neck_rot_init    = np.eye(3)
        self.quest_L_wrist_rot_init = np.eye(3)
        self.quest_R_wrist_rot_init = np.eye(3)
        self.calibrated             = False

        if os.path.exists(config.CALIB_PATH):
            with open(config.CALIB_PATH) as f:
                cd = json.load(f)
            self.quest_L_init           = np.array(cd['quest_L_init'])
            self.quest_R_init           = np.array(cd['quest_R_init'])
            self.quest_neck_rot_init    = np.array(cd.get('quest_neck_rot_init',    np.eye(3).tolist()))
            self.quest_L_wrist_rot_init = np.array(cd.get('quest_L_wrist_rot_init', np.eye(3).tolist()))
            self.quest_R_wrist_rot_init = np.array(cd.get('quest_R_wrist_rot_init', np.eye(3).tolist()))
            self.calibrated = True
            self._update_waiting_spheres()
            print(f"[INIT] Calibration loaded  L{self.quest_L_init.round(3)}  R{self.quest_R_init.round(3)}")
        elif not config.USE_ARM:
            self.quest_L_init = np.zeros(3)
            self.quest_R_init = np.zeros(3)
            self.calibrated   = True
            print("[INIT] USE_ARM=False, skipping arm calibration")
        else:
            print("[INIT] No calibration file found. Connect Quest and extend arms forward.")

    def _init_state(self):
        """상태 머신 변수 + 초기 q 초기화."""
        q = self.ros.wait_for_joint_states()
        self.arm_filter.reset(q)
        self.q                    = q
        self.current_q_for_smooth = [float(q[idx]) for idx in self.joint_ids]
        self.q_ref_current        = q.copy()
        self.cmd_data             = self.current_q_for_smooth.copy()

        # 초기 FK → 오버레이 초기값
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        self.ros.overlay.update({
            'l_actual': self.data.oMf[self.L_palm_id].translation.tolist(),
            'r_actual': self.data.oMf[self.R_palm_id].translation.tolist(),
        })

        # 상태 머신
        self.teleop_state        = TeleopState.WAITING_QUEST
        self.tracking_lost       = False
        self._waiting_printed    = False
        self._first_teleop_start = True
        self._countdown_start    = None

        # 캘리브레이션 수집
        self.moved_to_init       = False
        self._calib_wait_start   = None
        self.calib_samples_L     = []
        self.calib_samples_R     = []
        self.calib_samples_L_rot = []
        self.calib_samples_R_rot = []

        # SYNCING
        self.sync_target_q   = None
        self.sync_start_q    = None
        self.sync_start_time = None
        self.l_sync_target   = None
        self.r_sync_target   = None

        # FREEZE
        self.freeze_start_time = None

        # 현재 프레임 데이터
        self.l_raw    = np.zeros(3)
        self.r_raw    = np.zeros(3)
        self.left_mat = np.eye(4)
        self.right_mat= np.eye(4)
        self.prev_l_raw = None
        self.prev_r_raw = None

        # 목/손목 (TELEOP에서 계산)
        self.neck_yaw    = 0.0
        self.neck_pitch  = 0.0
        self.l_wrist_yaw = 0.0
        self.r_wrist_yaw = 0.0
        self.l_err       = 0.0
        self.r_err       = 0.0

        # AA 오프셋 (TELEOP 시작 시 캘리브, AA채널 짝수 인덱스 0,2,4,6,8,10,12,14)
        self.aa_offset = np.zeros(16)

        # Hz 모니터링
        self._prev_l         = np.zeros(3)
        self._prev_r         = np.zeros(3)
        self._l_update_count = 0
        self._r_update_count = 0
        self._frame_count    = 0
        self._last_status_time = 0.0
        self.STATUS_INTERVAL   = 0.5
        self.loop_start        = 0.0

    def _print_header(self):
        print("\n" + "=" * 44)
        print("  HR Teleop")
        fe_s = "ON" if config.USE_FINGER_FE else "OFF"
        aa_s = "ON" if config.USE_FINGER_AA else "OFF"
        nk_s = "ON" if config.USE_NECK else ("TRACK" if config.USE_NECK_TRACK else "OFF")
        print(f"  ARM={str(config.USE_ARM).upper()}  FE={fe_s}  AA={aa_s}  NECK={nk_s}")
        print("=" * 44 + "\n")

    def _init_intro_video(self):
        self.video_done     = threading.Event()
        self._video_started = False
        self._use_video     = config.USE_INTRO_VIDEO and bool(config.VIDEO_PATH)
        if not self._use_video:
            if config.USE_INTRO_VIDEO and not config.VIDEO_PATH:
                print("[video] USE_INTRO_VIDEO=True but VIDEO_PATH is empty, skipping")
            self.video_done.set()

    # ──────────────────────────────────────────────────────────
    # 메인 루프
    # ──────────────────────────────────────────────────────────

    def run(self):
        try:
            while not rospy.is_shutdown():
                self.loop_start = time.time()
                self._update_hand_data()

                if self._check_tracking_lost():
                    continue

                self._waiting_printed = False

                if self.teleop_state == TeleopState.WAITING_QUEST:
                    if self._handle_waiting():
                        continue

                if self.teleop_state == TeleopState.CALIBRATING:
                    if self._handle_calibrating():
                        continue

                self._check_jump()

                if config.USE_ARM:
                    self._compute_neck_wrist()

                if self.teleop_state == TeleopState.FREEZE:
                    if self._handle_freeze():
                        continue

                if self.teleop_state == TeleopState.SYNCING:
                    if self._handle_syncing():
                        continue

                if self.teleop_state == TeleopState.TELEOP:
                    if self._handle_teleop():
                        continue
                    self._handle_fingers()
                    self._update_status()

        except KeyboardInterrupt:
            print("\n[Interrupt] Ctrl+C received, shutting down...")
        finally:
            self._shutdown()

    # ──────────────────────────────────────────────────────────
    # 루프 공통: 입력 수집, 트래킹 소실, 점프 감지, 목/손목 계산
    # ──────────────────────────────────────────────────────────

    def _update_hand_data(self):
        """Quest에서 현재 프레임 손 데이터를 읽고 Hz를 모니터링."""
        self.left_mat  = self.tv.left_hand
        self.right_mat = self.tv.right_hand
        self.l_raw     = self.left_mat[:3, 3]
        self.r_raw     = self.right_mat[:3, 3]

        self._frame_count += 1
        if not np.array_equal(self.l_raw, self._prev_l):
            self._l_update_count += 1
            self._prev_l = self.l_raw.copy()
        if not np.array_equal(self.r_raw, self._prev_r):
            self._r_update_count += 1
            self._prev_r = self.r_raw.copy()
        if self._frame_count % 50 == 0:
            print(f"[Quest] {self.tv.hand_hz:.1f}Hz | update_rate={100*self._l_update_count/self._frame_count:.0f}%")

    def _check_tracking_lost(self) -> bool:
        """트래킹 소실(r_raw == 0) 감지. True 반환 시 loop continue."""
        if not (config.USE_ARM and np.allclose(self.r_raw, 0)):
            return False

        if self.teleop_state == TeleopState.TELEOP:
            print("[WARN] Tracking lost -> FREEZE")
            self.teleop_state      = TeleopState.FREEZE
            self.freeze_start_time = time.time()
            self.sync_target_q     = None
            self.tracking_lost     = True
            beep('warn')
        elif self.teleop_state == TeleopState.FREEZE:
            pass
        elif self.teleop_state not in (TeleopState.WAITING_QUEST,):
            print("[WARN] Tracking lost -> WAITING")
            self.teleop_state     = TeleopState.WAITING_QUEST
            self.tracking_lost    = True
            self._waiting_printed = False
            self._countdown_start = None

        self.ros.overlay.update({'state': self.teleop_state.name, 'countdown': -1})
        if not self._waiting_printed:
            print("[WAIT] Waiting for Quest connection...")
            self._waiting_printed = True
        time.sleep(1.0 / config.CONTROL_HZ)
        return True

    def _check_jump(self):
        """TELEOP 중 손 위치 점프 감지 → FREEZE 전환. prev_raw는 항상 갱신."""
        if (config.USE_ARM
                and self.teleop_state == TeleopState.TELEOP
                and self.prev_r_raw is not None):
            r_jump = np.linalg.norm(self.r_raw - self.prev_r_raw)
            l_jump = np.linalg.norm(self.l_raw - self.prev_l_raw) if self.prev_l_raw is not None else 0
            if r_jump > config.JUMP_THRESHOLD or l_jump > config.JUMP_THRESHOLD:
                print(f"[WARN] Position jump  R={r_jump:.3f}m L={l_jump:.3f}m -> FREEZE")
                self.teleop_state      = TeleopState.FREEZE
                self.freeze_start_time = time.time()
                self.sync_target_q     = None
                self.tracking_lost     = True
                beep('warn')

        self.prev_r_raw = self.r_raw.copy()
        self.prev_l_raw = self.l_raw.copy()

    def _compute_neck_wrist(self):
        """목 yaw/pitch + 손목 yaw를 self에 저장. 이후 각 상태 핸들러에서 사용."""
        head = self.tv.head_matrix
        if not np.allclose(head, 0):
            head_rot = head[:3, :3]
            R_rel    = self.quest_neck_rot_init.T @ head_rot
            qx, qy, qz, qw = Rot.from_matrix(R_rel).as_quat()
            if qw < 0:
                qx, qy, qz, qw = -qx, -qy, -qz, -qw
            self.neck_yaw   = np.clip(config.NECK_SCALE *  2.0 * np.arctan2(qy, qw),
                                      self.model.lowerPositionLimit[self.joint_ids[10]],
                                      self.model.upperPositionLimit[self.joint_ids[10]])
            self.neck_pitch = np.clip(config.NECK_SCALE * -2.0 * np.arctan2(qx, qw),
                                      self.model.lowerPositionLimit[self.joint_ids[11]],
                                      self.model.upperPositionLimit[self.joint_ids[11]])
        else:
            self.neck_yaw = self.neck_pitch = 0.0

        r_wrist_delta    = extract_wrist_twist_z(self.right_mat[:3, :3], self.quest_R_wrist_rot_init)
        l_wrist_delta    = extract_wrist_twist_z(self.left_mat[:3,  :3], self.quest_L_wrist_rot_init)
        self.r_wrist_yaw = np.clip(config.CALIB_POS[9] + config.WRIST_SCALE * r_wrist_delta,
                                   self.model.lowerPositionLimit[self.joint_ids[9]],
                                   self.model.upperPositionLimit[self.joint_ids[9]])
        self.l_wrist_yaw = np.clip(config.CALIB_POS[4] + config.WRIST_SCALE * l_wrist_delta,
                                   self.model.lowerPositionLimit[self.joint_ids[4]],
                                   self.model.upperPositionLimit[self.joint_ids[4]])

    # ──────────────────────────────────────────────────────────
    # 상태 핸들러
    # ──────────────────────────────────────────────────────────

    def _handle_waiting(self) -> bool:
        """WAITING_QUEST 상태. True = loop continue."""
        # 인트로 영상 (Quest 연결 시 딱 한 번 시작)
        if self._use_video and not self._video_started and not self.video_done.is_set():
            self._video_started = True
            if os.path.exists(config.VIDEO_PATH):
                threading.Thread(
                    target=play_video_to_quest,
                    args=(self.image_array, config.VIDEO_PATH),
                    kwargs={"ros": self.ros, "done_event": self.video_done},
                    daemon=True,
                ).start()
                print("[video] Quest connected, starting video...")
            else:
                print(f"[video] File not found, skipping: {config.VIDEO_PATH}")
                self.video_done.set()

        if not self.video_done.is_set():
            time.sleep(1.0 / config.CONTROL_HZ)
            return True

        # 카운트다운
        if self._countdown_start is None:
            self._countdown_start = time.time()
            print(f"[WAIT] Quest connected. Starting in {int(config.TELEOP_START_DELAY)}s...")
            beep('teleop_start')

        elapsed_cd  = time.time() - self._countdown_start
        remaining_s = config.TELEOP_START_DELAY - elapsed_cd
        countdown_n = max(int(remaining_s) + 1, 0)

        q_cur = self.ros.get_current_q()
        pin.forwardKinematics(self.model, self.data, q_cur)
        pin.updateFramePlacements(self.model, self.data)
        self.ros.overlay.update({
            'state':     'WAITING_QUEST',
            'countdown':  countdown_n,
            'l_actual':   self.data.oMf[self.L_palm_id].translation.tolist(),
            'r_actual':   self.data.oMf[self.R_palm_id].translation.tolist(),
        })

        if self.WAITING_L_POS is not None:
            self.tv._teleop_active.value = 1 if config.USE_SPHERE else 0
            self.tv.l_palm_quest = self.WAITING_L_POS
            self.tv.r_palm_quest = self.WAITING_R_POS
            self.tv.l_palm_dir   = self.WAITING_L_DIR
            self.tv.r_palm_dir   = self.WAITING_R_DIR

        if elapsed_cd < config.TELEOP_START_DELAY:
            time.sleep(1.0 / config.CONTROL_HZ)
            return True

        # 카운트다운 완료 → 다음 상태 전환
        self._countdown_start = None
        self.ros.overlay['countdown'] = 0
        if not self.calibrated and config.USE_ARM:
            print("[STATE] -> CALIBRATING  (extend arms forward)")
            self.teleop_state = TeleopState.CALIBRATING
            beep('calib_start')
        else:
            print("[STATE] -> SYNCING")
            self.teleop_state = TeleopState.SYNCING
        return False

    def _handle_calibrating(self) -> bool:
        """CALIBRATING 상태. True = loop continue."""
        if not self.moved_to_init:
            publish_init(self.ros.pub_arm, self.current_q_for_smooth)
            self.moved_to_init        = True
            self.current_q_for_smooth = [float(self.q_init[idx]) for idx in self.joint_ids]
            self._calib_wait_start    = time.time()

        # 3초 대기: 사용자가 손을 캘리브 위치에 맞출 시간
        if self._calib_wait_start is not None:
            wait_elapsed = time.time() - self._calib_wait_start
            self._update_calib_overlay(calib_wait=max(3.0 - wait_elapsed, 0.0), n=0)
            if wait_elapsed < 3.0:
                time.sleep(1.0 / config.CONTROL_HZ)
                return True
            self._calib_wait_start = None

        # 샘플 수집
        if not np.allclose(self.l_raw, 0) and not np.allclose(self.r_raw, 0):
            self.calib_samples_R.append(self.r_raw.copy())
            self.calib_samples_L.append(self.l_raw.copy())
            self.calib_samples_R_rot.append(self.right_mat[:3, :3].copy())
            self.calib_samples_L_rot.append(self.left_mat[:3,  :3].copy())

        n = len(self.calib_samples_R)
        self._update_calib_overlay(calib_wait=0.0, n=n)

        if time.time() - self._last_status_time > self.STATUS_INTERVAL:
            print(f"[CALIB] Collecting samples ({n}/{config.CALIB_COUNT})")
            self._last_status_time = time.time()

        if n >= config.CALIB_COUNT and len(self.calib_samples_L) >= config.CALIB_COUNT:
            self._finish_calibration()

        time.sleep(1.0 / config.CONTROL_HZ)
        return True

    def _update_calib_overlay(self, calib_wait: float, n: int):
        self.ros.overlay.update({
            'state':      'CALIBRATING',
            'calib_wait':  calib_wait,
            'calib_n':     n,
            'calib_total': config.CALIB_COUNT,
        })
        self.tv._teleop_active.value = 1 if config.USE_SPHERE else 0
        self.tv.l_palm_quest = self.CALIB_L_POS.copy()
        self.tv.r_palm_quest = self.CALIB_R_POS.copy()
        self.tv.l_palm_dir   = self.CALIB_L_DIR.copy()
        self.tv.r_palm_dir   = self.CALIB_R_DIR.copy()

    def _finish_calibration(self):
        """캘리브레이션 완료: 평균 계산 → calib.json 저장 → SYNCING 전환."""
        self.quest_R_init           = np.mean(self.calib_samples_R, axis=0)
        self.quest_L_init           = np.mean(self.calib_samples_L, axis=0)
        self.quest_R_wrist_rot_init = self.calib_samples_R_rot[-1]
        self.quest_L_wrist_rot_init = self.calib_samples_L_rot[-1]
        head_rot                    = self.tv.head_matrix[:3, :3]
        self.quest_neck_rot_init    = head_rot.copy()
        self.calibrated             = True

        calib_data = {
            'quest_L_init':           self.quest_L_init.tolist(),
            'quest_R_init':           self.quest_R_init.tolist(),
            'quest_neck_yaw_init':    float(np.arctan2(-head_rot[2, 0],
                                          np.sqrt(head_rot[2, 1]**2 + head_rot[2, 2]**2))),
            'quest_neck_pitch_init':  float(np.arctan2(head_rot[2, 1], head_rot[2, 2])),
            'quest_L_wrist_rot_init': self.quest_L_wrist_rot_init.tolist(),
            'quest_R_wrist_rot_init': self.quest_R_wrist_rot_init.tolist(),
            'quest_neck_rot_init':    self.quest_neck_rot_init.tolist(),
        }
        with open(config.CALIB_PATH, 'w') as f:
            json.dump(calib_data, f)

        print(f"[CALIB] Done  L{self.quest_L_init.round(3)}  R{self.quest_R_init.round(3)}")
        beep('calib_done')
        print("[STATE] -> SYNCING")
        self.teleop_state = TeleopState.SYNCING

    def _handle_freeze(self) -> bool:
        """FREEZE 상태. True = loop continue."""
        elapsed   = time.time() - self.freeze_start_time
        remaining = config.FREEZE_DURATION - elapsed

        if time.time() - self._last_status_time > self.STATUS_INTERVAL:
            print(f"[FREEZE] Resuming in {remaining:.1f}s...")
            self._last_status_time = time.time()
        self.ros.overlay.update({'state': 'FREEZE', 'freeze_remaining': max(remaining, 0.0)})

        if elapsed >= config.FREEZE_DURATION:
            print("[STATE] FREEZE released -> SYNCING")
            self.teleop_state  = TeleopState.SYNCING
            self.sync_target_q = None
            return False

        self.ros.publish_arm(self.current_q_for_smooth)
        self.ros.publish_hand(self.FINGER_NEUTRAL)
        time.sleep(1.0 / config.CONTROL_HZ)
        return True

    def _handle_syncing(self) -> bool:
        """SYNCING 상태. True = loop continue (USE_ARM=True면 항상 True)."""
        if not config.USE_ARM:
            print("[STATE] USE_ARM=False, skipping SYNCING -> TELEOP")
            self.teleop_state = TeleopState.TELEOP
            self._calibrate_aa_offset()   # TELEOP 시작 시 AA 기준점 설정
            beep('teleop_start' if self._first_teleop_start else 'sync_done')
            self._first_teleop_start = False
            return False  # fall through to TELEOP this frame

        if self.sync_target_q is None or self.tracking_lost:
            self._init_sync_target()

        elapsed_sync    = time.time() - self.sync_start_time
        fraction        = min(elapsed_sync / config.SYNC_DURATION, 1.0)
        fraction_smooth = fraction * fraction * (3 - 2 * fraction)
        cmd_data        = [s + (t - s) * fraction_smooth
                           for s, t in zip(self.sync_start_q, self.sync_target_q)]

        q_actual  = self.ros.get_current_q()
        joint_err = max(abs(float(q_actual[idx]) - t)
                        for idx, t in zip(self.joint_ids[:8], self.sync_target_q[:8]))
        pin.forwardKinematics(self.model, self.data, q_actual)
        pin.updateFramePlacements(self.model, self.data)
        l_pos_err = np.linalg.norm(self.data.oMf[self.L_palm_id].translation - self.l_sync_target)
        r_pos_err = np.linalg.norm(self.data.oMf[self.R_palm_id].translation - self.r_sync_target)

        if time.time() - self._last_status_time > self.STATUS_INTERVAL:
            print(f"[SYNC] {fraction*100:.0f}%  joint={joint_err:.3f}rad  L={l_pos_err:.3f}m R={r_pos_err:.3f}m")
            self._last_status_time = time.time()

        self.ros.overlay.update({
            'state':        'SYNCING',
            'sync_elapsed': elapsed_sync,
            'sync_timeout': config.SYNC_TIMEOUT,
        })

        sync_done = (
            (joint_err  < config.SYNC_JOINT_THRESH    and
             l_pos_err  < config.SYNC_POSITION_THRESH and
             r_pos_err  < config.SYNC_POSITION_THRESH)
            or fraction >= 1.0
            or elapsed_sync >= config.SYNC_TIMEOUT
        )
        if sync_done:
            print("[STATE] Sync complete -> TELEOP" if fraction < 1.0
                  else "[STATE] Sync timeout, forcing -> TELEOP")
            self.q             = self.ros.get_current_q()
            self.q_ref_current = self.q.copy()
            self.arm_filter.reset(self.q)
            self.sync_target_q = None
            self.teleop_state  = TeleopState.TELEOP
            self._calibrate_aa_offset()   # TELEOP 시작 시 AA 기준점 설정
            beep('teleop_start' if self._first_teleop_start else 'sync_done')
            self._first_teleop_start = False

        if np.any(np.isnan(cmd_data)):
            time.sleep(1.0 / config.CONTROL_HZ)
            return True

        self.current_q_for_smooth = cmd_data.copy()
        self.ros.publish_arm(cmd_data)
        self.ros.publish_hand(self.FINGER_NEUTRAL)

        elapsed = time.time() - self.loop_start
        time.sleep(max(0, 1.0 / config.CONTROL_HZ - elapsed))
        return True  # USE_ARM=True면 SYNCING 후 항상 next iteration

    def _init_sync_target(self):
        """SYNCING 시작 시 IK 목표 자세를 계산하고 필터를 리셋."""
        self.q             = self.ros.get_current_q()
        self.q_ref_current = self.q.copy()

        self.l_sync_target, self.r_sync_target = calc_target_from_calib(
            self.l_raw, self.r_raw,
            self.quest_L_init, self.quest_R_init,
            self.robot_L_init, self.robot_R_init,
        )
        q_sync = compute_ik(self.model, self.data, self.L_palm_id, self.l_sync_target, self.q,
                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.L_joint_mask)
        q_sync = compute_ik(self.model, self.data, self.R_palm_id, self.r_sync_target, q_sync,
                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.R_joint_mask)

        q_sync_cmd       = [float(q_sync[idx]) for idx in self.joint_ids]
        q_sync_cmd[4]    = float(self.l_wrist_yaw)
        q_sync_cmd[9]    = float(self.r_wrist_yaw)
        q_sync_cmd[10]   = float(self.neck_yaw)   if config.USE_NECK else 0.0
        q_sync_cmd[11]   = float(self.neck_pitch) if config.USE_NECK else 0.0

        self.sync_target_q   = q_sync_cmd
        self.sync_start_q    = [float(self.q[idx]) for idx in self.joint_ids]
        self.sync_start_time = time.time()
        self.tracking_lost   = False

        self.arm_filter.reset(self.q)
        self.wrist_filter_l.reset(np.array([float(self.l_wrist_yaw)]))
        self.wrist_filter_r.reset(np.array([float(self.r_wrist_yaw)]))
        self.quest_pos_filter_l.reset(self.l_raw)
        self.quest_pos_filter_r.reset(self.r_raw)
        self.neck_filter.reset(np.array([float(self.neck_yaw), float(self.neck_pitch)]))

        print(f"[SYNC] Target  L{self.l_sync_target.round(3)}  R{self.r_sync_target.round(3)}")

    def _handle_teleop(self) -> bool:
        """TELEOP 상태: 팔/목 IK + publish. True = NaN 발산 시 loop continue."""
        if not config.USE_ARM:
            self.cmd_data = self.current_q_for_smooth.copy()
            self.l_err = self.r_err = 0.0
            return False

        # 1. 입력 필터링 + 좌표 변환
        l_filt = self.quest_pos_filter_l.filter(self.l_raw)
        r_filt = self.quest_pos_filter_r.filter(self.r_raw)
        l_target, r_target = calc_target_from_calib(
            l_filt, r_filt,
            self.quest_L_init, self.quest_R_init,
            self.robot_L_init, self.robot_R_init,
        )

        # 2. IK
        self.q = compute_ik(self.model, self.data, self.L_palm_id, l_target, self.q,
                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.L_joint_mask)
        self.q = compute_ik(self.model, self.data, self.R_palm_id, r_target, self.q,
                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.R_joint_mask)

        # 3. FK → IK 오차 계산 (구체 오버레이 방향에도 이 data 재활용)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        l_actual  = self.data.oMf[self.L_palm_id].translation.copy()
        r_actual  = self.data.oMf[self.R_palm_id].translation.copy()
        self.l_err = float(np.linalg.norm(l_actual - l_target))
        self.r_err = float(np.linalg.norm(r_actual - r_target))

        # 4. 스무딩 + 구체 오버레이
        l_wrist_filt = self.wrist_filter_l.filter(np.array([float(self.l_wrist_yaw)]))[0]
        r_wrist_filt = self.wrist_filter_r.filter(np.array([float(self.r_wrist_yaw)]))[0]

        # 구체 방향은 손목 yaw 필터값이 반영된 q_vis로 FK를 다시 돌려야 함
        # (self.q에는 아직 wrist_filt가 반영 안 된 상태)
        q_vis = self.q.copy()
        q_vis[self.joint_ids[4]] = l_wrist_filt
        q_vis[self.joint_ids[9]] = r_wrist_filt
        pin.forwardKinematics(self.model, self.data, q_vis)
        pin.updateFramePlacements(self.model, self.data)

        self.tv._teleop_active.value = 1 if config.USE_SPHERE else 0
        self.tv.l_palm_quest = robot_to_quest(l_actual, self.robot_L_init, self.quest_L_init)
        self.tv.r_palm_quest = robot_to_quest(r_actual, self.robot_R_init, self.quest_R_init)
        self.tv.l_palm_dir   = robot_dir_to_quest(self.data.oMf[self.L_palm_id].rotation, 'L')
        self.tv.r_palm_dir   = robot_dir_to_quest(self.data.oMf[self.R_palm_id].rotation, 'R')

        # 5. 관절각 명령 조합
        self.q        = self.arm_filter.filter(self.q)
        self.cmd_data = [float(self.q[idx]) for idx in self.joint_ids]
        self.cmd_data[4] = l_wrist_filt
        self.cmd_data[9] = r_wrist_filt
        self._apply_neck(self.cmd_data)
        self._update_neck_warn(self.cmd_data)
        self.ros.publish_neck(self.cmd_data[10], self.cmd_data[11])  # 젯슨으로 목 각도 전송

        # 6. 안전 체크 (NaN/Inf → FREEZE)
        if not np.all(np.isfinite(self.cmd_data)):
            print("[WARN] IK diverged -> FREEZE")
            self.q                 = self.ros.get_current_q()
            self.arm_filter.reset(self.q)
            self.sync_target_q     = None
            self.teleop_state      = TeleopState.FREEZE
            self.freeze_start_time = time.time()
            self.tracking_lost     = True
            beep('warn')
            time.sleep(1.0 / config.CONTROL_HZ)
            return True

        # 7. Publish
        self.current_q_for_smooth = self.cmd_data.copy()
        self.ros.publish_arm(self.cmd_data)
        return False

    def _apply_neck(self, cmd_data: list):
        """목 제어 모드(USE_NECK / USE_NECK_TRACK / OFF)에 따라 cmd_data[10:12]에 값을 씀."""
        if config.USE_NECK:
            neck_filt    = self.neck_filter.filter(np.array([float(self.neck_yaw), float(self.neck_pitch)]))
            cmd_data[10] = neck_filt[0]
            cmd_data[11] = neck_filt[1]
            return

        if config.USE_NECK_TRACK:
            head_pos = self.tv.head_matrix[:3, 3]
            if (not np.allclose(head_pos, 0)
                    and not np.allclose(self.l_raw, 0)
                    and not np.allclose(self.r_raw, 0)):
                mid    = (self.l_raw + self.r_raw) * 0.5
                d      = mid - head_pos
                d_norm = np.linalg.norm(d)
                if d_norm > 0.05:
                    d /= d_norm
                    nt_yaw   = np.clip(config.NECK_SCALE * float(np.arctan2(-d[0], d[2])),
                                       self.model.lowerPositionLimit[self.joint_ids[10]],
                                       self.model.upperPositionLimit[self.joint_ids[10]])
                    nt_pitch = np.clip(config.NECK_SCALE * float(np.arctan2(-d[1], np.sqrt(d[0]**2 + d[2]**2))),
                                       self.model.lowerPositionLimit[self.joint_ids[11]],
                                       self.model.upperPositionLimit[self.joint_ids[11]])
                    nt_filt      = self.neck_filter.filter(np.array([nt_yaw, nt_pitch]))
                    cmd_data[10] = nt_filt[0]
                    cmd_data[11] = nt_filt[1]
                    return

        cmd_data[10] = 0.0
        cmd_data[11] = 0.0

    def _update_neck_warn(self, cmd_data: list):
        """목 관절이 한계의 90% 이상이면 overlay에 경고 세팅."""
        yaw_limit   = self.model.upperPositionLimit[self.joint_ids[10]]
        pitch_limit = self.model.upperPositionLimit[self.joint_ids[11]]
        THRESH = 0.9

        at_yaw   = abs(cmd_data[10]) >= THRESH * yaw_limit
        at_pitch = abs(cmd_data[11]) >= THRESH * pitch_limit

        if at_yaw and at_pitch:
            warn = 'YAW + PITCH'
        elif at_yaw:
            warn = 'YAW'
        elif at_pitch:
            warn = 'PITCH'
        else:
            warn = ''

        self.ros.overlay['neck_warn'] = warn

    def _calibrate_aa_offset(self):
        """TELEOP 시작 시 현재 손 자세를 AA 기준점(0°)으로 설정.

        현재 랜드마크에서 AA값을 계산해 self.aa_offset에 저장.
        이후 _handle_fingers()에서 raw_cmd의 AA 채널에서 이 오프셋을 뺌.
        AA 채널 인덱스: 0,2,4,6 (왼손), 8,10,12,14 (오른손).
        """
        if not config.USE_FINGER_AA:
            self.aa_offset = np.zeros(16)
            return

        left_lm  = self.tv.left_landmarks
        right_lm = self.tv.right_landmarks
        l_valid  = is_landmark_valid(left_lm)
        r_valid  = is_landmark_valid(right_lm)

        if not (l_valid or r_valid):
            self.aa_offset = np.zeros(16)
            print("[FINGER] AA offset calibration skipped (no valid landmarks)")
            return

        _left_lm  = left_lm  if l_valid else np.zeros((25, 3))
        _right_lm = right_lm if r_valid else np.zeros((25, 3))

        # AA만 계산 (FE는 오프셋 불필요)
        offset_raw = build_hand_cmd(_left_lm, _right_lm, use_fe=False, use_aa=True)

        self.aa_offset = offset_raw.copy()
        print(f"[FINGER] AA offset set: "
              f"L[{','.join(f'{np.degrees(offset_raw[i]):+.1f}°' for i in [0,2,4,6])}] "
              f"R[{','.join(f'{np.degrees(offset_raw[i]):+.1f}°' for i in [8,10,12,14])}]")

    def _handle_fingers(self):
        """TELEOP 상태에서 손가락 리타게팅 + publish."""
        if not (config.USE_FINGER_FE or config.USE_FINGER_AA):
            self.ros.publish_hand(self.FINGER_NEUTRAL)
            return

        left_lm  = self.tv.left_landmarks
        right_lm = self.tv.right_landmarks
        l_valid  = is_landmark_valid(left_lm)
        r_valid  = is_landmark_valid(right_lm)

        if config.FINGER_DEBUG and time.time() - self._last_status_time > 0.3:
            self._print_finger_debug(left_lm, right_lm, l_valid, r_valid)

        if not (l_valid or r_valid):
            self.ros.publish_hand(self.FINGER_NEUTRAL)
            return

        _left_lm  = left_lm  if l_valid else np.zeros((25, 3))
        _right_lm = right_lm if r_valid else np.zeros((25, 3))
        raw_cmd   = build_hand_cmd(_left_lm, _right_lm,
                                   use_fe=config.USE_FINGER_FE,
                                   use_aa=config.USE_FINGER_AA)
        if not l_valid: raw_cmd[:8]  = self.FINGER_NEUTRAL[:8]
        if not r_valid: raw_cmd[8:]  = self.FINGER_NEUTRAL[8:]

        # AA 오프셋 제거 (AA 채널 = 짝수 인덱스 0,2,4,6,8,10,12,14)
        if config.USE_FINGER_AA:
            for i in range(0, 16, 2):
                raw_cmd[i] = np.clip(
                    raw_cmd[i] - self.aa_offset[i],
                    -0.349, 0.349
                )

        finger_cmd = self.finger_filter.filter(raw_cmd)
        if np.all(np.isfinite(finger_cmd)):
            self.ros.publish_hand(finger_cmd.tolist())
        else:
            self.ros.publish_hand(self.FINGER_NEUTRAL)

    def _print_finger_debug(self, left_lm, right_lm, l_valid, r_valid):
        finger_names = {1: "엄지", 2: "검지", 3: "중지", 4: "약지"}
        PINKY_ACTUAL = (0, 21, 24)

        def _fmt(lm, side):
            parts = [f"{side} {finger_names[f]}:{_flex_angle(lm[p[0]], lm[p[1]], lm[p[2]]):.0f}°"
                     for f, p in FINGER_ANGLE_POINTS.items()]
            pinky = _flex_angle(lm[PINKY_ACTUAL[0]], lm[PINKY_ACTUAL[1]], lm[PINKY_ACTUAL[2]])
            parts.append(f"{side} 소지(참고):{pinky:.0f}°")
            return parts

        if l_valid: print("  [FLEX]", "  ".join(_fmt(left_lm,  "L")))
        if r_valid: print("  [FLEX]", "  ".join(_fmt(right_lm, "R")))

    def _update_status(self):
        """TELEOP 오버레이 갱신 + 터미널 상태 출력 + Hz sleep."""
        elapsed   = time.time() - self.loop_start
        loop_time = max(elapsed, 1e-6)
        time.sleep(max(0, 1.0 / config.CONTROL_HZ - elapsed))

        self.ros.overlay.update({
            'state':    'TELEOP',
            'hz':       1.0 / loop_time,
            'l_err':    self.l_err,
            'r_err':    self.r_err,
            'l_joints': self.cmd_data[:5],
            'r_joints': self.cmd_data[5:10],
        })

        if time.time() - self._last_status_time > self.STATUS_INTERVAL:
            arm_s = f"L={self.l_err*100:.1f}cm R={self.r_err*100:.1f}cm" if config.USE_ARM else "ARM=OFF"
            fe_s  = "FE=ON"  if config.USE_FINGER_FE  else "FE=OFF"
            aa_s  = "AA=ON"  if config.USE_FINGER_AA  else "AA=OFF"
            nk_s  = "NECK=ON" if config.USE_NECK else ("NECK=TRACK" if config.USE_NECK_TRACK else "NECK=OFF")
            print(f"[TELEOP] {1/loop_time:.1f}Hz | {arm_s} | {fe_s} {aa_s} | {nk_s}")
            if config.USE_ARM:
                print(f"[WRIST]  L={np.degrees(self.l_wrist_yaw):+.1f}deg  R={np.degrees(self.r_wrist_yaw):+.1f}deg")
            self._last_status_time = time.time()

    # ──────────────────────────────────────────────────────────
    # 종료
    # ──────────────────────────────────────────────────────────

    def _shutdown(self):
        publish_fin(self.ros.pub_arm,
                    current_vals=self.cmd_data,
                    duration=2.5)
        self.ros.publish_hand(self.FINGER_NEUTRAL)
        if hasattr(self, 'shm'):
            self.shm.close()
            self.shm.unlink()
        print("[DONE] Shutdown complete")


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    TeleopController().run()