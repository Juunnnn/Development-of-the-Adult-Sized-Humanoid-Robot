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
import sys, os, json, time, threading, csv, signal
from multiprocessing import shared_memory, Queue, Event
from enum import Enum, auto

import numpy as np
import pinocchio as pin
import rospy
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Empty

import config
sys.path.insert(0, os.path.join(config.TELEVISION_DIR, 'teleop'))
sys.path.insert(0, config.CURRENT_DIR)
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
from TeleVision import OpenTeleVision

from robot_model  import (build_robot_model, compute_ik,
                           apply_torso_compensation,
                           extract_wrist_twist_z, calc_target_from_calib,
                           robot_to_quest, robot_dir_to_quest, fk_palm_pose,
                           quest_dir_to_robot)
from motion_utils import (EMAFilter, make_filters, beep,
                           publish_smooth_move, publish_init, publish_fin,
                           publish_torso_fin)
from ros_interface import RosInterface
from finger_mapping import (build_hand_cmd, is_landmark_valid, FingerEMAFilter,
                            FINGER_ANGLE_POINTS, _flex_angle, compute_grip_ratio,
                            grip_debug_angles)
from quest_video import play_video_to_quest


class SyncDataLogger:
    """
    SYNCING / 트래킹 디버깅용 CSV 로거.

    실행 시작 시각(YYYYMMDD_HHMMSS)으로 파일명을 만들어
    <CURRENT_DIR>/sync_logs/ 아래에 저장합니다.
    한 줄 = 한 이벤트(또는 한 루프 프레임). 크래시/Ctrl+C에도 데이터가
    남도록 매 log() 호출마다 flush 합니다.

    사용 예:
        self.data_logger.log(self.teleop_state.name, 'SYNC_INIT',
                              l_raw=self.l_raw, r_raw=self.r_raw,
                              l_target=self.l_sync_target,
                              r_target=self.r_sync_target,
                              sync_q=q_sync_cmd,
                              note='SYNCING 시작 시점 스냅샷')
    """
    HEADER = [
        'epoch_time', 'iso_time', 'state', 'event',
        'l_raw_x', 'l_raw_y', 'l_raw_z',
        'r_raw_x', 'r_raw_y', 'r_raw_z',
        'l_target_x', 'l_target_y', 'l_target_z',
        'r_target_x', 'r_target_y', 'r_target_z',
        'sync_q',   # 12개 관절값을 ';'로 join한 문자열 (joint_ids 순서)
        'note',
    ]

    def __init__(self, log_dir=None):
        base = log_dir or getattr(config, 'CURRENT_DIR', os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base, 'sync_logs')
        os.makedirs(self.log_dir, exist_ok=True)

        fname     = time.strftime('synclog_%Y%m%d_%H%M%S.csv')
        self.path = os.path.join(self.log_dir, fname)

        self._f      = open(self.path, 'w', newline='')
        self._writer = csv.writer(self._f)
        self._writer.writerow(self.HEADER)
        self._f.flush()
        print(f"[LOG] Sync debug log -> {self.path}")

    @staticmethod
    def _xyz(v):
        if v is None:
            return ['', '', '']
        return [f'{float(x):.6f}' for x in v]

    def log(self, state, event, l_raw=None, r_raw=None,
            l_target=None, r_target=None, sync_q=None, note=''):
        now = time.time()
        row = [
            f'{now:.6f}',
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)) + f'.{int((now % 1) * 1000):03d}',
            state, event,
        ]
        row += self._xyz(l_raw)
        row += self._xyz(r_raw)
        row += self._xyz(l_target)
        row += self._xyz(r_target)
        row.append(';'.join(f'{float(v):.5f}' for v in sync_q) if sync_q is not None else '')
        row.append(note)
        try:
            self._writer.writerow(row)
            self._f.flush()
        except Exception as e:
            print(f"[LOG] write failed: {e}")

    def close(self):
        try:
            self._f.close()
            print(f"[LOG] Sync debug log closed -> {self.path}")
        except Exception:
            pass


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
        self._init_logger()
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

    def _init_logger(self):
        """날짜/시간 기반 CSV 디버그 로거 초기화 (가장 먼저 호출)."""
        self.data_logger = SyncDataLogger()

    def _init_model(self):
        """피노키오 모델 + 손바닥 프레임 사전계산."""
        self.model, self.data, ids, self.q_init, robot_init = build_robot_model()
        self.joint_ids    = ids['joint_ids']
        self.L_palm_id    = ids['L_palm']
        self.R_palm_id    = ids['R_palm']
        self.L_joint_mask = ids['L_joint_mask']
        self.R_joint_mask = ids['R_joint_mask']
        self.torso_idx    = ids['torso_idx']    # Waist_joint q 인덱스 (None이면 fixed)
        self.L_roll_idx   = ids['L_roll_idx']   # L_shoulder_roll q 인덱스
        self.R_roll_idx   = ids['R_roll_idx']   # R_shoulder_roll q 인덱스
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
        self._pub_episode_start = rospy.Publisher('/teleop/episode_start', Empty, queue_size=1)
        self._pub_episode_end   = rospy.Publisher('/teleop/episode_end',   Empty, queue_size=1)

    def _init_filters(self):
        """EMA 필터 + 손가락 필터 초기화."""
        filters = make_filters()
        self.arm_filter         = filters['arm']
        self.wrist_filter_l     = filters['wrist_l']
        self.wrist_filter_r     = filters['wrist_r']
        self.quest_pos_filter_l = filters['quest_pos_l']
        self.quest_pos_filter_r = filters['quest_pos_r']
        self.neck_filter        = filters['neck']
        self.torso_filter       = EMAFilter(alpha=config.EMA_TORSO)
        self.finger_filter      = FingerEMAFilter(alpha=config.EMA_FINGER, n=16)
        self.FINGER_NEUTRAL     = [0.0] * 16
        self.neck_cmd_last      = [config.INIT_POS[10], config.INIT_POS[11]]

        # 오른손 그리퍼(Dynamixel XM430) 전용 1채널 필터
        # config.EMA_GRIPPER가 없으면 EMA_FINGER 값으로 폴백
        self.grip_filter = FingerEMAFilter(alpha=getattr(config, 'EMA_GRIPPER', config.EMA_FINGER), n=1)

        # 전완 방향(orientation 보조 과제) 타겟용 EMA 필터
        # config.EMA_FOREARM_DIR이 아직 없으면 0.3으로 폴백 (config.py에 추가 권장)
        ema_forearm_dir           = getattr(config, 'EMA_FOREARM_DIR', 0.3)
        self.forearm_dir_filter_l = EMAFilter(alpha=ema_forearm_dir)
        self.forearm_dir_filter_r = EMAFilter(alpha=ema_forearm_dir)

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

        # 보조 싱킹 (TELEOP 진입 직후 잔여 오차를 짧게 한 번 더 보간)
        self.blend_active     = False
        self.blend_start_time = None
        self.blend_start_q    = None   # 보조 싱킹 시작 시점 실제 관절각 (12개, joint_ids 순서)

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

        # 전완 방향 타겟 (orientation 보조 과제) - 필터 reset 전 기본값
        self.l_forearm_dir_target = None
        self.r_forearm_dir_target = None

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

        self._rate = rospy.Rate(config.CONTROL_HZ)

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
                    self._handle_gripper()
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

        # 매 프레임 raw 손 위치 로깅 (WAITING/CALIBRATING/SYNCING 구간 위주 디버깅용)
        self.data_logger.log(self.teleop_state.name, 'LOOP', l_raw=self.l_raw, r_raw=self.r_raw)

        self._frame_count += 1
        if not np.array_equal(self.l_raw, self._prev_l):
            self._l_update_count += 1
            self._prev_l = self.l_raw.copy()
        if not np.array_equal(self.r_raw, self._prev_r):
            self._r_update_count += 1
            self._prev_r = self.r_raw.copy()
        if self._frame_count % 50 == 0:
            # print(f"[Quest] {self.tv.hand_hz:.1f}Hz | update_rate={100*self._l_update_count/self._frame_count:.0f}%")
            print(f"[Quest] {self.tv.hand_hz:.1f}Hz")

    def _check_tracking_lost(self) -> bool:
        """트래킹 소실(r_raw == 0) 감지. True 반환 시 loop continue."""
        # 진단용: 기존 로직이 못 잡는 케이스(왼손만 0)를 별도로 로그만 남김.
        # 동작은 바꾸지 않고 데이터만 수집 -> 가설 확인용.
        if config.USE_ARM and np.allclose(self.l_raw, 0) and not np.allclose(self.r_raw, 0):
            self.data_logger.log(self.teleop_state.name, 'L_RAW_ZERO_ONLY',
                                  l_raw=self.l_raw, r_raw=self.r_raw,
                                  note='왼손만 0 - 기존 tracking_lost 체크가 못 잡는 케이스')

        if not (config.USE_ARM and np.allclose(self.r_raw, 0)):
            return False

        self.data_logger.log(self.teleop_state.name, 'TRACK_LOST_R',
                              l_raw=self.l_raw, r_raw=self.r_raw,
                              note='r_raw≈0 감지')

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
        # else:
        #     self.neck_yaw = self.neck_pitch = 0.0

        r_wrist_delta    = extract_wrist_twist_z(self.right_mat[:3, :3], self.quest_R_wrist_rot_init)
        l_wrist_delta    = extract_wrist_twist_z(self.left_mat[:3,  :3], self.quest_L_wrist_rot_init)
        self.r_wrist_yaw = np.clip(config.CALIB_POS[9] + config.WRIST_SCALE * r_wrist_delta,
                                   self.model.lowerPositionLimit[self.joint_ids[9]],
                                   self.model.upperPositionLimit[self.joint_ids[9]])
        self.l_wrist_yaw = np.clip(config.CALIB_POS[4] + config.WRIST_SCALE * l_wrist_delta,
                                   self.model.lowerPositionLimit[self.joint_ids[4]],
                                   self.model.upperPositionLimit[self.joint_ids[4]])

    def _compute_forearm_dir_targets(self):
        """
        Quest 손 rotation 행렬에서 전완 방향(로컬축)을 뽑아 로봇 world 좌표계
        목표 방향벡터로 변환합니다. (필터링 전 raw 값 반환 - 필터링은 호출부에서)

        """
        axis_l = getattr(config, 'QUEST_FOREARM_LOCAL_AXIS_L', np.array([0.0, 0.0, -1.0]))
        axis_r = getattr(config, 'QUEST_FOREARM_LOCAL_AXIS_R', np.array([0.0, 0.0, -1.0]))
        l_dir = quest_dir_to_robot(self.left_mat[:3, :3],  axis_l)
        r_dir = quest_dir_to_robot(self.right_mat[:3, :3], axis_r)
        return l_dir, r_dir

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
            self._calibrate_aa_offset()
            beep('teleop_start' if self._first_teleop_start else 'sync_done')
            self._first_teleop_start = False
            self._pub_episode_start.publish(Empty())   # ← 추가
            return False

        if self.sync_target_q is None or self.tracking_lost:
            self._init_sync_target()

        if config.USE_NECK:
            self.sync_target_q[10] = float(self.neck_yaw)
            self.sync_target_q[11] = float(self.neck_pitch)

        elapsed_sync    = time.time() - self.sync_start_time
        fraction        = min(elapsed_sync / config.SYNC_DURATION, 1.0)
        # fraction_smooth = fraction * fraction * (3 - 2 * fraction)
        fraction_smooth = fraction**3 * (10-15*fraction + 6*fraction**2)  # quintic smoothstep
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
            fraction >= 1.0
            or elapsed_sync >= config.SYNC_TIMEOUT
        )
        if sync_done:
            print("[STATE] Sync complete -> TELEOP" if elapsed_sync < config.SYNC_TIMEOUT
                  else "[STATE] Sync timeout, forcing -> TELEOP")
            self.q             = self.ros.get_current_q()
            self.q_ref_current = self.q.copy()

            # 필터 리셋: arm_filter 외에 quest_pos/wrist/neck 필터도 현재 실제 값으로 맞춤.
            # 안 하면 SYNCING 시작 시점에 멈춰있던 prev값과 TELEOP 첫 입력값 사이에
            # 갭이 생겨 첫 프레임에 작은 점프가 섞일 수 있음.
            self.arm_filter.reset(self.q)
            self.quest_pos_filter_l.reset(self.l_raw)
            self.quest_pos_filter_r.reset(self.r_raw)
            self.wrist_filter_l.reset(np.array([float(self.l_wrist_yaw)]))
            self.wrist_filter_r.reset(np.array([float(self.r_wrist_yaw)]))
            self.neck_filter.reset(np.array([float(self.neck_yaw), float(self.neck_pitch)]))
            l_dir0, r_dir0 = self._compute_forearm_dir_targets()
            self.forearm_dir_filter_l.reset(l_dir0)
            self.forearm_dir_filter_r.reset(r_dir0)

            self.sync_target_q = None
            self.teleop_state  = TeleopState.TELEOP

            # 보조 싱킹 시작: SYNCING이 못 채운 잔여 오차(joint_err<0.1rad, pos_err<0.05m)를
            # TELEOP의 IK가 1프레임(20ms)에 메우지 않도록, SYNC_BLEND_DURATION 동안
            # "지금 실제 자세 -> IK가 원하는 자세"를 한 번 더 부드럽게 보간한다.
            self.blend_active     = True
            self.blend_start_time = time.time()
            self.blend_start_q    = [float(self.q[idx]) for idx in self.joint_ids]
            # print(f"[DEBUG] SYNC DONE: {[f'{v:.3f}' for v in self.blend_start_q]}")
            self.data_logger.log(self.teleop_state.name, 'SYNC_DONE',
                                  l_raw=self.l_raw, r_raw=self.r_raw,
                                  sync_q=self.blend_start_q,
                                  note=f'fraction={fraction:.3f} elapsed={elapsed_sync:.2f}s')

            self._calibrate_aa_offset()   # TELEOP 시작 시 AA 기준점 설정
            beep('teleop_start' if self._first_teleop_start else 'sync_done')
            self._first_teleop_start = False

        if np.any(np.isnan(cmd_data)):
            time.sleep(1.0 / config.CONTROL_HZ)
            return True

        self.current_q_for_smooth = cmd_data.copy()
        self.ros.publish_arm(cmd_data)
        self.ros.publish_hand(self.FINGER_NEUTRAL)
        self.ros.publish_neck(cmd_data[10], cmd_data[11])

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
        self.sync_start_q[10] = self.neck_cmd_last[0]   # 목만 self.q 대신 자체 추적값 사용
        self.sync_start_q[11] = self.neck_cmd_last[1]
        self.sync_start_time = time.time()
        self.tracking_lost   = False

        self.arm_filter.reset(self.q)
        self.wrist_filter_l.reset(np.array([float(self.l_wrist_yaw)]))
        self.wrist_filter_r.reset(np.array([float(self.r_wrist_yaw)]))
        self.quest_pos_filter_l.reset(self.l_raw)
        self.quest_pos_filter_r.reset(self.r_raw)
        self.neck_filter.reset(np.array([float(self.neck_yaw), float(self.neck_pitch)]))
        l_dir0, r_dir0 = self._compute_forearm_dir_targets()
        self.forearm_dir_filter_l.reset(l_dir0)
        self.forearm_dir_filter_r.reset(r_dir0)

        # print(f"[SYNC] Target  L{self.l_sync_target.round(3)}  R{self.r_sync_target.round(3)}")

        # ★ 핵심 진단 로그: SYNCING 시작 순간의 raw 손 위치와, 그로부터 계산된
        # Cartesian target 및 IK 결과(sync_target_q)를 그대로 남긴다.
        # 여기서 l_raw가 (0,0,0)에 가깝거나 l_sync_target/sync_q가 비정상적으로
        # 크면, "왼손 데이터가 아직 유효하지 않은 상태에서 SYNCING이 시작됐다"는
        # 가설이 확인되는 것.
        self.data_logger.log(self.teleop_state.name, 'SYNC_INIT',
                              l_raw=self.l_raw, r_raw=self.r_raw,
                              l_target=self.l_sync_target, r_target=self.r_sync_target,
                              sync_q=q_sync_cmd,
                              note='SYNCING 시작 시점 스냅샷')

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

        # 1-1. 전완 방향 타겟 (orientation 보조 과제 - shoulder_yaw가 이 자유도를 주로 사용)
        l_dir_raw, r_dir_raw = self._compute_forearm_dir_targets()
        l_dir_filt = self.forearm_dir_filter_l.filter(l_dir_raw)
        r_dir_filt = self.forearm_dir_filter_r.filter(r_dir_raw)
        # EMA 후에는 벡터 크기가 1에서 벗어나므로 반드시 재정규화
        self.l_forearm_dir_target = l_dir_filt / max(np.linalg.norm(l_dir_filt), 1e-6)
        self.r_forearm_dir_target = r_dir_filt / max(np.linalg.norm(r_dir_filt), 1e-6)

        # config.py에 없으면 기본값으로 폴백 (확정 전 임시 튜닝값)
        orient_weight = getattr(config, 'ORIENT_WEIGHT', 0.5)
        null_weight   = getattr(config, 'NULL_WEIGHT', 0.15)
        orient_min_iter = getattr(config, 'ORIENT_MIN_ITER', 5)
        orient_eps      = getattr(config, 'ORIENT_EPS', 0.02)
        orient_max_delta = getattr(config, 'ORIENT_MAX_DELTA', 0.05)
        orient_limit_margin = getattr(config, 'ORIENT_LIMIT_MARGIN', 0.15)

        # blend(보조 싱킹) 중에는 orientation을 끔 - SYNCING 직후 잔여오차를 부드럽게
        # 흡수하는 구간인데, orientation까지 같이 당기면 "블렌딩이 향해 가는 목표" 자체가
        # 매 프레임 계속 움직여서 의도와 다르게 느껴짐 (실측으로 확인된 문제).
        # 필터 자체는 계속 갱신해서(위 l_dir_filt/r_dir_filt), blend가 끝나는 순간
        # 차가운 값이 아니라 이미 워밍업된 값으로 바로 자연스럽게 시작하게 함.
        l_dir_for_ik = None if self.blend_active else self.l_forearm_dir_target
        r_dir_for_ik = None if self.blend_active else self.r_forearm_dir_target

        # 2. IK
        self.q = compute_ik(self.model, self.data, self.L_palm_id, l_target, self.q,
                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.L_joint_mask,
                            target_forearm_dir=l_dir_for_ik,
                            orient_weight=orient_weight, null_weight=null_weight,
                            orient_min_iter=orient_min_iter, orient_eps=orient_eps,
                            orient_max_delta=orient_max_delta,
                            orient_limit_margin=orient_limit_margin)
        self.q = compute_ik(self.model, self.data, self.R_palm_id, r_target, self.q,
                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.R_joint_mask,
                            target_forearm_dir=r_dir_for_ik,
                            orient_weight=orient_weight, null_weight=null_weight,
                            orient_min_iter=orient_min_iter, orient_eps=orient_eps,
                            orient_max_delta=orient_max_delta,
                            orient_limit_margin=orient_limit_margin)

        # 2-1. Torso yaw 보상 (shoulder_roll이 한계에 막힌 경우)
        # compensated=True: 보상 중 / False: 복귀 중 — 어느 쪽이든 항상 publish
        if config.USE_TORSO:
            self.q, raw_torso, compensated, comp_arm = apply_torso_compensation(
                self.model, self.data, self.q,
                self.L_palm_id, self.R_palm_id,
                l_target, r_target,
                self.torso_idx, self.L_roll_idx, self.R_roll_idx,
            )
            self._torso_compensated_arm = comp_arm  # 'L', 'R', or None
            # EMA 필터를 먼저 적용한 torso_filt로 IK 재실행
            # raw_torso 기준 IK는 실제 로봇 torso(torso_filt)와 불일치 → 팔이 엉뚱한 곳으로 감
            torso_filt = float(self.torso_filter.filter(np.array([raw_torso]))[0])
            if compensated and self.torso_idx is not None:
                self.q[self.torso_idx] = torso_filt  # 실제 publish될 torso값으로 교체

                # 보상 팔만 IK 재실행 — 반대쪽 팔은 재실행하지 않음
                # 반대쪽 팔까지 재실행하면 허리 회전에 "반대로 싸우는" 불안정한 루프 발생
                # 반대쪽 팔은 첫 번째 IK 결과를 유지 → 허리 회전만큼 자연스럽게 따라감
                if self._torso_compensated_arm == 'R':
                    for _ in range(config.TORSO_IK_ITER):
                        self.q = compute_ik(self.model, self.data, self.R_palm_id, r_target, self.q,
                                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.R_joint_mask,
                                            target_forearm_dir=r_dir_for_ik,
                                            orient_weight=orient_weight, null_weight=null_weight,
                                            orient_min_iter=orient_min_iter, orient_eps=orient_eps,
                            orient_max_delta=orient_max_delta,
                            orient_limit_margin=orient_limit_margin)
                else:
                    for _ in range(config.TORSO_IK_ITER):
                        self.q = compute_ik(self.model, self.data, self.L_palm_id, l_target, self.q,
                                            q_ref=self.q_ref_current, q_init=self.q_init, joint_mask=self.L_joint_mask,
                                            target_forearm_dir=l_dir_for_ik,
                                            orient_weight=orient_weight, null_weight=null_weight,
                                            orient_min_iter=orient_min_iter, orient_eps=orient_eps,
                            orient_max_delta=orient_max_delta,
                            orient_limit_margin=orient_limit_margin)
            self._torso_compensated = compensated
        else:
            self._torso_compensated = False
            raw_torso  = float(self.q[self.torso_idx]) if self.torso_idx is not None else 0.0
            torso_filt = float(self.torso_filter.filter(np.array([raw_torso]))[0])



        # 3. FK (raw q) → IK 오차 계산 전용
        # self.q에는 unfiltered torso + wrist가 들어있어서 오차 계산에만 사용
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        l_actual  = self.data.oMf[self.L_palm_id].translation.copy()
        r_actual  = self.data.oMf[self.R_palm_id].translation.copy()
        self.l_err = float(np.linalg.norm(l_actual - l_target))
        self.r_err = float(np.linalg.norm(r_actual - r_target))

        # 4. 스무딩 + 구체 오버레이
        l_wrist_filt = self.wrist_filter_l.filter(np.array([float(self.l_wrist_yaw)]))[0]
        r_wrist_filt = self.wrist_filter_r.filter(np.array([float(self.r_wrist_yaw)]))[0]

        # 5. 관절각 명령 조합
        # arm_filter를 q_vis보다 먼저 적용해야 시각화와 실제 명령이 일치함
        self.q        = self.arm_filter.filter(self.q)
        self.cmd_data = [float(self.q[idx]) for idx in self.joint_ids]
        self.cmd_data[4] = l_wrist_filt
        self.cmd_data[9] = r_wrist_filt

        # 시각화: /joint_states 피드백으로 FK → 실제 로봇 손바닥 위치를 Quest에 표시
        # 제어(IK 명령)와 시각화(실제 상태)를 완전히 분리
        # torso 포함 모든 관절의 실제 값이 반영됨
        q_actual = self.ros.get_current_q()
        pin.forwardKinematics(self.model, self.data, q_actual)
        pin.updateFramePlacements(self.model, self.data)
        l_actual_vis = self.data.oMf[self.L_palm_id].translation.copy()
        r_actual_vis = self.data.oMf[self.R_palm_id].translation.copy()

        self.tv._teleop_active.value = 1 if config.USE_SPHERE else 0
        self.tv.l_palm_quest = robot_to_quest(l_actual_vis, self.robot_L_init, self.quest_L_init)
        self.tv.r_palm_quest = robot_to_quest(r_actual_vis, self.robot_R_init, self.quest_R_init)
        self.tv.l_palm_dir   = robot_dir_to_quest(self.data.oMf[self.L_palm_id].rotation, 'L')
        self.tv.r_palm_dir   = robot_dir_to_quest(self.data.oMf[self.R_palm_id].rotation, 'R')
        self._apply_neck(self.cmd_data)
        self._update_neck_warn(self.cmd_data)
        self.ros.publish_neck(self.cmd_data[10], self.cmd_data[11])  # 젯슨으로 목 각도 전송
        self.neck_cmd_last = [self.cmd_data[10], self.cmd_data[11]] 
        if config.USE_TORSO:
            self.ros.publish_torso(torso_filt)

        # 6. 안전 체크 (NaN/Inf → FREEZE)
        if not np.all(np.isfinite(self.cmd_data)):
            print("[WARN] IK diverged -> FREEZE")
            self.data_logger.log(self.teleop_state.name, 'IK_DIVERGED',
                                  l_raw=self.l_raw, r_raw=self.r_raw,
                                  sync_q=[v if np.isfinite(v) else 999.0 for v in self.cmd_data],
                                  note='cmd_data에 NaN/Inf 포함')
            self.q                 = self.ros.get_current_q()
            self.arm_filter.reset(self.q)
            self.sync_target_q     = None
            self.teleop_state      = TeleopState.FREEZE
            self.freeze_start_time = time.time()
            self.tracking_lost     = True
            self.blend_active      = False
            beep('warn')
            time.sleep(1.0 / config.CONTROL_HZ)
            return True

        # DEBUG: TELEOP 첫 IK 결과와 SYNC 종료 자세 차이
        if self.blend_active and self.blend_start_time and (time.time() - self.blend_start_time) < 0.1:
            diff = [abs(c - s) for c, s in zip(self.cmd_data, self.blend_start_q)]
            print(f"[DEBUG] TELEOP 첫 IK 결과와 SYNC 종료 자세 차이: max={max(diff):.4f} rad")

        # 6-1. 보조 싱킹: TELEOP 진입 직후 SYNC_BLEND_DURATION 동안
        # "진입 시점 실제 자세 -> 지금 IK가 원하는 자세"를 한 번 더 부드럽게 보간.
        # IK(self.cmd_data)는 매 프레임 정상적으로 새로 계산되므로 사용자가 움직이면 반영되지만,
        # publish되는 값만 시작 자세에서 점진적으로 풀려나가 SYNCING 잔여 오차가 1프레임에
        # 메워지는 걸 막는다.
        if self.blend_active:
            elapsed_blend = time.time() - self.blend_start_time
            blend_frac    = min(elapsed_blend / config.SYNC_BLEND_DURATION, 1.0)
            blend_smooth  = blend_frac**3 * (10-15*blend_frac + 6*blend_frac**2)
            # 목(10,11)은 블렌드에서 제외. blend_start_q[10],[11]은 self.q에서 오는데
            # 목은 /joint_states가 없어 self.q의 목 슬롯을 신뢰할 수 없다(항상 0으로 리셋됨).
            # 이 값으로 블렌드하면 SYNCING이 이미 맞춰놓은 목을 엉뚱한 값에서 다시
            # 보간하게 됨 → 목은 SYNCING의 보간만으로 충분하므로 _apply_neck()이
            # 이미 계산해둔 값을 그대로 통과시킨다.
            neck_now = (self.cmd_data[10], self.cmd_data[11])
            self.cmd_data = [s + (t - s) * blend_smooth
                             for s, t in zip(self.blend_start_q, self.cmd_data)]
            self.cmd_data[10], self.cmd_data[11] = neck_now
            if blend_frac >= 1.0:
                self.blend_active = False
                self._pub_episode_start.publish(Empty()) 

        # 7. Publish
        self.current_q_for_smooth = self.cmd_data.copy()
        self.ros.publish_arm(self.cmd_data)
        self.ros.publish_neck(self.cmd_data[10], self.cmd_data[11])
        self.neck_cmd_last = [self.cmd_data[10], self.cmd_data[11]]   # ← 추가
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

    def _handle_gripper(self):
        """TELEOP 상태에서 오른손 → 그리퍼(Dynamixel XM430) 리타게팅 + publish.

        기존 AmazingHand 왼손/오른손 파이프라인(_handle_fingers, 16채널)과는
        완전히 독립적으로 동작 — 그쪽 코드는 건드리지 않음. 오른손 물리 하드웨어를
        그리퍼로 바꾼 뒤에도 왼손 AmazingHand 경로는 그대로 살아있어도 무해함.

        config.USE_GRIPPER = False (또는 미정의) 면 아무것도 하지 않음.
        """
        if not getattr(config, 'USE_GRIPPER', False):
            return

        right_lm = self.tv.right_landmarks
        r_valid  = is_landmark_valid(right_lm)

        if config.GRIPPER_DEBUG and time.time() - self._last_status_time > 0.3:
            self._print_gripper_debug(right_lm, r_valid)

        if r_valid:
            grip_fingers = getattr(config, 'GRIP_FINGERS', (2, 3, 4))
            ratio = compute_grip_ratio(right_lm, fingers=grip_fingers)
        else:
            # 트래킹 소실 시 안전하게 '열림' 유지 (쥔 물체를 계속 붙잡고 있고 싶다면
            # ratio = self.grip_filter._prev[0] 로 바꿔서 '마지막 값 유지'로 변경 가능)
            ratio = 0.0

        ratio_filt = float(self.grip_filter.filter(np.array([ratio]))[0])
        if np.isfinite(ratio_filt):
            self.ros.publish_gripper(ratio_filt)

    def _print_gripper_debug(self, right_lm, r_valid):
        """GRIPPER_DEBUG=True일 때 검지/중지/약지 flex 각도(deg)를 콘솔에 출력.
        GRIPPER_ANGLE_OPEN/CLOSE(finger_mapping.py) 튜닝용 —
        손을 완전히 펼친 상태/완전히 쥔 상태에서 여기 찍힌 각도를 그대로
        GRIPPER_ANGLE_OPEN/GRIPPER_ANGLE_CLOSE에 반영하면 됨."""
        finger_names = {2: "검지", 3: "중지", 4: "약지"}
        if not r_valid:
            print("[GripperDebug] 오른손 트래킹 없음")
            return
        angles = grip_debug_angles(right_lm, fingers=(2, 3, 4))
        parts = [f"{finger_names[f]}:{angles[f]:.0f}°" for f in (2, 3, 4)]
        print(f"[GripperDebug] {'  '.join(parts)}")

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
        elapsed = time.time() - self.loop_start
        time.sleep(max(0, 1.0 / config.CONTROL_HZ - elapsed))
        loop_time = max(time.time() - self.loop_start, 1e-6)  # ← sleep 후로 이동

        # torso 상태 오버레이 갱신
        if config.USE_TORSO and self.torso_idx is not None:
            torso_val = float(self.q[self.torso_idx])
            if abs(torso_val) < 0.005:
                torso_state = ''
            elif getattr(self, '_torso_compensated', False):
                torso_state = 'COMP'
            else:
                torso_state = 'RETURN'
        else:
            torso_val   = 0.0
            torso_state = ''

        # 보조 싱킹 중에는 Quest 화면에 SYNCING과 동일한 "n.ns / 1s" 진행률 표시를 이어서 보여줌.
        # 내부 상태(self.teleop_state)는 이미 TELEOP이지만, overlay['state']만 잠깐 SYNCING으로 유지.
        if self.blend_active:
            elapsed_blend = time.time() - self.blend_start_time
            self.ros.overlay.update({
                'state':        'SYNCING',
                'sync_elapsed': elapsed_blend,
                'sync_timeout': config.SYNC_BLEND_DURATION,
            })
        else:
            self.ros.overlay.update({
                'state':       'TELEOP',
                'hz':          1.0 / loop_time,
                'l_err':       self.l_err,
                'r_err':       self.r_err,
                'l_joints':    self.cmd_data[:5],
                'r_joints':    self.cmd_data[5:10],
                'torso_yaw':   torso_val,
                'torso_state': torso_state,
            })

        if time.time() - self._last_status_time > self.STATUS_INTERVAL:
            arm_s = f"L={self.l_err*100:.1f}cm R={self.r_err*100:.1f}cm" if config.USE_ARM else "ARM=OFF"
            fe_s  = "FE=ON"  if config.USE_FINGER_FE  else "FE=OFF"
            aa_s  = "AA=ON"  if config.USE_FINGER_AA  else "AA=OFF"
            nk_s  = "NECK=ON" if config.USE_NECK else ("NECK=TRACK" if config.USE_NECK_TRACK else "NECK=OFF")
            print(f"[TELEOP] {1/loop_time:.1f}Hz | {arm_s} | {fe_s} {aa_s} | {nk_s}")
            if config.USE_ARM:
                # print(f"[WRIST]  L={np.degrees(self.l_wrist_yaw):+.1f}deg  R={np.degrees(self.r_wrist_yaw):+.1f}deg")
                if self.l_forearm_dir_target is not None:
                    # ① 피드백 기준 (/joint_states, q_actual) - 지금까지 보시던 값
                    # 이 시점의 self.data.oMf는 위쪽 시각화 섹션에서 이미
                    # q_actual로 FK된 상태 (self.q가 아님).
                    d_cur_l = self.data.oMf[self.L_palm_id].rotation[:, 2]
                    d_cur_r = self.data.oMf[self.R_palm_id].rotation[:, 2]
                    ang_l = np.degrees(np.arccos(np.clip(np.dot(d_cur_l, self.l_forearm_dir_target), -1, 1)))
                    ang_r = np.degrees(np.arccos(np.clip(np.dot(d_cur_r, self.r_forearm_dir_target), -1, 1)))
                    # print(f"[FOREARM] 정렬오차(피드백) L={ang_l:.1f}deg  R={ang_r:.1f}deg  (튜닝 중 orient_weight 확인용)")

                    # ② IK 명령 기준 (self.q) - IK가 실제로 뭘 계산해서 내보냈는지.
                    # ①과 크게 다르면 "IK는 잘 도는데 피드백/실행 경로에서 문제"라는 뜻.
                    pin.forwardKinematics(self.model, self.data, self.q)
                    pin.updateFramePlacements(self.model, self.data)
                    d_cmd_l = self.data.oMf[self.L_palm_id].rotation[:, 2]
                    d_cmd_r = self.data.oMf[self.R_palm_id].rotation[:, 2]
                    ang_cmd_l = np.degrees(np.arccos(np.clip(np.dot(d_cmd_l, self.l_forearm_dir_target), -1, 1)))
                    ang_cmd_r = np.degrees(np.arccos(np.clip(np.dot(d_cmd_r, self.r_forearm_dir_target), -1, 1)))
                    # print(f"[FOREARM] 정렬오차(IK명령)  L={ang_cmd_l:.1f}deg  R={ang_cmd_r:.1f}deg  (①과 크게 다르면 피드백 경로 문제)")
                    # 이후 다른 곳에서 q_actual 기준 data.oMf를 기대할 수 있으니 되돌려둠
                    # (q_actual은 이 메서드 스코프에 없으므로 여기서 새로 가져옴)
                    q_actual_for_restore = self.ros.get_current_q()
                    pin.forwardKinematics(self.model, self.data, q_actual_for_restore)
                    pin.updateFramePlacements(self.model, self.data)

                    yaw_l = np.degrees(self.q[self.joint_ids[2]])
                    yaw_r = np.degrees(self.q[self.joint_ids[7]])
                    # print(f"[SHOULDER_YAW] L={yaw_l:+.1f}deg  R={yaw_r:+.1f}deg  (실제 이 값이 움직여야 함)")
            self._last_status_time = time.time()

    # ──────────────────────────────────────────────────────────
    # 종료
    # ──────────────────────────────────────────────────────────

    def _shutdown(self):
        # 종료 시퀀스 도중 Ctrl+C가 또 들어와도 무시 → 끝까지 완료 보장
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self._pub_episode_end.publish(Empty())
            DURATION = 2.5

            # self.cmd_data는 TELEOP을 거치지 않으면 세션 시작 때 값 그대로임.
            # 종료 직전엔 항상 최신 /joint_states를 다시 읽어서 진짜 위치를 확인한다.
            q_actual = self.ros.get_current_q()
            current_vals_actual = [float(q_actual[idx]) for idx in self.joint_ids]

            if config.USE_TORSO:
                current_torso = float(q_actual[self.torso_idx]) if self.torso_idx is not None else 0.0
                torso_thread  = threading.Thread(
                    target=publish_torso_fin,
                    args=(self.ros.pub_torso, current_torso, DURATION),
                    daemon=True,
                )
                torso_thread.start()

            if config.USE_NECK or config.USE_NECK_TRACK:
                current_neck = self.neck_cmd_last   # get_current_q() 대신 자체 추적값
                target_neck  = [config.INIT_POS[10],     config.INIT_POS[11]]
                neck_thread  = threading.Thread(
                    target=publish_smooth_move,
                    args=(self.ros.pub_neck, target_neck, current_neck),
                    kwargs={'duration': DURATION, 'label': '목 종료'},
                    daemon=True,
                )
                neck_thread.start()

            publish_fin(self.ros.pub_arm, current_vals=current_vals_actual, duration=DURATION)

            if config.USE_TORSO:
                torso_thread.join()
            if config.USE_NECK or config.USE_NECK_TRACK:
                neck_thread.join()

            if self.finger_filter._prev is not None:
                current_fingers = self.finger_filter._prev.tolist()
            else:
                current_fingers = self.FINGER_NEUTRAL

            FINGER_DURATION = 1.0
            steps = int(FINGER_DURATION * config.CONTROL_HZ)
            for i in range(1, steps + 1):
                t = i / steps
                interp = [c + (g - c) * t for c, g in zip(current_fingers, self.FINGER_NEUTRAL)]
                self.ros.publish_hand(interp)
                time.sleep(1.0 / config.CONTROL_HZ)

            if hasattr(self, 'tv') and hasattr(self.tv, 'process'):
                self.tv.process.terminate()
                self.tv.process.join(timeout=2.0)

            if hasattr(self, 'shm'):
                self.shm.close()
                self.shm.unlink()
            print("[DONE] Shutdown complete")
        finally:
            signal.signal(signal.SIGINT, old_handler)  # 원래 핸들러로 복구

# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    TeleopController().run()