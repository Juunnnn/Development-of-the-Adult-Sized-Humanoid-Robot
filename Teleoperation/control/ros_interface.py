"""
ros_interface.py
─────────────────────────────────────────────────────────────
ROS 연결 전담 모듈입니다.

- RosInterface 클래스로 Publisher/Subscriber를 캡슐화
- 카메라 영상 수신 → Quest 스트리밍용 shared memory에 복사
- /joint_states 수신 → 현재 로봇 관절각 읽기
- pub_arm / pub_hand publish 헬퍼
"""

import time
import numpy as np
import cv2
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, JointState
import pinocchio as pin

import config


class RosInterface:
    """
    텔레옵에 필요한 모든 ROS 연결을 한 클래스에서 관리합니다.

    원본 teleop_ik.py에서는 rospy.init_node, Publisher, Subscriber,
    콜백 함수들이 전역에 흩어져 있었는데, 이를 하나로 묶었습니다.

    Attributes (자주 직접 접근하는 것)
    -----------
    pub_arm  : Publisher → /arm_controller/command  (팔+목 관절 12개)
    pub_hand : Publisher → /finger_controller/command (손가락 16개)
    """

    def __init__(self, model, q_init, image_array: np.ndarray):
        """
        Parameters
        ----------
        model       : pin.Model    관절명 → pinocchio q 인덱스 변환에 사용
        q_init      : np.ndarray   /joint_states 미수신 시 fallback 자세
        image_array : np.ndarray   카메라 영상을 넣는 shared memory 배열
                                   (Quest 스트리밍 서버가 이 배열을 읽어서 Quest로 전송)
        """
        self.model       = model
        self.q_init      = q_init
        self.image_array = image_array

        # 현재 관절 상태. /joint_states 수신 전에는 None.
        # get_current_q()에서 None이면 q_init을 fallback으로 사용함.
        self._current_joint_state = None

        # ROS 노드 초기화.
        # disable_signals=True: ROS가 SIGINT를 가로채지 않도록 함.
        # 메인 루프의 KeyboardInterrupt가 정상 동작하려면 이게 필요함.
        rospy.init_node('teleop_ik', disable_signals=True)

        # 팔+목 관절 명령 publisher. Float64MultiArray 12개 값.
        self.pub_arm  = rospy.Publisher(config.TOPIC_ARM,  Float64MultiArray, queue_size=10)

        # 손가락 관절 명령 publisher. Float64MultiArray 16개 값.
        # [L_AA_1, L_FE_1, ..., L_AA_4, L_FE_4, R_AA_1, ..., R_AA_4, R_FE_4]
        self.pub_hand = rospy.Publisher(config.TOPIC_HAND, Float64MultiArray, queue_size=10)

        # D435i 카메라 영상 subscriber
        rospy.Subscriber(config.TOPIC_CAMERA,       Image,      self._camera_callback)

        # 로봇 실제 관절각 subscriber
        rospy.Subscriber(config.TOPIC_JOINT_STATES, JointState, self._joint_state_callback)


    # ── 카메라 콜백 ────────────────────────────────────────────
    def _camera_callback(self, msg: Image):
        """
        D435i 카메라에서 받은 영상을 Quest 스트리밍용 shared memory에 복사합니다.

        Quest에는 좌우 스테레오 형식으로 보내야 하므로
        같은 프레임을 image_array 왼쪽 절반(:1280)과 오른쪽 절반(1280:)에 각각 복사함.
        실제로는 단안 카메라 영상을 좌우 동일하게 복사해서 퀘스트가 3D처럼 보이게 함.
        """
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            frame = cv2.resize(frame, (1280, 720))
            self.image_array[:, :1280, :]  = frame   # 왼쪽 눈용
            self.image_array[:, 1280:, :]  = frame   # 오른쪽 눈용
        except Exception as e:
            print(f"카메라 오류: {e}")


    # ── 관절 상태 콜백 ─────────────────────────────────────────
    def _joint_state_callback(self, msg: JointState):
        """
        로봇에서 발행하는 실제 관절각(/joint_states)을 받아 저장합니다.

        이 값이 필요한 이유:
          1. 텔레옵 시작 시 q_init이 아닌 현재 실제 자세에서 시작해야 안전함
             (q_init으로 시작하면 로봇이 앞으로나란히 자세로 갑자기 튀어나감)
          2. SYNCING에서 로봇이 실제로 목표에 도달했는지 확인하는 데 사용
        """
        self._current_joint_state = msg


    # ── 현재 q 읽기 ────────────────────────────────────────────
    def get_current_q(self) -> np.ndarray:
        """
        /joint_states에서 받은 실제 관절각으로 pinocchio q 벡터를 구성합니다.

        pinocchio의 q 벡터는 모든 관절(고정 관절 포함)을 다 포함하는 긴 벡터임.
        /joint_states는 제어 관절 이름과 값 쌍으로 오므로,
        이름으로 pinocchio 인덱스를 찾아서 해당 위치에 값을 채워야 함.

        /joint_states 미수신 시: q_init으로 대체 + 경고 출력.
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
        프로그램 시작 시 /joint_states가 올 때까지 대기합니다.

        왜 대기하나?
        ────────────
        프로그램이 막 시작되면 로봇 컨트롤러가 아직 publish를 안 했을 수 있음.
        기다리지 않고 바로 get_current_q()를 쓰면 q_init(앞으로나란히)이 반환되어
        텔레옵 시작 시 로봇이 앞으로나란히 자세로 갑자기 이동할 수 있음.

        timeout(s) 초과 시 q_init 반환 (경고 출력).
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


    # ── publish 헬퍼 ───────────────────────────────────────────
    def publish_arm(self, data: list):
        """
        팔+목 관절 명령을 publish합니다.
        Float64MultiArray 12개: [L어깨3 + L팔꿈치 + L손목, R어깨3 + R팔꿈치 + R손목, 목yaw, 목pitch]
        원본의 반복 패턴(msg 생성 → data 대입 → publish)을 한 줄로 줄임.
        """
        msg      = Float64MultiArray()
        msg.data = data
        self.pub_arm.publish(msg)

    def publish_hand(self, data: list):
        """
        손가락 관절 명령을 publish합니다.
        Float64MultiArray 16개: [L_AA_1, L_FE_1, ..., L_AA_4, L_FE_4,
                                  R_AA_1, R_FE_1, ..., R_AA_4, R_FE_4]
        landmark가 없거나 캘리브 전이면 FINGER_NEUTRAL([0.0]*16)을 넘겨 손가락을 펼침.
        """
        msg      = Float64MultiArray()
        msg.data = data
        self.pub_hand.publish(msg)
