import sys
sys.path.insert(0, '/home/teleopstation/TeleVision/teleop')
from multiprocessing import shared_memory, Queue, Event
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
# 이걸로 교체
from sensor_msgs.msg import Image
import cv2
import time
from TeleVision import OpenTeleVision

sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.insert(0, '/home/teleopstation/TeleVision/teleop')

# opentelevision 서버 시작
resolution_cropped = (480, 640)
shm = shared_memory.SharedMemory(create=True, size=resolution_cropped[0] * resolution_cropped[1] * 3 * 2)
image_array = np.ndarray((resolution_cropped[0], resolution_cropped[1] * 2, 3), dtype=np.uint8, buffer=shm.buf)
image_array[:] = 128

image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming, ngrok=False)

# ROS 초기화
rospy.init_node('teleop_gazebo')
pub = rospy.Publisher('/arm_controller/command', Float64MultiArray, queue_size=10)

# 카메라 콜백: 이미지를 shared memory에 업데이트
def camera_callback(msg):
    try:
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        frame = cv2.resize(frame, (640, 480))
        image_array[:, :640, :] = frame
        image_array[:, 640:, :] = frame
    except Exception as e:
        print(f"카메라 오류: {e}")

rospy.Subscriber('/camera/color/image_raw', Image, camera_callback)

rate = rospy.Rate(50)
print("=== Teleop Gazebo 시작! ===")
print("Quest 접속: https://localhost:8012?ws=wss://localhost:8012")

try:
    while not rospy.is_shutdown():
        left = tv.left_hand
        right = tv.right_hand
        r_x = float(right[0, 3])
        r_shoulder_pitch = np.clip(r_x * 2.0, -1.57, 1.57)
        cmd = Float64MultiArray()
        cmd.data = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            r_shoulder_pitch, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ]
        pub.publish(cmd)
        print(f"[RIGHT x] {r_x:.3f} → [R_shoulder_pitch] {r_shoulder_pitch:.3f}")
        rate.sleep()
except KeyboardInterrupt:
    shm.unlink()
    print("종료")