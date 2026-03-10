import sys
sys.path.insert(0, '/home/teleopstation/TeleVision/teleop')
from multiprocessing import shared_memory, Queue, Event, Array, Value
import numpy as np
import time

resolution_cropped = (480, 640)
shm = shared_memory.SharedMemory(create=True, size=resolution_cropped[0] * resolution_cropped[1] * 3 * 2)
image_array = np.ndarray((resolution_cropped[0], resolution_cropped[1] * 2, 3), dtype=np.uint8, buffer=shm.buf)
image_array[:] = 128
image_queue = Queue()
toggle_streaming = Event()
from TeleVision import OpenTeleVision
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming, ngrok=False)
print("=== 서버 시작! ===")

try:
    while True:
        left = tv.left_hand
        right = tv.right_hand
        head = tv.head_matrix
        print(f"[LEFT  위치] {left[:3,3].round(3)}")
        print(f"[RIGHT 위치] {right[:3,3].round(3)}")
        print(f"[HEAD  위치] {head[:3,3].round(3)}")
        print(f"[HEAD  회전]\n{head[:3,:3].round(3)}")
        print("-"*40)
        time.sleep(0.5)
except KeyboardInterrupt:
    shm.unlink()
    print("종료")