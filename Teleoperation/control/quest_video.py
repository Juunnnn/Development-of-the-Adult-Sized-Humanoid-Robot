"""
quest_video.py
──────────────
Quest VR 화면에 영상을 재생하는 모듈.

사용법:
    from quest_video import play_video_to_quest
    video_done = threading.Event()
    threading.Thread(
        target=play_video_to_quest,
        args=(image_array, "/home/teleopstation/Downloads/exia_startup.mp4"),
        kwargs={"ros": ros, "done_event": video_done},
        daemon=True
    ).start()
"""

import time
import cv2


def play_video_to_quest(image_array, video_path,
                        target_w=1280, target_h=720,
                        ros=None, done_event=None):
    """
    영상을 읽어 공유 메모리(image_array)에 프레임을 써넣어 Quest에 표시.

    Parameters
    ----------
    image_array : np.ndarray  shape=(720, 1280*2, 3)
        HR_teleop.py의 공유 메모리 배열
    video_path  : str
        재생할 영상 파일 경로
    target_w    : int  (기본 1280)
    target_h    : int  (기본 720)
    ros         : RosInterface | None
        전달하면 재생 중 _last_camera_time을 갱신해
        _overlay_loop이 영상을 덮어쓰지 못하도록 억제함
    done_event  : threading.Event | None
        재생 완료 시 set() 호출 → HR_teleop이 카운트다운 시작
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video] Cannot open: {video_path}")
        if done_event:
            done_event.set()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    spf = 1.0 / fps
    print(f"[video] Playing: {video_path}  ({fps:.1f} fps)")

    if ros is not None:
        ros.video_playing = True   # 카메라 콜백 + 오버레이 루프 억제 시작

    while cap.isOpened():
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_w, target_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 공유 메모리 왼쪽/오른쪽 모두 기록 (양눈 동일 영상)
        image_array[:target_h, :target_w,  :] = frame
        image_array[:target_h,  target_w:, :] = frame

        # overlay_loop 억제: 카메라가 방금 들어온 것처럼 시각 갱신
        if ros is not None:
            ros._last_camera_time = time.time()

        elapsed = time.time() - t0
        time.sleep(max(0.0, spf - elapsed))

    cap.release()
    if ros is not None:
        ros.video_playing = False   # 카메라 콜백 + 오버레이 루프 재개
    print("[video] Playback finished")

    # 영상 끝 → 카운트다운 시작 신호
    if done_event is not None:
        done_event.set()