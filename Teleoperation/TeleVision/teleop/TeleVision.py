import time
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value
from vuer.schemas import Sphere
import numpy as np
import asyncio
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webrtc'))
    from zed_server import *
except ImportError as e:
    print(f"webrtc import 실패: {e}")
    pass

class OpenTeleVision:
    def __init__(self, img_shape, shm_name, queue, toggle_streaming, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False):
        self.img_shape = (img_shape[0], 2*img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]

        self.l_palm_shared    = Array('d', 3, lock=True)
        self.r_palm_shared    = Array('d', 3, lock=True)
        self.l_palm_dir_shared = Array('d', 3, lock=True)  # 손바닥 방향벡터
        self.r_palm_dir_shared = Array('d', 3, lock=True)
        self._teleop_active   = Value('i', 0)  # 0=비활성, 1=활성

        if ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            self.app.spawn(start=False)(self.main_image)
        elif stream_mode == "webrtc":
            self.app.spawn(start=False)(self.main_webrtc)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")

        self.left_hand_shared       = Array('d', 16, lock=True)
        self.right_hand_shared      = Array('d', 16, lock=True)
        self.left_landmarks_shared  = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        self.head_matrix_shared     = Array('d', 16, lock=True)
        self.aspect_shared          = Value('d', 1.0, lock=True)
        self._hand_last_time_shared = Value('d', 0.0, lock=True)
        self._hand_hz_shared        = Value('d', 0.0, lock=True)

        if stream_mode == "webrtc":
            if Args.verbose:
                logging.basicConfig(level=logging.DEBUG)
            else:
                logging.basicConfig(level=logging.INFO)
            Args.img_shape = img_shape
            Args.fps = 60
            ssl_context = ssl.SSLContext()
            ssl_context.load_cert_chain(cert_file, key_file)
            app = web.Application()
            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True, expose_headers="*",
                    allow_headers="*", allow_methods="*",
                )
            })
            rtc = RTC(img_shape, queue, toggle_streaming, 60)
            app.on_shutdown.append(on_shutdown)
            cors.add(app.router.add_get("/", index))
            cors.add(app.router.add_get("/client.js", javascript))
            cors.add(app.router.add_post("/offer", rtc.offer))
            self.webrtc_process = Process(target=web.run_app, args=(app,),
                                          kwargs={"host": "0.0.0.0", "port": 8081, "ssl_context": ssl_context})
            self.webrtc_process.daemon = True
            self.webrtc_process.start()

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    def run(self):
        self.app.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value   = event.value['camera']['aspect']
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        _now  = time.time()
        _last = self._hand_last_time_shared.value
        if _last > 0:
            self._hand_hz_shared.value = 1.0 / max(_now - _last, 1e-9)
        self._hand_last_time_shared.value = _now
        try:
            lh = event.value["leftHand"]
            rh = event.value["rightHand"]
            ll = np.array(event.value["leftLandmarks"]).flatten()
            rl = np.array(event.value["rightLandmarks"]).flatten()
            if lh and len(lh) == 16:  self.left_hand_shared[:]       = lh
            if rh and len(rh) == 16:  self.right_hand_shared[:]      = rh
            if len(ll) == 75:         self.left_landmarks_shared[:]  = ll
            if len(rl) == 75:         self.right_landmarks_shared[:] = rl
        except Exception:
            pass

    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always", grid=False)
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=True, showRight=True)
        session.upsert @ WebRTCStereoVideoPlane(
            src="https://192.168.8.102:8080/offer",
            key="zed", aspect=1.33334, height=8, position=[0, -2, -0.2],
        )
        while True:
            await asyncio.sleep(1)

    async def main_image(self, session, fps=60):
        session.set @ DefaultScene(grid=False)
        session.upsert @ Hands(fps=fps, stream=True, key="hands",
                               showLeft=True, showRight=True)

        while True:
            # ── 구체: 오른손 → 왼손 순서로, await로 분리해서 전송 ──
            if self._teleop_active.value:
                lp    = self.l_palm_quest.tolist()
                rp    = self.r_palm_quest.tolist()
                ld    = np.array(self.l_palm_dir_shared[:])
                rd    = np.array(self.r_palm_dir_shared[:])
                tip_l = (np.array(lp) + 0.10 * ld).tolist()
                tip_r = (np.array(rp) + 0.10 * rd).tolist()

                # 오른손 먼저
                try:
                    session.upsert @ Sphere(
                        key="robot_r_palm",
                        position=rp,
                        args=[0.015, 10, 10],
                        material={"color": "yellow"},
                    )
                except Exception as e:
                    print(f"[Sphere robot_r_palm ERR] {e}")

                await asyncio.sleep(0)  # 이벤트 루프에 양보

                try:
                    session.upsert @ Sphere(
                        key="robot_r_tip",
                        position=tip_r,
                        args=[0.008, 8, 8],
                        material={"color": "orange"},
                    )
                except Exception as e:
                    print(f"[Sphere robot_r_tip ERR] {e}")

                await asyncio.sleep(0)

                # 왼손 나중에
                try:
                    session.upsert @ Sphere(
                        key="robot_l_palm",
                        position=lp,
                        args=[0.015, 10, 10],
                        material={"color": "green"},
                    )
                except Exception as e:
                    print(f"[Sphere robot_l_palm ERR] lp={lp}  {e}")

                await asyncio.sleep(0)

                try:
                    session.upsert @ Sphere(
                        key="robot_l_tip",
                        position=tip_l,
                        args=[0.008, 8, 8],
                        material={"color": "lime"},
                    )
                except Exception as e:
                    print(f"[Sphere robot_l_tip ERR] tip={tip_l}  {e}")

            # ── 이미지: 실패해도 구체에 영향 없음 ─────────────────────
            try:
                image = self.img_array[:, :self.img_width, :]
                session.upsert @ ImageBackground(
                    image,
                    format="jpeg",
                    quality=95,
                    key="background",
                    interpolate=True,
                    distanceToCamera=8,
                    height=4.47,
                )
            except AssertionError:
                pass
            except Exception as e:
                print(f"[main_image] {e}")

            await asyncio.sleep(1.0 / fps)

    # ── Properties ────────────────────────────────────────
    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)

    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)

    @property
    def hand_hz(self):
        return float(self._hand_hz_shared.value)

    @property
    def l_palm_quest(self):
        return np.array(self.l_palm_shared[:])

    @l_palm_quest.setter
    def l_palm_quest(self, val):
        self.l_palm_shared[:] = np.asarray(val, dtype=float)

    @property
    def r_palm_quest(self):
        return np.array(self.r_palm_shared[:])

    @r_palm_quest.setter
    def r_palm_quest(self, val):
        self.r_palm_shared[:] = np.asarray(val, dtype=float)

    @property
    def l_palm_dir(self):
        return np.array(self.l_palm_dir_shared[:])

    @l_palm_dir.setter
    def l_palm_dir(self, val):
        self.l_palm_dir_shared[:] = np.asarray(val, dtype=float)

    @property
    def r_palm_dir(self):
        return np.array(self.r_palm_dir_shared[:])

    @r_palm_dir.setter
    def r_palm_dir(self, val):
        self.r_palm_dir_shared[:] = np.asarray(val, dtype=float)


if __name__ == "__main__":
    resolution = (720, 1280)
    crop_size_w = 340
    crop_size_h = 270
    resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)
    img_shape  = (2 * resolution_cropped[0], resolution_cropped[1], 3)
    shm        = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    img_array  = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
    tv = OpenTeleVision(resolution_cropped, shm.name, None, None,
                        cert_file="../cert.pem", key_file="../key.pem")
    while True:
        time.sleep(1)