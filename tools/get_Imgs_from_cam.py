from __future__ import annotations

import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import signal
from typing import cast
from time import perf_counter
import cv2
import numpy as np
import time
import ultralytics as ul
import os


def _update(
    args: tuple,
    props: dict[int, float],
    buffer: ctypes.Array[ctypes.c_uint8],
    ready: multiprocessing.synchronize.Event,
    cancel: multiprocessing.synchronize.Event,
):

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    _set_props(video_capture, props)

    try:
        while not cancel.is_set():
            ret, mat = cast("tuple[bool, cv2.Mat]", video_capture.read())
            if not ret:
                continue

            ready.clear()
            memoryview(buffer).cast("B")[:] = memoryview(mat).cast("B")[:]
            ready.set()

    finally:
        video_capture.release()


def _set_props(video_capture: cv2.VideoCapture, props: dict[int, float]):
    for key, value in props.items():
        try:
            video_capture.set(key, value)
        except:
            pass


def _get_props(video_capture: cv2.VideoCapture) -> dict[int, float]:
    ids = [cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH]
    return cast(
        "dict[int, float]", dict([[prop, video_capture.get(prop)] for prop in ids])
    )


def _get_information(
    args: tuple, in_props: dict[int, float]
) -> tuple[tuple[int, int, int], dict[int, float]]:

    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    _set_props(video_capture, in_props)
    out_props = _get_props(video_capture)

    try:
        ret, mat = cast("tuple[bool, cv2.Mat]", video_capture.read())
        if not ret:
            raise IOError()

        return mat.shape, out_props

    finally:
        video_capture.release()


class VideoCaptureWrapper:
    def __init__(self, *args) -> None:
        self.__args = ()
        self.__shape = cast(int, 0), cast(int, 0), cast(int, 0)
        self.__props: dict[int, float] = {}

        self.__buffer = multiprocessing.sharedctypes.RawArray(ctypes.c_uint8, 1)
        self.__ready = multiprocessing.Event()
        self.__cancel = multiprocessing.Event()
        self.__enqueue = multiprocessing.Process()

        self.__released = cast(bool, True)

        if len(args) == 0:
            return

        self.open(*args)

    def open(self, *args):
        if not self.__released:
            raise RuntimeError()

        self.__args = args
        self.__shape, self.__props = _get_information(self.__args, self.__props)

        height, width, channels = self.__shape
        self.__buffer = multiprocessing.sharedctypes.RawArray(
            ctypes.c_uint8, height * width * channels
        )

        self.__ready = multiprocessing.Event()
        self.__cancel = multiprocessing.Event()
        self.__enqueue = multiprocessing.Process(
            target=_update,
            args=(
                self.__args,
                self.__props,
                self.__buffer,
                self.__ready,
                self.__cancel,
            ),
            daemon=True,
        )
        self.__enqueue.start()

        self.__released = cast(bool, False)

    def get(self, propId: int):
        if self.__released:
            raise RuntimeError()

        return self.__props[propId]

    def set(self, propId: int, value: float):
        if self.__released:
            raise RuntimeError()

        self.__props[propId] = value
        self.release()
        self.open(*self.__args)

        return cast(bool, True)

    def read(self):
        if self.__released:
            raise RuntimeError()

        self.__ready.wait()
        return cast(bool, True), np.reshape(self.__buffer, self.__shape).copy()

    def isOpened(self):
        return not self.__released

    def release(self):
        if self.__released:
            return

        self.__cancel.set()
        self.__enqueue.join()
        self.__released = True

    def __del__(self):
        try:
            self.release()
        except:
            pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    dst_dir = "image\output_cam"
    video_path = (
        # "rtsp://admin:vuletech123@192.168.1.64/" #vule wifi
        "rtsp://admin:vuletech123@192.168.1.126/"  # isoft wifi
    )
    # video_path = "videos/cam1/new/cam 2/9.mp4"

    video_capture = VideoCaptureWrapper(video_path)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_name = "".join(str(time.time()).split(".")) + ".mp4"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_path = os.path.join(dst_dir, video_name)
    print(dst_path)
    W = int(1920)
    H = int(1080)
    writer = cv2.VideoWriter(
        dst_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=25, frameSize=(1280, 720)
    )

    # model = ul.YOLO("models/stait-seg.pt", task="segment")
    desired_width = 1920
    desired_height = 1080
    init_time = time.time()
    count = 0
    num = 0
    try:
        while True:
            start_time = perf_counter()
            ret, mat = video_capture.read()
            if not ret:
                break

            end_time = time.time()
            if end_time - init_time > 0.5:
                init_time = end_time
                count = 0

            # res = model.predict(mat, verbose=False)
            # mat = res[0].plot()
            count += 1
            mat = cv2.resize(mat, (1280, 720))

            writer.write(mat)
            cv2.imshow("Camera Feed", mat)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s"):  # wait for 's' key to save and exit
                cv2.imwrite("images/img" + str(num) + ".png", mat)
                print("image saved!")
                num += 1

    except KeyboardInterrupt:
        pass

    # Release and destroy all windows before termination
    video_capture.release()
    cv2.destroyAllWindows()
