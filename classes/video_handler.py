import logging
import sys
from typing import AsyncIterator

import cv2
import numpy
from collections.abc import Iterator


class VideoHandler:
    """A class that handles a video stream."""

    def __init__(self, vid_path: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._path: str = vid_path
        self._vid = cv2.VideoCapture(vid_path)
        if not self._vid.isOpened():
            self._logger.error("Error opening video stream or file")
            raise RuntimeError("Error opening video stream")

    def get_frames(self) -> Iterator[numpy.ndarray]:
        """A generator that generates a frame every time it is called.

        :return: ndarray of numpy.ndarray of frame.
        """
        while self._vid.isOpened():
            ret, frame = self._vid.read()
            if ret:
                yield frame
                continue
            break


def main():
    vh = VideoHandler('../data/test_vid.mp4')
    for frame in vh.get_frames():
        cv2.imshow('', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
