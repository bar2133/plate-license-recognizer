import logging
import os.path
from difflib import SequenceMatcher
import cv2
import easyocr
import numpy as np
from numpy import ndarray
from ultralytics.nn.autoshape import Detections
from classes.picture_handler import PictureHandler
from classes.video_handler import VideoHandler
from enum import Enum

import torch

from defenetions import MAIN_PATH

MODEL_PATH = os.path.join(MAIN_PATH, 'models', 'best.pt')
ALLOW_LIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


class StreamType(str, Enum):
    VIDEO = 'video'
    PICTURE = 'picture'


class PlateLicenseRecognitionSystem(object):
    def __init__(self, data_path: str, data_type: StreamType = StreamType.VIDEO,
                 model_path: str = MODEL_PATH, gpu: bool = False, match_ratio: float = 0.8,
                 *args, **kwargs):
        # initialize attributes
        self.__match_ratio = match_ratio
        self.__data_type = data_type
        self.__logger = logging.getLogger(self.__class__.__name__)
        # initialize the video handler
        self.__data_path = data_path
        self._data_handler: VideoHandler | PictureHandler = VideoHandler(
            self.__data_path) if data_type == StreamType.VIDEO else PictureHandler(self.__data_path)
        # init the model object and the GPU.
        self.__processing_device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        self.__easyocr = easyocr.Reader(['en'])
        self.__model_path = model_path
        self._model = self.__init_model()

    def __init_model(self):
        return torch.hub.load(
            model='custom',
            source="local",
            repo_or_dir=os.path.join(MAIN_PATH, 'yolov5-master'),
            path=self.__model_path,
            force_reload=True,
            device=self.__processing_device)

    def find_plates(self, plates_license=None):
        plates_to_find = plates_license if plates_license is not None else []
        match self.__data_type:
            case StreamType.VIDEO:
                for frame in self._data_handler.get_frames():
                    image = self.__analayze_frame(frame, plates_to_find)
                    cv2.imshow('', image)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            case StreamType.PICTURE:
                frame = self._data_handler.get_picture()
                image = self.__analayze_frame(frame, plates_to_find)
                cv2.imshow('', image)
                cv2.waitKey(0)
            case _:
                self.__logger.error('Stream type not supported')
        cv2.destroyAllWindows()

    def __analayze_frame(self, frame, plates_to_find) -> ndarray:
        model_detection: Detections = self._model(frame)
        image: ndarray = np.squeeze(model_detection.ims)
        try:
            self.__prepare_image(model_detection, image, plates_to_find)
        except Exception:
            ...
        return image

    def __prepare_image(self, res: Detections, image: ndarray, plates_to_find: list[str]) -> None:
        green = (0, 255, 0)
        red = (0, 0, 255)
        for pred in res.pred:
            for *box, conf, cls in reversed(pred):
                x1, y1, x2, y2 = [int(x) for x in box]
                plate_number_img = image[y1:y2, x1:x2]  # crop the plate number from the image
                license_plate_gray = cv2.cvtColor(plate_number_img, cv2.COLOR_BGR2GRAY)  # convert to gray
                text_results = self.__easyocr.readtext(license_plate_gray,
                                                       allowlist=ALLOW_LIST)  # parse plate number to string
                extracted_plate_number_txt = self.__extract_text(text_results)  # extract the plate number
                if len(extracted_plate_number_txt) < 4:
                    continue
                required_plate = self.__check_if_required_plate_number(extracted_plate_number_txt.lower(),
                                                                       plates_to_find)
                image = cv2.putText(image, extracted_plate_number_txt, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, red if not required_plate else green, 2, cv2.LINE_AA)
                cv2.rectangle(image, (x1, y1), (x2, y2), red if not required_plate else green, 2)

    def __extract_text(self, text_results: list) -> str:
        for out in text_results:
            text_bbox, text, text_score = out
            text = text.replace(' ', '')
            text = text.replace('.', '')
            return text

    def __check_if_required_plate_number(self, text_result: str, plates_to_find: list[str]) -> bool:
        return self.__check_match_ratio(output_txt=text_result, required_plates_numbers=plates_to_find)

    def __check_match_ratio(self, output_txt: str, required_plates_numbers: list[str]) -> bool:
        for plate in required_plates_numbers:
            ratio = SequenceMatcher(a=output_txt, b=plate).ratio()
            self.__logger.debug(f"__check_match_ratio: {output_txt=} {plate=} {ratio=}")
            if ratio >= self.__match_ratio:
                return True
        return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    plrs = PlateLicenseRecognitionSystem(data_path="../data/il_plate.webp", data_type=StreamType.PICTURE)
    plrs.find_plates(plates_license=["7brc266"])

    # plrs = PlateLicenseRecognitionSystem(data_path="../data/test_vid2.mov")
    # plrs.find_plates(plates_license=["nj295s"])

    # plrs = PlateLicenseRecognitionSystem(data_path="../data/il_plate.webp", data_type=StreamType.PICTURE)
    # plrs.find_plates(plates_license=["56021802"])
