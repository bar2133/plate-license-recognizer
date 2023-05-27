import cv2


class PictureHandler:
    def __init__(self, path: str):
        self.path = path
        try:
            self.picture = cv2.imread(self.path)
        except Exception:
            raise RuntimeError(f"Could not open picture file for {self.path}")

    def get_picture(self):
        return self.picture
