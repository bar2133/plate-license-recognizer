from classes.video_handler import VideoHandler
import cv2


def main():
    sh = VideoHandler('vid.mp4')
    for frame in sh.get_frames():
        cv2.imshow('', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
