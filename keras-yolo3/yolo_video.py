
from yolo import YOLO
from yolo import detect_video



if __name__ == '__main__':
    video_path='1.avi'
    detect_video(YOLO(), video_path)
