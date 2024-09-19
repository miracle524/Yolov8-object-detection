from ultralytics import YOLO
import cv2

class ObjectDetector:
    """
    Class to detect objects from a video

    :param video_path: path to the video file:
    :param model: YOLO object to detect the objects
    """
    def __init__(self, video_path: str, model: YOLO, display_size=(640, 480)):
        self.video_path = video_path
        self.model = model
        self.display_size = display_size

    def detect(self):
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return
        
        print(cv2.ocl.haveOpenCL())
        if cv2.ocl.haveOpenCL():  
            cv2.ocl.setUseOpenCL(True)  
            print("OpenCL is enabled in OpenCV.")  
        else:  
            print("OpenCL is not available on your system.")  

        fps = video.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second: {fps}")
        skip_frames = int(fps / 2)
        
        frame_count = 0
        flag = True

        while flag:
            flag, frame = video.read()
            if flag:
                frame_count += 1
                if frame_count % skip_frames == 0:
                    detection = self.model.track(frame, persist=True)
                    frame_ = detection[0].plot()
                    frame_resized = cv2.resize(frame_, self.display_size, interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(f"Video: {self.video_path}", frame_resized)
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break

        video.release()
        cv2.destroyAllWindows()