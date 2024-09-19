from ultralytics import YOLO
from object_detector import ObjectDetector
import cv2

model_video = YOLO('yolov8x.pt')
model_image = YOLO('yolov8x.pt')

def detect_image(image_path):
    results = model_image(image_path, show=True)
    cv2.waitKey(0)

def detect_video(video_path):  
    detector = ObjectDetector(video_path, model_video)
    detector.detect()

def main():
    print("Select an option:")
    print("1: Detect objects in trafficcars.jpeg")
    print("2: Detect objects in monkey.jpg")
    print("3: Detect objects in demovideo1.avi using ObjectDetector class")

    choice = input("Enter the number of your choice: ")

    if choice == '1':
        detect_image('./assets/trafficcars.jpeg')
    elif choice == '2':
        detect_image('./assets/monkey.jpg')
    elif choice == '3':
        video_path = './assets/vehicle.mp4'
        detect_video(video_path)
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
