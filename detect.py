from ultralytics import YOLO
import cv2

def main():
    print("Loading model...")
    model = YOLO('yolov8n.pt')
    source_img = 'https://ultralytics.com/images/bus.jpg'
         print(f"Running detection on: {source_img}")
    results = model(source_img)
    result = results[0]
    annotated_frame = result.plot()
    output_filename = "results.jpg"
    cv2.imwrite(output_filename, annotated_frame)
    print(f"Detection complete. Image saved as '{output_filename}'")
if __name__ == "__main__":

    main()
