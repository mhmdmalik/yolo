from ultralytics import YOLO
import cv2

def main():
    # 1. Load a pre-trained model (yolov8n.pt is the 'nano' version, fastest)
    # The model will automatically download if not present in the current folder.
    print("Loading model...")
    model = YOLO('yolov8n.pt')

    # --- OPTION A: Detect on a Static Image ---
    # Define the image source (URL or local path)
    source_img = 'https://ultralytics.com/images/bus.jpg'
    
    print(f"Running detection on: {source_img}")
    
    # Run inference
    # results is a list of Result objects (one per image)
    results = model(source_img)

    # Process the first result (since we only passed one image)
    result = results[0]

    # Generate an image with the bounding boxes drawn
    # labels=True draws the class names, boxes=True draws the rectangles
    annotated_frame = result.plot()

    # Save the resulting image to the current directory
    output_filename = "results.jpg"
    cv2.imwrite(output_filename, annotated_frame)
    print(f"Detection complete. Image saved as '{output_filename}'")


    # --- OPTION B: Live Webcam Detection (Uncomment to use) ---
    # To use this, comment out the "OPTION A" section above and uncomment below.
    
    # print("Starting Webcam...")
    # # source="0" specifies the default webcam
    # # show=True opens a window displaying the feed live
    # model.predict(source="0", show=True) 

if __name__ == "__main__":
    main()