import cv2
from ultralytics import YOLO

def main():
    # Load your trained model
    model = YOLO(r"runs\detect\hand-throw2\HandV2\weights\best.pt")

    # Create a resizable window
    window_name = "Snowball Trajectory - Hand Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Force the window to a specific size on one monitor
    cv2.resizeWindow(window_name, 800, 600)

    # Run prediction (0 for webcam, or path to your video file)
    # We use stream=True for better performance
    source = "snowball-3second-360.mp4" 
    
    results = model.predict(source=source, stream=True, device=0, conf=0.4)

    for r in results:
        # Plot the detections (boxes/labels) onto the frame
        annotated_frame = r.plot()

        # Display the frame in our controlled window
        cv2.imshow(window_name, annotated_frame)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()