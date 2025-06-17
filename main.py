import argparse
import os
import sys
from ultralytics import YOLO
import cv2

def main():
    # Set up argparse to get the model path and output video path from the user
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('model_path', type=str, help='Path to the YOLO model file (should end with .pt)')
    parser.add_argument('--output', type=str, default='output.avi', help='Path to save the output video (should end with .avi)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--fps', type=float, default=20.0, help='Output video FPS (default: 20.0)')
    args = parser.parse_args()

    # Check if the model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        sys.exit(1)

    # Check if the model path ends with .pt
    if not args.model_path.endswith('.pt'):
        print("Error: Model path must end with .pt")
        sys.exit(1)

    # Check if the output path ends with .avi
    if not args.output.endswith('.avi'):
        print("Error: Output path must end with .avi")
        sys.exit(1)

    # Initialize variables
    cap = None
    out = None
    
    try:
        # Load the video capture
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            sys.exit(1)

        # Load the YOLO model
        print(f"Loading YOLO model from {args.model_path}...")
        model = YOLO(args.model_path)
        print("Model loaded successfully!")

        # Get video frame width, height and define the codec
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
        out = cv2.VideoWriter(args.output, fourcc, args.fps, (frame_width, frame_height))

        print(f"Recording to {args.output}")
        print("Press 'q' to quit or Ctrl+C to stop...")

        frame_count = 0
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Failed to capture image")
                break

            # Run YOLO inference
            results = model(img, verbose=False)  # Set verbose=False to reduce output

            # Draw bounding boxes and labels
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class and confidence
                        cls = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Only show detections with confidence > 0.5
                        if confidence > 0.5:
                            # Draw bounding box
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            
                            # Draw label with background
                            label = f"{r.names[cls]}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (255, 0, 255), -1)
                            cv2.putText(img, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write the frame to the video file
            out.write(img)
            frame_count += 1

            # Show the output frame
            cv2.imshow('YOLO Object Detection', img)
            
            # Check for 'q' key press or window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed, exiting...")
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up resources
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print(f"Recorded {frame_count} frames to {args.output}")

if __name__ == "__main__":
    main()