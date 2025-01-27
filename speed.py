import cv2
import os
from ultralytics_1 import solutions

# Directory containing videos
video_folder = "videos/"  # Replace with the correct path to your folder

# Initialize SpeedEstimator
speed_region = [(1920, 360), (0, 360)]
speed = solutions.SpeedEstimator(
    show=True,  # Display the output
    model="yolo11n-pose.pt",  # Path to the YOLO11 model file
    region=speed_region,  # Pass region points
    classes=[0],  # If you want to estimate speed of specific classes
)

# Loop over all files in the video folder
for filename in os.listdir(video_folder):
    video_path = os.path.join(video_folder, filename)

    # Check if the file is a valid video (e.g., mp4 or avi)
    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more extensions if needed
        print(f"Processing video: {filename}")

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Error opening video file {filename}"

        # Get video properties
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter(
            os.path.join(video_folder, f"processed_{filename}"), 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            fps, 
            (w, h)
        )

        fram_num = -1
        # Process video
        while cap.isOpened():
            fram_num += 1
            success, im0 = cap.read()

            if success:
                out = speed.estimate_speed(im0, fram_num)
                video_writer.write(out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(f"End of video {filename}.")
                break

        cap.release()
        video_writer.release()

        print(f"Finished processing {filename}. Output saved.")
    else:
        print(f"Skipping non-video file: {filename}")

cv2.destroyAllWindows()
