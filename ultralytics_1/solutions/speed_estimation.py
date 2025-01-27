from time import time
import numpy as np
from ultralytics_1.solutions.solutions import BaseSolution
from ultralytics_1.utils.plotting import Annotator, colors
from collections import defaultdict
import math

class SpeedEstimator(BaseSolution):
    """
    A class to estimate the speed of objects in a real-time video stream based on their tracks.
    This class extends the BaseSolution class and provides functionality for estimating object speeds using
    tracking data in video streams.
    """

    def __init__(self, **kwargs):
        """Initializes the SpeedEstimator object with speed estimation parameters and data structures."""
        super().__init__(**kwargs)

        self.initialize_region()  # Initialize speed region
        self.spd = {}  # Dictionary for speed data
        self.trkd_ids = []  # List for already speed_estimated and tracked IDs
        self.trk_pt = {}  # Dictionary for tracks previous time
        self.trk_pp = {}  # Dictionary for tracks previous point
        self.crossed_ids = [] # Set to track objects that have crossed the line
        self.track_history = defaultdict(list)
        self.f_n = 0
        self.check = False

    def estimate_speed(self, im0, fram_num):
        """
        Estimates the speed of objects at every point in their trajectory.
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Draw the region of interest
        # self.annotator.draw_region(
        #     reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        # )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Update the track history with the new centroid position
            self.track_line = self.track_history[track_id]
            self.track_line.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            
            if len(self.track_line) > 30:
                self.track_line.pop(0)  # Limit the history length to avoid overfilling
            
            # Initialize tracking history if not present
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = [fram_num]  # Initialize with current frame number
            
            # Speed Calculation for every point
            if len(self.track_line) > 1:  # Ensure we have at least two points to calculate speed
                # Get the previous and current centroid (position)
                x1, y1 = self.track_line[-2]
                x2, y2 = self.track_line[-1]
                
                # Calculate the distance between the two points (Euclidean distance)
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Time difference (since we're tracking frames, it's just the difference in frame numbers)
                time_difference = fram_num - self.trk_pt[track_id][-1]  # Use the previous frame time
                
                if time_difference > 0:
                    # Calculate speed in pixels per frame (or units per frame)
                    speed = distance / time_difference
                    self.spd[track_id] = speed  # Store speed in dictionary
                    
                    # Update previous frame info
                    self.trk_pt[track_id].append(fram_num)
                    
                    # Annotate the image with speed at this point
                    speed_label = f"Speed: {speed:.2f} px/frame"  # You can format this however you like
                    self.annotator.box_label(box, label=speed_label, color=colors(track_id, True))
                
            # Draw object tracks
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

        # Display the output with annotations
        self.display_output(im0)

        return im0  # Return the processed image with speed annotations
