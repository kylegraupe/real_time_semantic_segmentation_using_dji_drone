"""
This file contains the functions to diagnose the characteristics of the stream.
"""

import cv2
import time

def calculate_fps(process, window_name, in_bytes, in_frame):
    """
    Calculates the frames per second (FPS) of the video and displays it on the named window.

    Args:
        process (ffmpeg process): The FFmpeg process object.
        window_name (str): The name of the window to display the FPS on.
        in_bytes (bytes): The bytes of the frame.
        in_frame (numpy array): The frame.

    Returns:
        None
    """
    fps_values = []
    start_time = time.time()
    fps = 1 / (time.time() - start_time)
    fps_values.append(fps)

    # Display the FPS on the window
    fps_text = f"FPS: {int(sum(fps_values) / len(fps_values))}"
    cv2.putText(in_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def get_latency():
    return None

