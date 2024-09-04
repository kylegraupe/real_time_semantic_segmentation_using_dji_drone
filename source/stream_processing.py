"""
This script establishes the connection to the drone's RTMP stream, reads frames, and applies the semenatic segmentation
model to the frames. The segmented frames are then displayed in a named window until the 'q' key is pressed.
"""
import cv2
import numpy as np
import time
from PIL import Image
import ffmpeg
import settings
import model_inference

def livestream_executive(url, app):
    """
    Establishes a livestream connection to the provided URL, reads video frames,
    applies a segmentation model to the frames, and displays both the original
    and segmented frames side-by-side in the UI window.

    Args:
        url (str): The URL of the livestream to connect to.
        app: The StreamApp instance for updating the UI with video frames.

    Returns:
        None
    """

    print(f'Initiated FFmpeg process at time {time.ctime()}')
    # Set up FFmpeg process
    process = (
        ffmpeg
        .input(url, an=None)  # Audio Disabled in second parameter.
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.OUTPUT_FPS}')
        .global_args('-c:v', 'libx264')  # Use VideoToolbox for encoding
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
    )
    print(f'FFmpeg process connected at time {time.ctime()}')

    frame_size = settings.FRAME_WIDTH * settings.FRAME_HEIGHT * settings.NUM_CHANNELS

    while True:
        in_bytes = process.stdout.read(frame_size) # Read frame in byte format

        # If no data is read, continue to check for errors
        if len(in_bytes) != frame_size:
            if not in_bytes:
                print("End of stream or error reading frame")
            else:
                print("Error: Read incomplete frame")
            break

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy() # Convert to numpy array

        if settings.MODEL_ON:
            # Apply segmentation model to the frame
            segmented_frame_np_gray = model_inference.image_to_tensor(Image.fromarray(in_frame), settings.MODEL, settings.DEVICE).astype(np.uint8)
            segmented_frame_img_rgb = settings.COLOR_MAP[segmented_frame_np_gray]
            segmented_frame_np_rgb = np.array(segmented_frame_img_rgb)

            in_frame = cv2.resize(in_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)

            if settings.SIDE_BY_SIDE:
                # Stack the original and segmented frames horizontally
                output_frame = np.hstack((in_frame, segmented_frame_np_rgb))
            else:
                output_frame = segmented_frame_np_rgb

        else:
            output_frame = in_frame

        # Resize the output frame to fit the display area
        display_width = settings.VIDEO_DISPLAY_WIDTH
        display_height = settings.VIDEO_DISPLAY_HEIGHT

        if output_frame.shape[1] > display_width or output_frame.shape[0] > display_height:
            # Calculate the aspect ratio
            aspect_ratio = output_frame.shape[1] / output_frame.shape[0]
            if output_frame.shape[1] > display_width:
                new_width = display_width
                new_height = int(display_width / aspect_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * aspect_ratio)

            # Resize the frame
            output_frame = cv2.resize(output_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Update the UI with the processed frame
        app.update_video_display(output_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    process.stdout.close()
    process.wait()
