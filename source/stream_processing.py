import time
import threading
from queue import Queue
import ffmpeg
import numpy as np
import cv2
import model_inference
import stream_diagnostics
from PIL import Image

MODEL = model_inference.model_unet
MODEL_ON = True
FPS = 1

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RESIZE_WIDTH = 704
FRAME_RESIZE_HEIGHT = 720

# Set font properties
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_LOCATION = (10, 50)
SHAPE_LOCATION = (10, 75)
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
THICKNESS = 1
LINE_TYPE = 2

COLOR_MAP = np.array([
    [0, 0, 0],        # Class 0: black
    [128, 0, 0],      # Class 1: dark red
    [0, 128, 0],      # Class 2: dark green
    [128, 128, 0],    # Class 3: dark yellow
    [0, 0, 128],      # Class 4: dark blue
    [128, 0, 128],    # Class 5: dark purple
    [0, 128, 128],    # Class 6: dark cyan
    [128, 128, 128],  # Class 7: gray
    [64, 0, 0],       # Class 8: maroon
    [192, 0, 0],      # Class 9: red
    [64, 128, 0],     # Class 10: olive
    [192, 128, 0],    # Class 11: orange
    [64, 0, 128],     # Class 12: purple
    [192, 0, 128],    # Class 13: magenta
    [64, 128, 128],   # Class 14: teal
    [192, 128, 128],  # Class 15: light gray
    [0, 64, 0],       # Class 16: dark green
    [128, 64, 0],     # Class 17: brown
    [0, 192, 0],      # Class 18: lime
    [128, 192, 0],    # Class 19: chartreuse
    [0, 64, 128],     # Class 20: navy
    [128, 64, 128],   # Class 21: medium purple
    [0, 192, 128],    # Class 22: aquamarine
], dtype=np.uint8)


def get_frame_size(url):
    """
    Retrieves the width and height of a video frame from a given URL.

    Args:
        url (str): The URL of the video to probe.

    Returns:
        tuple: A tuple containing the width and height of the video frame.
    """
    probe = ffmpeg.probe(url, v='error', select_streams='v:0', show_entries='stream=width,height')
    video_stream = probe['streams'][0]
    width = video_stream['width']
    height = video_stream['height']
    return width, height


def livestream_2(url):
    """
    Establishes a livestream connection to the provided URL, reads video frames,
    applies a segmentation model to the frames, and displays both the original
    and segmented frames side-by-side in a named window until the 'q' key is pressed.

    Args:
        url (str): The URL of the livestream to connect to.
        model: The segmentation model to apply to the frames.

    Returns:
        None
    """

    # Get frame size
    # width, height = get_frame_size(url)
    frame_size = 1280 * 720 * 3

    print(f'Initiated FFmpeg process at time {time.ctime()}')
    # Set up FFmpeg process
    process = (
        ffmpeg
        .input(url, **{'an': None, 'r': f'{FPS}'})  # Audio Disabled in second parameter.
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .global_args('-threads', '4')
        .global_args('-c:v', 'h264_videotoolbox')  # Use VideoToolbox for encoding
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    print(f'FFmpeg process connected at time {time.ctime()}')


    # Create a named window
    cv2.namedWindow('RTMP Stream', cv2.WINDOW_NORMAL)

    while True:
        # print(f'{time.time()}: Processing frame...')
        # Calculate frame size based on width and height
        in_bytes = process.stdout.read(frame_size)

        # If no data is read, continue to check for errors
        if len(in_bytes) != frame_size:
            if not in_bytes:
                print("End of stream or error reading frame")
            else:
                print("Error: Read incomplete frame")
            break

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy()

        if MODEL_ON:

            # Apply segmentation model to the frame
            segmented_frame_np_gray = model_inference.image_to_tensor(Image.fromarray(in_frame), MODEL).astype(np.uint8)
            segmented_frame_img_rgb = COLOR_MAP[segmented_frame_np_gray]
            # segmented_frame_img_rgb = cv2.cvtColor(segmented_frame_np_gray, cv2.COLOR_GRAY2RGB)
            segmented_frame_np_rgb = np.array(segmented_frame_img_rgb)

            in_frame = cv2.resize(in_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)

            # Stack the original and segmented frames horizontally
            output_frame = np.hstack((in_frame, segmented_frame_np_rgb))

        else:
            output_frame = in_frame

        cv2.putText(output_frame, f'FPS: {0}',
                    FPS_LOCATION,
                    FONT,
                    FONT_SCALE,
                    FONT_COLOR,
                    THICKNESS,
                    LINE_TYPE)
        cv2.putText(output_frame, f'Shape: {1280}x{720}',
                    SHAPE_LOCATION,
                    FONT,
                    FONT_SCALE,
                    FONT_COLOR,
                    THICKNESS,
                    LINE_TYPE)

        # Display the frame
        cv2.imshow('RTMP Stream', output_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    process.stdout.close()
    process.wait()
    cv2.destroyAllWindows()

