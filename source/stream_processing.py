"""
This script establishes the connection to the drone's RTMP stream, reads frames, and applies the semenatic segmentation
model to the frames. The segmented frames are then displayed in a named window until the 'q' key is pressed.
"""

# public libraries
import time
import ffmpeg
import numpy as np
import cv2
from PIL import Image

# imports from source
import model_inference
import settings


def livestream_executive(url):
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

    print(f'Initiated FFmpeg process at time {time.ctime()}')
    # # Set up FFmpeg process
    process = (
        ffmpeg
        .input(url, an=None)  # Audio Disabled in second parameter.
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.OUTPUT_FPS}')
        .global_args('-c:v', 'libx264')  # Use VideoToolbox for encoding
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
    )

    print(f'FFmpeg process connected at time {time.ctime()}')

    cv2.namedWindow('RTMP Stream', cv2.WINDOW_NORMAL)
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

        cv2.putText(output_frame, f'FPS: {settings.OUTPUT_FPS}',
                    settings.FPS_LOCATION,
                    settings.FONT,
                    settings.FONT_SCALE,
                    settings.FONT_COLOR,
                    settings.THICKNESS,
                    settings.LINE_TYPE)
        cv2.putText(output_frame, f'Shape: {settings.RESIZE_FRAME_WIDTH}x{settings.RESIZE_FRAME_HEIGHT}',
                    settings.SHAPE_LOCATION,
                    settings.FONT,
                    settings.FONT_SCALE,
                    settings.FONT_COLOR,
                    settings.THICKNESS,
                    settings.LINE_TYPE)
        cv2.putText(output_frame, f'Encoder/Decoder: {settings.MODEL_ENCODER_NAME}/{settings.MODEL_DECODER_NAME}',
                    settings.MODEL_DESCRIPTION_LOCATION,
                    settings.FONT,
                    settings.FONT_SCALE,
                    settings.FONT_COLOR,
                    settings.THICKNESS,
                    settings.LINE_TYPE)


        # Display the frame
        cv2.imshow('RTMP Stream', output_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    process.stdout.close()
    process.wait()
    cv2.destroyAllWindows()