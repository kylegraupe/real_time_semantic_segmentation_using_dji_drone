"""
This script establishes the connection to the drone's RTMP stream, reads frames, and applies the semantic segmentation
model to the frames. The segmented frames are then displayed in a named window until the 'q' key is pressed.
"""
import time
import ffmpeg
import numpy as np
import cv2
from PIL import Image

import settings
import model_inference
import mask_postprocessing
import ui_input_variables


# def get_rtmp_frames(url):
#     """
#     Establishes a connection to the drone's RTMP stream using FFmpeg.
#
#     Args:
#         url (str): The URL of the RTMP stream to connect to.
#
#     Returns:
#         process (subprocess.Popen): The FFmpeg subprocess object.
#
#     """
#     print(f'Initiated FFmpeg process at time {time.ctime()}')
#     # Set up FFmpeg process
#     process = (
#         ffmpeg
#         .input(url, an=None)  # Disable audio
#         .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.OUTPUT_FPS}')
#         .global_args('-c:v', 'libx264')
#         .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
#     )
#     print(f'FFmpeg process connected at time {time.ctime()}')
#     return process


def livestream_executive(url):
    """
    Establishes a livestream connection to the provided URL, reads video frames,
    applies a segmentation model to the frames, and displays both the original
    and segmented frames side-by-side in a window.

    Args:
        url (str): The URL of the livestream to connect to.

    Returns:
        None
    """

    print(f'Initiated FFmpeg process at {time.ctime()}')
    # Set up FFmpeg process
    process = (
        ffmpeg
        .input(url, an=None)  # Disable audio
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.OUTPUT_FPS}')
        .global_args('-c:v', 'libx264')
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
    )
    print(f'FFmpeg process connected at {time.ctime()}')

    frame_size = settings.FRAME_WIDTH * settings.FRAME_HEIGHT * settings.NUM_CHANNELS

    while True:
        # Read frame in byte format
        in_bytes = process.stdout.read(frame_size)

        # Check for end of stream or error
        if len(in_bytes) != frame_size:
            if not in_bytes:
                print("End of stream or error reading frame")
            else:
                print("Error: Read incomplete frame")
            break

        # Convert the byte array to a numpy array representing the frame
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy()
        segmented_frame_np_gray = model_inference.image_to_tensor(Image.fromarray(in_frame), settings.MODEL,
                                                                  settings.DEVICE).astype(np.uint8)

        if settings.MODEL_ON:

            # Define the CRF layer with the number of classes
            if settings.CRF_ON:
                segmentation_results = mask_postprocessing.apply_crf(in_frame, segmented_frame_np_gray)

            else:
                # Apply segmentation model to the frame
                segmented_frame_np_gray = model_inference.image_to_tensor(Image.fromarray(in_frame), settings.MODEL, settings.DEVICE).astype(np.uint8)
                segmented_frame_img_rgb = settings.COLOR_MAP[segmented_frame_np_gray]
                segmented_frame_np_rgb = np.array(segmented_frame_img_rgb)

                # Resize original frame to match segmented frame
                in_frame = cv2.resize(in_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)
                segmentation_results = segmented_frame_np_rgb

            if settings.SIDE_BY_SIDE:
                # Stack the original and segmented frames horizontally
                output_frame = np.hstack((in_frame, segmentation_results))
            else:
                output_frame = segmentation_results
        else:
            output_frame = in_frame

        # Resize the output frame to fit the display window if necessary
        display_width = settings.VIDEO_DISPLAY_WIDTH
        display_height = settings.VIDEO_DISPLAY_HEIGHT

        if output_frame.shape[1] > display_width or output_frame.shape[0] > display_height:
            aspect_ratio = output_frame.shape[1] / output_frame.shape[0]
            if output_frame.shape[1] > display_width:
                new_width = display_width
                new_height = int(display_width / aspect_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * aspect_ratio)

            # Resize the frame to fit within the window
            output_frame = cv2.resize(output_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Display the frame in a window
        cv2.imshow('Livestream Output', output_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the process and close the OpenCV window
    process.stdout.close()
    process.wait()
    cv2.destroyAllWindows()


def livestream_executive_ui(url, app):
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

    # process = get_rtmp_frames(settings.RTMP_URL)

    print(f'Initiated FFmpeg process at time {time.ctime()}')
    # Set up FFmpeg process
    process = (
        ffmpeg
        .input(url, an=None)  # Disable audio
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{ui_input_variables.OUTPUT_FPS}')
        .global_args('-c:v', 'libx264')
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR)
    )
    print(f'FFmpeg process connected at time {time.ctime()}')

    app.process = process

    frame_size = settings.FRAME_WIDTH * settings.FRAME_HEIGHT * settings.NUM_CHANNELS

    while app.is_streaming:
        in_bytes = process.stdout.read(frame_size)

        if len(in_bytes) != frame_size:
            if not in_bytes:
                print("End of stream or error reading frame")
                break
            else:
                print("Error: Read incomplete frame")
                break

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy()

        if settings.MODEL_ON:

            in_frame = cv2.resize(in_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)
            segmentation_results_rgb = model_inference.image_to_tensor(Image.fromarray(in_frame),
                                                                       settings.MODEL, settings.DEVICE).astype(np.uint8)
            segmentation_results = segmentation_results_rgb

            _, segmentation_results = mask_postprocessing.apply_mask_postprocessing(in_frame, segmentation_results)

            if settings.SIDE_BY_SIDE:
                output_frame = np.vstack((in_frame.astype(np.uint8), segmentation_results.astype(np.uint8)))
            else:
                output_frame = segmentation_results
        else:
            output_frame = in_frame

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
            output_frame = cv2.resize(output_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        app.update_video_display(output_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    if process:
        process.stdin.close()  # Close the input pipe
        process.terminate()    # Terminate the FFmpeg process
        process.wait()         # Wait for the process to exit