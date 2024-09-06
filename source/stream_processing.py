"""
This script establishes the connection to the drone's RTMP stream, reads frames, and applies the semenatic segmentation
model to the frames. The segmented frames are then displayed in a named window until the 'q' key is pressed.
"""
import time
import ffmpeg
import numpy as np
import cv2
from PIL import Image
import settings
import model_inference
import torch
import torchcrf
from torchcrf import CRF
from skimage.util import img_as_float
import segmentation_postprocessing_library as seg_post_proc


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

        if settings.MODEL_ON:

            # Define the CRF layer with the number of classes
            if settings.CRF_ON:
                in_frame, segmentation_results = seg_post_proc.apply_crf(in_frame)

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
    print(f'Initiated FFmpeg process at time {time.ctime()}')
    # Set up FFmpeg process
    process = (
        ffmpeg
        .input(url, an=None)  # Audio Disabled in second parameter.
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=f'{settings.OUTPUT_FPS}')
        .global_args('-c:v', 'libx264')  # Use VideoToolbox for encoding
        .run_async(pipe_stdout=settings.PIPE_STDOUT, pipe_stderr=settings.PIPE_STDERR, pipe_stdin=True)
    )
    app.process = process
    print(f'FFmpeg process connected at time {time.ctime()}')

    frame_size = settings.FRAME_WIDTH * settings.FRAME_HEIGHT * settings.NUM_CHANNELS

    while app.is_streaming:
        in_bytes = process.stdout.read(frame_size)  # Read frame in byte format

        # If no data is read, continue to check for errors
        if len(in_bytes) != frame_size:
            if not in_bytes:
                print("End of stream or error reading frame")
                break
            else:
                print("Error: Read incomplete frame")
                break

        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]).copy()  # Convert to numpy array

        if settings.MODEL_ON:

            in_frame = cv2.resize(in_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)
            segmentation_results = model_inference.image_to_tensor(Image.fromarray(in_frame), settings.MODEL, settings.DEVICE).astype(np.uint8)


            if settings.SMALL_ITEM_FILTER_ON:
                segmentation_results = seg_post_proc.apply_conn(segmentation_results)

            if settings.CRF_ON:
                in_frame, segmentation_results = seg_post_proc.apply_crf(in_frame)

            if settings.EROSION_ON:
                segmentation_results = seg_post_proc.apply_erosion(segmentation_results)

            if settings.DILATION_ON:
                segmentation_results = seg_post_proc.apply_erosion(segmentation_results)

            if settings.GAUSSIAN_SMOOTHING_ON:
                segmentation_results = seg_post_proc.apply_gaussian_smoothing(segmentation_results)

            if settings.MEDIAN_FILTERING_ON:
                segmentation_results = seg_post_proc.apply_median_filtering(segmentation_results)


            # if settings.ACTIVE_CONTOURS_ON:
            #     segmentation_results = seg_post_proc.apply_active_contours(segmentation_results)

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
                output_frame = np.hstack((in_frame.astype(np.uint8), segmentation_results.astype(np.uint8)))
            else:
                output_frame = segmentation_results
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
    if process:
        process.stdin.close()  # Close the input pipe
        process.terminate()    # Terminate the FFmpeg process
        process.wait()         # Wait for the process to exit