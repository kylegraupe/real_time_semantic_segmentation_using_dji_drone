import ffmpeg
import numpy as np
import cv2


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


def livestream_1(url):
    """
    Establishes a livestream connection to the provided URL, reads video frames,
    and displays them in a named window until the 'q' key is pressed.

    Args:
        url (str): The URL of the livestream to connect to.

    Returns:
        None
    """
    # Get frame size
    width, height = get_frame_size(url)

    # Start the FFmpeg process
    process = (
        ffmpeg
        .input(url)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    # Create a named window
    cv2.namedWindow('RTMP Stream', cv2.WINDOW_NORMAL)

    while True:
        # Calculate frame size based on width and height
        frame_size = width * height * 3
        in_bytes = process.stdout.read(frame_size)

        # If no data is read, continue to check for errors
        if len(in_bytes) != frame_size:
            if not in_bytes:
                print("End of stream or error reading frame")
            else:
                print("Error: Read incomplete frame")
            break

        # Convert bytes to numpy array
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

        # Display the frame
        cv2.imshow('RTMP Stream', in_frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    process.stdout.close()
    process.wait()
    cv2.destroyAllWindows()

