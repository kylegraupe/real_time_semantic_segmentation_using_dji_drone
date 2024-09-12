import time
import queue
import threading
import random
import sys

MAX_BUFFER_SIZE = 5

def simulate_frame_production(frame_queue):
    """Simulates frame production by adding frames to the queue."""
    while True:
        frame_queue.put(f"Frame-{random.randint(1, 1000)}")
        time.sleep(random.uniform(0.1, 0.5))  # Simulate variable frame production time


def custom_buffer(frames, buffer_queue):
    """
    Takes in frames as a NumPy array and stores them in a queue with a max size of 10.

    Args:
        frames (np.ndarray): A NumPy array containing frames to be added to the queue.
        buffer_queue (queue.Queue): The queue where frames will be stored.
    """
    # Loop through each frame in the provided frames array
    for frame in frames:
        # Check if the queue is full
        if buffer_queue.full():
            # Remove the oldest frame to make space (FIFO)
            buffer_queue.get()

        # Add the new frame to the queue
        buffer_queue.put(frame)


def display_queue_status(frame_queue):
    """Displays the queue status in an animated terminal output."""
    try:            # Clear the previous output
        sys.stdout.write("\033[K")
        sys.stdout.write(f"\rQueue size: {frame_queue.qsize()} frames")
        sys.stdout.flush()

        # Wait for a short period before updating
        time.sleep(0.5)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        sys.stdout.write("\nTerminating the display...\n")
        sys.stdout.flush()

if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=100)  # Define a queue with a maximum size of 10

    # Start a thread to simulate frame production
    producer_thread = threading.Thread(target=simulate_frame_production, args=(frame_queue,))
    producer_thread.daemon = True
    producer_thread.start()

    # Start displaying the queue status
    display_queue_status(frame_queue)