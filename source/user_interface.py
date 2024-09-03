import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import settings


class StreamApp:
    def __init__(self, root, start_stream_callback):
        self.root = root
        self.root.title("RTMP Stream GUI")
        self.start_stream_callback = start_stream_callback

        # Video display label
        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Create inputs for settings
        self.create_settings_inputs()

        # Execute button
        self.execute_button = tk.Button(root, text="Start Stream", command=self.start_stream)
        self.execute_button.grid(row=5, column=0, columnspan=2, pady=10)

    def create_settings_inputs(self):
        # Example: input for OUTPUT_FPS
        tk.Label(self.root, text="Output FPS:").grid(row=1, column=0, sticky="e")
        self.output_fps_entry = tk.Entry(self.root)
        self.output_fps_entry.insert(0, str(settings.OUTPUT_FPS))
        self.output_fps_entry.grid(row=1, column=1, padx=10)

        # Add more inputs as needed
        # ...

    def start_stream(self):
        # Update settings based on user inputs
        settings.OUTPUT_FPS = float(self.output_fps_entry.get())
        # Update other settings similarly

        # Start the livestream in a new thread
        stream_thread = threading.Thread(target=self.start_stream_callback)
        stream_thread.start()

    def update_video_display(self, output_frame):
        # Convert to ImageTk format
        img = Image.fromarray(output_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video display
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
