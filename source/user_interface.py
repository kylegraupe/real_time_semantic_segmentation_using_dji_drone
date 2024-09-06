import tkinter as tk
from PIL import Image, ImageTk
import threading
import settings


class StreamApp:
    def __init__(self, root, start_stream_callback, stop_stream_callback):
        # Set the theme colors
        self.bg_color = "#1e1e2e"  # Dark purple
        self.fg_color = "#ffffff"  # White
        self.button_color = "#5a5a8b"  # Dark purple-grey
        self.button_text_color = "#000000"  # Black
        self.entry_bg_color = "#2b2b3c"  # Slightly lighter purple
        self.entry_fg_color = "#ffffff"  # White
        self.video_label_bg = "#000000"  # Black for video background

        self.root = root
        self.root.title("RTMP Stream GUI")
        self.start_stream_callback = start_stream_callback
        self.stop_stream_callback = stop_stream_callback
        self.stream_thread = None
        self.is_streaming = False
        self.process = None

        # Set initial window size to the screen's resolution
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")

        # Enable resizing for macOS full-screen functionality
        self.root.resizable(True, True)

        # Allow full screen with the green button and bind the Escape key to exit full screen
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<Escape>', self.exit_fullscreen)


        # Apply the background color to the window
        self.root.configure(bg=self.bg_color)

        # Load and display the logo
        self.load_logo()

        # Create the sidebar frame
        self.sidebar_frame = tk.Frame(self.root, bg=self.bg_color, width=200, padx=10, pady=10)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="ns")

        # Create the input settings frame
        self.input_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, relief="raised")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create the button frame
        self.button_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, relief="raised")
        self.button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Create the video display frame
        self.video_frame = tk.Frame(self.root, bg=self.video_label_bg, bd=2, relief="raised")
        self.video_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Create the stream characteristics frame
        self.stream_char_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, relief="raised")
        self.stream_char_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Create the grid layout
        self.create_input_settings()
        self.create_buttons()
        self.create_video_display()
        self.create_stream_characteristics()

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

    def load_logo(self):
        # Load and display the logo
        logo_path = "/Users/kylegraupe/Documents/Programming/GitHub/Computer Vision Dataset Generator/real_time_semantic_segmentation_using_dji_drone/assets/graupe.io logo 1.png"
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((250, 150))  # Adjust size as needed
        self.logo_imgtk = ImageTk.PhotoImage(logo_image)

        # Create a label for the logo
        self.logo_label = tk.Label(self.root, image=self.logo_imgtk, bg=self.bg_color, bd=2, relief="solid")

        # Position the logo with margins
        self.logo_label.grid(row=2, column=0, padx=20, pady=20)  # Margin of 20 pixels from the left and bottom

    def create_input_settings(self):
        # Input settings for various parameters
        tk.Label(self.input_frame, text="Output FPS:", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w")
        self.output_fps_entry = tk.Entry(self.input_frame, bg=self.entry_bg_color, fg=self.entry_fg_color)
        self.output_fps_entry.insert(0, str(settings.OUTPUT_FPS))
        self.output_fps_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Add more inputs as needed
        # ...

    def create_buttons(self):
        # Set the button width to be slightly less than the frame width
        button_width = 30  # Adjust this value as needed for your layout

        # Execute button
        self.execute_button = tk.Button(self.button_frame, text="Start Stream", command=self.start_stream,
                                        bg=self.button_color, fg=self.button_text_color, bd=1, relief="raised",
                                        width=button_width)
        self.execute_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")  # Use sticky="ew" to center

        # Stop button
        self.stop_button = tk.Button(self.button_frame, text="Stop Stream", command=self.stop_stream,
                                     bg=self.button_color, fg=self.button_text_color, state=tk.DISABLED, bd=1,
                                     relief="raised", width=button_width)
        self.stop_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")  # Use sticky="ew" to center

        # Close App button
        self.close_button = tk.Button(self.button_frame, text="Close App", command=self.exit_and_close,
                                      bg=self.button_color, fg=self.button_text_color, bd=1, relief="raised",
                                      width=button_width)
        self.close_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")  # Use sticky="ew" to center

    def create_video_display(self):
        # Video display label (for the video frame)
        self.video_label = tk.Label(self.video_frame, bg=self.video_label_bg)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def start_stream(self):
        # Update settings based on user inputs
        settings.OUTPUT_FPS = float(self.output_fps_entry.get())
        # Update other settings similarly

        # Start the livestream in a new thread
        if not self.is_streaming:
            self.is_streaming = True
            self.stop_button.config(state=tk.NORMAL)
            self.stream_thread = threading.Thread(target=self.start_stream_callback)
            self.stream_thread.start()

    def create_stream_characteristics(self):
        # Stream characteristics (e.g., resolution, bitrate)
        tk.Label(self.stream_char_frame, text="Stream Characteristics:", underline=True, bg=self.bg_color, fg=self.fg_color).grid(row=0,
                                                                                                                  column=0,
                                                                                                                  sticky="w")

        # Add stream characteristics labels and values
        tk.Label(self.stream_char_frame, text=f"Resolution: {settings.RESIZE_FRAME_WIDTH}x{settings.RESIZE_FRAME_HEIGHT}"
                 , bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="w")

        tk.Label(self.stream_char_frame, text=f"Listening Port: {str(settings.LISTENING_PORT)}", bg=self.bg_color, fg=self.fg_color).grid(row=2,
                                                                                                             column=0,
                                                                                                             sticky="w")
        tk.Label(self.stream_char_frame, text=f"Input FPS: {str(settings.INPUT_FPS)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=3, column=0, sticky="w")

        tk.Label(self.stream_char_frame, text=f"Output FPS: {str(settings.OUTPUT_FPS)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=4, column=0, sticky="w")

        tk.Label(self.stream_char_frame, text=f"Model Encoder: {str(settings.MODEL_ENCODER_NAME)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=3, column=1, sticky="w")

        tk.Label(self.stream_char_frame, text=f"Model Decoder: {str(settings.MODEL_DECODER_NAME)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=4, column=1, sticky="w")

        tk.Label(self.stream_char_frame, text="RGB Mask Post-Processing:", underline=True, bg=self.bg_color, fg=self.fg_color).grid(row=0,
                                                                                                                  column=2,
                                                                                                                  sticky="w")

        tk.Label(self.stream_char_frame, text=f"Dilation: {str(settings.DILATION_ON)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=1, column=2, sticky="w")

        tk.Label(self.stream_char_frame, text=f"Erosion: {str(settings.EROSION_ON)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=2, column=2, sticky="w")

        tk.Label(self.stream_char_frame, text=f"Median Blur: {str(settings.MEDIAN_FILTERING_ON)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=3, column=2, sticky="w")

        tk.Label(self.stream_char_frame, text=f"Gaussian Blur: {str(settings.GAUSSIAN_SMOOTHING_ON)}",
                 bg=self.bg_color, fg=self.fg_color).grid(row=4, column=2, sticky="w")

    def stop_stream(self):
        # Stop the stream
        if self.is_streaming:
            self.is_streaming = False
            if self.process:
                self.process.stdin.close()  # Close the input pipe
                self.process.terminate()    # Terminate the FFmpeg process
                self.process.wait()         # Wait for the process to exit
                self.process = None
            self.stream_thread.join()     # Wait for the stream thread to exit
            self.stop_button.config(state=tk.DISABLED)

    def update_video_display(self, output_frame):
        # Convert to ImageTk format
        img = Image.fromarray(output_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video display
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def exit_and_close(self, event=None):
        # Stop the stream if it's running
        self.stop_stream()
        # Exit the application
        self.root.quit()  # This will close the application

    def toggle_fullscreen(self, event=None):
        # Toggle full-screen mode
        self.root.attributes("-fullscreen", True)

    def exit_fullscreen(self, event=None):
        # Exit full-screen mode
        self.root.attributes("-fullscreen", False)

