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

        # Set the window to full-screen
        self.root.attributes('-fullscreen', True)

        # Disable resizing
        self.root.resizable(False, False)

        # Bind the Escape key to exit full-screen mode and close the app
        self.root.bind('<Escape>', self.exit_and_close)

        # Set the theme colors
        self.bg_color = "#1e1e2e"  # Dark purple
        self.fg_color = "#ffffff"  # White
        self.button_color = "#5a5a8b"  # Dark purple-grey
        self.button_text_color = "#000000"  # Black
        self.entry_bg_color = "#2b2b3c"  # Slightly lighter purple
        self.entry_fg_color = "#ffffff"  # White
        self.video_label_bg = "#000000"  # Black for video background

        # Apply the background color to the window
        self.root.configure(bg=self.bg_color)

        # Define video display dimensions
        self.video_display_width = 1280
        self.video_display_height = 704

        # Video display label (for the bottom half of the screen)
        self.video_label = tk.Label(root, bg=self.video_label_bg)
        self.video_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="s")

        # Create inputs for settings
        self.create_settings_inputs()

        # Execute button
        self.execute_button = tk.Button(root, text="Start Stream", command=self.start_stream,
                                        bg=self.button_color, fg=self.button_text_color)
        self.execute_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Close App button
        self.close_button = tk.Button(root, text="Close App", command=self.exit_and_close,
                                      bg=self.button_color, fg=self.button_text_color)
        self.close_button.grid(row=7, column=0, columnspan=2, pady=10)

    def create_settings_inputs(self):
        # Example: input for OUTPUT_FPS
        tk.Label(self.root, text="Output FPS:", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="e")
        self.output_fps_entry = tk.Entry(self.root, bg=self.entry_bg_color, fg=self.entry_fg_color)
        self.output_fps_entry.insert(0, str(settings.OUTPUT_FPS))
        self.output_fps_entry.grid(row=1, column=1, padx=10, pady=10)

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

    def exit_and_close(self, event=None):
        # Exit full-screen mode and close the application
        self.root.attributes('-fullscreen', False)
        self.root.quit()  # This will close the application


# Example usage:
# root = tk.Tk()
# app = StreamApp(root, start_stream_callback)
# root.mainloop()
