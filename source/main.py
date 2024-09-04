import stream_processing
import settings
import user_interface

import time
import cProfile
import pstats
import tkinter as tk


def main():
    print(f'Application started at time {time.ctime()}')
    print(f'\tVersion: {settings.VERSION}')
    print(f'\tEnvironment: {settings.ENVIRONMENT}')
    print(f'\tRTMP URL: {settings.RTMP_URL}')
    print(f'\tIP Address: {settings.ip_address}')
    print(f'\tListening Port: {settings.LISTENING_PORT}')

    if settings.UI_ON:
        root = tk.Tk()
        app = user_interface.StreamApp(root, lambda: stream_processing.livestream_executive(settings.RTMP_URL, app))
        root.mainloop()
    else:
        stream_processing.livestream_executive(settings.RTMP_URL)


if __name__ == "__main__":

    if settings.SHOW_DEBUG_PROFILE:
        profiler = cProfile.Profile()
        profiler.enable()

        main()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
    else:
        main()
