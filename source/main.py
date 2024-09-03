"""
This file contains the main code for the application. Run to execute the application.
"""

import stream_processing
import settings
import time

if __name__ == "__main__":
    print(f'Application started at time {time.ctime()}')
    print(f'\tVersion: {settings.VERSION}')
    print(f'\tEnvironment: {settings.ENVIRONMENT}')
    print(f'\tRTMP URL: {settings.RTMP_URL}')
    print(f'\tIP Address: {settings.ip_address}')
    print(f'\tListening Port: {settings.LISTENING_PORT}')

    stream_processing.livestream_executive(settings.RTMP_URL)