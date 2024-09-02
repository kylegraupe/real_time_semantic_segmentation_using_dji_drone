import stream_processing
import settings
import time

if __name__ == "__main__":
    print(f'Application started at time {time.ctime()}')
    print(f'\tVersion: {settings.VERSION}')
    print(f'\tEnvironment: {settings.ENVIRONMENT}')
    print(f'\tRTMP URL: {settings.RTMP_URL}')
    print(f'\tListening Port: {settings.LISTENING_PORT}')

    stream_processing.livestream_2(settings.RTMP_URL)