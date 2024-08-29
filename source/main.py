import stream_processing
import settings
import model_inference

if __name__ == "__main__":
    print('Hello, World!')

    stream_processing.livestream_2(settings.RTMP_URL)