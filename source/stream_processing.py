import cv2
import settings

def live_stream():
    cap = cv2.VideoCapture(settings.RTMP_URL)

    if not cap.isOpened():
        print("Error: Unable to open video capture object")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from video stream")
            break

        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Error: Frame is empty or has invalid size")
            break

        cv2.imshow('GoPro RTMP Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()