# Real Time Image Processing and Semantic Segmentation on DJI Drone via RTMP Server

This application takes a video stream from a DJI drone via RTMP Server and performs
image processing and semantic segmentation on the video stream.

SETUP (MacOS Apple Silicon):
- install NGINX with RTMP module: 'brew install nginx-full --with-rtmp'
- configure NGINX Configuration: 'sudo nano /opt/homebrew/etc/nginx/nginx.conf'
- set RTMP URL in settings.py file: 'RTMP_URL = "rtmp://your_ip_address:1935/live"'
  - Uses Port 1935, which is the default port, but can be changed in settings.py file.
- install OpenCV with FFMPEG Support (in IDE Terminal):
  - run: 'brew install ffmpeg'
  - run: 'brew install cmake git'
  - run: 'git clone https://github.com/opencv/opencv.git'
  - run: 'cd opencv'
  - run: 'mkdir opencv_build'
  - run: 'cd opencv_build'
  - run: 'cmake -DWITH_FFMPEG=ON -DFFMPEG_INCLUDE_DIRS=/usr/local/include/ffmpeg -DFFMPEG_LIBRARIES=/usr/local/lib/libavcodec.dylib ..'
  - run: 'make -j4'
  - run: 'sudo make install'
  - verify installation with 'print(cv2.getBuildInformation())' and check FFMPEG section.

DEBUGGING:
 - run: 'ffplay -f flv **_your_rtmp_url_**' to verify if stream is being sent via RTMP Server. 
 - run: 'sudo nano /opt/homebrew/etc/nginx/nginx.conf' to edit nginx.conf file.
   - nginx.conf file controls the functionality of the NGINX Server. There should be a block for the RTMP Server, which will specify the location of the Listening Port (typically 1935).

EXECUTION:
- open: Local RTMP Server application. This will facilitate the connection from the drone.
- navigate to transmission tab on DJI RC2. select 'Live Streaming Platforms' and select 'RTMP'.
  - currently working with high FPS and low latency on the following settings:
    - Frequency: 2.4 GHz
    - Channel Mode: Auto
    - Resolution: 720p (only option)
    - Bit Rate: 5 Mbps
