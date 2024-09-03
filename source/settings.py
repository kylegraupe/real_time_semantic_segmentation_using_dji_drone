import socket

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)


print(f"IP Address: {ip_address}")

# Application Environment
VERSION = '1.0.0'
ENVIRONMENT = 'development'
TRAIN = False

# RTMP/NGINX settings
LISTENING_PORT=1935
RTMP_URL=f'rtmp://{ip_address}:{LISTENING_PORT}/live/'


# DJI Mini 4 Pro Specs
FOV = 82.1

