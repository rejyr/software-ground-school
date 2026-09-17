import cv2
import numpy as np

# video URL
url = "./Minecraft_stitch_test.mp4"

# open video
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("cannot open video capture")
    exit()

# get frames
frames = []
while True:
    # frame by frame
    ret, frame = cap.read()

    # check frame is proper
    if not ret:
        print("can't receive frame")
        break

    frames.append(frame)

# downsample by taking every 4th pixel in x and y
# and every 8th frame
# question: how to choose every nth frame so that there is the right amount of overlap?
# ran out of memory so downsample
frames = list(map(lambda f: f[::4, ::4], frames[::8]))

# create stitcher
# Stitcher_SCANS for flat images/panning
stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)

# stitch that joint
status, stitched = stitcher.stitch(frames)

if status == cv2.Stitcher_OK:
    cv2.imwrite("stitched.png", stitched)
    print("stitched successfully")
else:
    print("stitching failed, status:", status)

# cleanup
cap.release()
cv2.destroyAllWindows()
