import cv2

url = "./Minecraft_stitch_test.mp4"

orb = cv2.ORB_create()

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("cannot open video capture")
    exit()

while True:
    # frame by frame
    ret, frame = cap.read()

    # check frame is proper
    if not ret:
        print("can't receive frame")
        break

    # compute and annotate keypoints
    keypoints, descriptors = orb.detectAndCompute(frame, mask=None)
    frame_with_keypoints = cv2.drawKeypoints(
        frame,
        keypoints,
        outImage=None,
        color=(0, 0, 255), # wtf is this BGR
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # needs this to display circles and directions?
    )

    # show frame
    # cv2.imshow('frame', frame)
    cv2.imshow('frame with keypoints', frame_with_keypoints)

    # quit with q
    if cv2.waitKey(1) == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
