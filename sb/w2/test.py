import cv2

# video URL
url = "./Minecraft_stitch_test.mp4"

orb = cv2.ORB_create()

# open video
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("cannot open video capture")
    exit()

# get frames, keypoints, descriptors
# also interactively display them
frames = []
kps_des = []
while True:
    # frame by frame
    ret, frame = cap.read()

    # check frame is proper
    if not ret:
        print("can't receive frame")
        break

    frames.append(frame)

    # compute and annotate keypoints
    # keypoints: image coordinate of feature
    # descriptors: compact representation of feature (for comparison)
    keypoints, descriptors = orb.detectAndCompute(frame, mask=None)
    kps_des.append((keypoints, descriptors))

    # # annotate keypoints
    # frame_with_keypoints = cv2.drawKeypoints(
    #     frame,
    #     keypoints,
    #     outImage=None,
    #     color=(0, 0, 255), # wtf is this BGR
    #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # needs this to display circles and directions?
    # )
    # # show frame
    # # cv2.imshow('frame', frame)
    # cv2.imshow('frame with keypoints', frame_with_keypoints)
    #
    # # quit with q
    # if cv2.waitKey(1) == ord('q'):
    #     break

# start matching sequentially
# NORM_HAMMING for binary ORB descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
frame_matches = []
# n^2 brute force search over all kps and des
for i in range(len(frames) - 1):
    kp1, des1 = kps_des[i]
    kp2, des2 = kps_des[i + 1]

    if des1 is not None and des2 is not None:
        matches = bf.match(des1, des2)
        # sort matches by lowest distance (best match)
        matches = sorted(matches, key=lambda x: x.distance)
        frame_matches.append((i, i+1, matches))

print(frame_matches)


# cleanup
cap.release()
cv2.destroyAllWindows()
