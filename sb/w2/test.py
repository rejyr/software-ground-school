import cv2
import numpy as np

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

print("frame matches len:", len(frame_matches))

# calculate homography matrices
Hs = []
for i1, i2, matches in frame_matches:
    kp1, _ = kps_des[i1]
    kp2, _ = kps_des[i2]
    # requires 4 point matches to solve Homography matrix's 8 DOF
    if len(matches) < 4:
        print(f"Not enough matches between frame {i1} and {i2}")
        continue

    # find matching 2d point coords
    # reshape resizes (n, 2) to (n, 1, 2) for cv2.findHomography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # compute homography matrix
    # homography matrix is 3x3 transformation matrix, maps views of same 2d plan in 3d space
    # cv2.RANSAC is random sample consensus filtering bad matches (moving foreground, etc)
    # 5.0 is reprojection threshold, if transformed point is within 5 pixels of matched point, it is an inlier
    # mask is 2d array where 1s are inliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    Hs.append(H)

    # count inliers vs matches
    inliers = np.sum(mask)
    print(f"frame {i1} to {i2}: computed H matrix with {inliers}/{len(matches)} inliers.")


# cleanup
cap.release()
cv2.destroyAllWindows()
