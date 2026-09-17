import cv2

url = "./Minecraft_stitch_test.mp4"

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

    # show frame
    cv2.imshow('frame', frame)

    # quit with q
    if cv2.waitKey(1) == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
