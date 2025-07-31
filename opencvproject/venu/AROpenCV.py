import cv2
import numpy as np

image_path = 'd:/opencvproject/venu/targetimage.jpg'
video_path = 'd:/opencvproject/venu/video.mp4'

# Load target image
imgTarget = cv2.imread(image_path)
if imgTarget is None:
    print("Target image not found!")
    exit()
hT, wT, _ = imgTarget.shape

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found!")
    exit()

# Setup video, but don't start reading it yet
myVid = cv2.VideoCapture(video_path)
if not myVid.isOpened():
    print("Video file not found!")
    exit()

# ORB feature detector
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
bf = cv2.BFMatcher()

video_playing = False  # Only start playing video after image is detected

while True:
    success, imgWebcam = cap.read()
    if not success:
        continue

    # Detect features in webcam frame
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    good = []

    if des1 is not None and des2 is not None:
        matches = bf.knnMatch(des1, des2, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        imgMatch = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

        if len(good) > 20:
            if not video_playing:
                myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind the video
                video_playing = True  # Flag to start video playback

            successVid, imgVideo = myVid.read()
            if not successVid or imgVideo is None:
                myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                successVid, imgVideo = myVid.read()

            imgVideo = cv2.resize(imgVideo, (wT, hT))

            srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

            if matrix is not None:
                imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
                pts = np.float32([[0, 0], [wT, 0], [wT, hT], [0, hT]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                imgWebcam = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 4)

                maskNew = np.zeros_like(imgWebcam)
                cv2.fillConvexPoly(maskNew, np.int32(dst), (255, 255, 255))
                maskInv = cv2.bitwise_not(maskNew)
                imgMasked = cv2.bitwise_and(imgWebcam, maskInv)
                imgAug = cv2.bitwise_or(imgWarp, imgMasked)

                cv2.imshow("Overlay with Video", imgAug)
            else:
                cv2.imshow("Overlay with Video", imgWebcam)

            cv2.imshow("Feature Matches", imgMatch)
        else:
            video_playing = False  # Stop video if image is not detected
            cv2.imshow("Overlay with Video", imgWebcam)
    else:
        video_playing = False
        cv2.imshow("Overlay with Video", imgWebcam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
myVid.release()
cv2.destroyAllWindows()
