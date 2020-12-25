import argparse
import sys
import time
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import aruco_generator

def get_ar_image(image, frame, position_display):
    """
    get ar image
    @param image ar image
    @param frame background has ArUCo code
    @param positionDisplay position of ArUCo corners
    @return image has been perspective warping with black background
    """
    size = image.shape

    position_src = np.array([
        [0, 0],
        [size[1] - 1, 0],
        [size[1] - 1, size[0] - 1],
        [0, size[0] - 1]
    ], dtype=float)

    h, status = cv2.findHomography(position_src, position_display)
    return cv2.warpPerspective(image, h, (frame.shape[1], frame.shape[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL",
                    help="type of ArUCo tag to generate")
    args = vars(ap.parse_args())
    #loading image and resize it

    #vetify that the supplied tag exists and is supported by OpenCV
    if aruco_generator.ARUCO_DICT.get(args["type"]) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
        sys.exit(0)

    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)
        #load the ArUCo dictionary, grab the ArUCo paramesters and detect the markers
        # print("[INFO] detecting '{}' tages...".format(args["type"]))
        arucoDict = cv2.aruco.Dictionary_get(aruco_generator.ARUCO_DICT[args["type"]])
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        #vetify "at least" one ArUco marker was detected
        if len(corners) > 0:
            # print("%r" % ids.flatten())
            for (markerCorner, markerID) in zip(corners, ids.flatten()):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))


                # cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                # cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                # cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                # cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                # center = (int((topLeft[0] + bottomRight[0]) / 2.0), int((topLeft[1] + bottomRight[1]) / 2.0))

                # cv2.circle(frame, center, 4, (0, 0, 255), -1)

                imDisplay = cv2.imread("../nearest_neibor.jpg")

                positionDisplay = np.array([topLeft, topRight, bottomRight, bottomLeft])
                #fill the ArUCo with black to simply add image to ArUCo
                cv2.fillConvexPoly(frame, positionDisplay.astype(int), 0, 16)
                frame = cv2.add(frame, get_ar_image(imDisplay, frame, positionDisplay))
                # cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print("[INFO] ArUCo marker ID: {}".format(markerID))

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()
