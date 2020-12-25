import argparse
import imutils
import cv2
import sys
import aruco_generator


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
    args = vars(ap.parse_args())
    
    #loading image and resize it
    print("[INFO] loading image...")
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=600)

    #vetify that the supplied tag exists and is supported by OpenCV
    if aruco_generator.ARUCO_DICT.get(args["type"]) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
        sys.exit(0)

    #load the ArUCo dictionary, grab the ArUCo paramesters and detect the markers
    print("[INFO] detecting '{}' tages...".format(args["type"]))
    arucoDict = cv2.aruco.Dictionary_get(aruco_generator.ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

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

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            center = (int((topLeft[0] + bottomRight[0]) / 2.0), int((topLeft[1] + bottomRight[1]) / 2.0))

            cv2.circle(image, center, 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[INFO] ArUCo marker ID: {}".format(markerID))

    cv2.imshow("Image", image)
    cv2.waitKey(0)
