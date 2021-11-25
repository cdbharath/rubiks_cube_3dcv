# Author: Bharath Kumar
# Contact: kumar7bharath@gmail.com

import cv2
import numpy as np
from enum import Enum

# cube center points in world frame
# x - right, y - up, z - away
TILE_SIZE = 1.6
TILES_X = 4
TILES_Y = 4
CUBE_CORNER_WF = np.array([[(TILE_SIZE/2 + y*TILE_SIZE, TILE_SIZE/2 + x*TILE_SIZE) for x in range(TILES_X)] for y in range(TILES_Y)])
COLOR_LIMITS = np.array([[(79, 0, 0), (99, 255, 255)],
                         [(0, 155, 0), (17, 255, 255)],
                         [(112, 155, 0), (129, 255, 255)],
                         [(18, 0, 0), (43, 255, 255)],
                         [(0, 0, 0), (179, 20, 255)]])

class Colors(Enum):
    GREEN = 4
    ORANGE = 3
    VIOLET = 2
    YELLOW = 1
    WHITE = 5

class CameraCallibration():
    def __init__(self, image):
        self.image = image 
    
    def get_tiles_centers(self):
        result_image = self.image
        image = self.separate_tile_colors()
        ret, thresh = cv2.threshold(image, 150, 255, 0)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 2)
        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)

            if M["m00"]:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            if 100 < area < 100000:
                cv2.circle(result_image, (cX, cY), 5, [255, 0, 0], -1)
        
        cv2.imshow("frame", result_image)
        cv2.waitKey(0)
        
    def separate_tile_colors(self):
        image = self.image
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        result = np.zeros(image.shape[:2], dtype="uint8")

        for color in range(len(COLOR_LIMITS)):
            low_thresh = np.array(COLOR_LIMITS[color][0])
            high_thresh = np.array(COLOR_LIMITS[color][1])
            mask = cv2.inRange(img_hsv, low_thresh, high_thresh)
            temp = cv2.GaussianBlur(mask, (25, 25), 1)
            temp = cv2.erode(temp, (25, 25))
            result = cv2.bitwise_or(result, temp)

        # cv2.imshow("frame", result)
        # cv2.waitKey(0)
    
        return result

def test_video():
    vid = cv2.VideoCapture("data/sample_video1.mp4")
    print("inside")
    while True:
        ret, frame = vid.read()
        callib = CameraCallibration(frame)
        erosion = callib.get_tiles_centers()
        
        cv2.imshow("output", erosion)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    vid.release()
    cv2.destroyAllWindows()
 

if __name__ == "__main__":
    image = cv2.imread("callib_data/1.jpg")
    callib = CameraCallibration(image)
    callib.get_tiles_centers()
#    callib.separate_tile_colors()
