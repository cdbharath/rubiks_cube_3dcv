# HOW TO USE
# Choose points in this order
# top left -> top right -> bottom left -> bottom right

import cv2
import numpy as np

point_list = []
  
# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    global point_list
     
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' +
        #             str(y), (x,y), font,
        #             1, (255, 0, 0), 2)
        cv2.circle(img, (x,y), 3, (255, 255, 0), -1)
        cv2.imshow('image', img)
        point_list.append([x, y])
        
        if len(point_list) == 4:
            point_list = np.array(point_list, dtype='float32')
            output = np.array([[0,0],[0, 600],[600, 0],[600, 600]], dtype='float32') 
            matrix = cv2.getPerspectiveTransform(point_list, output)
            result = cv2.warpPerspective(img, matrix, (600, 600))
            result = cv2.resize(result, (300, 300))
            cv2.imwrite('../data/result.jpg', result)
             
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)
 
if __name__== "__main__":
 
    # reading the image
    img = cv2.imread('../data/train_images/6.jpg', 1)
 
    # displaying the image
    cv2.imshow('image', img)
 
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
 
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
 
    # close the window
    cv2.destroyAllWindows()