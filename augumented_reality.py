# Author: Bharath Kumar
# Contact: kumar7bharath@gmail.com
# Reference: opencv.org

import cv2
import numpy as np
from plane_tracker import planeTracker, SelectRect

# Simple model of a typical house (prism over cuboid)
ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                       [0, 0.5, 2], [1, 0.5, 2]])
ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]

class VideoPlayer:
    def __init__(self):
        self.cap = cv2.VideoCapture(2)
        self.frame = None
        self.tracker = planeTracker()
        
        cv2.namedWindow("PlaneTracker")
        cv2.createTrackbar('focal', 'PlaneTracker', 25, 50, self.empty)
        self.rect = SelectRect("PlaneTracker", self.rect_cb)
    
    def empty(*arg, **kw):
        pass
    
    def rect_cb(self, rect):
        self.tracker.add_target(self.frame, rect)
    
    def play(self):
        while True:
            if self.rect.tp_rect is None:
                ret, frame = self.cap.read()
                self.frame = frame.copy()
            frame = self.frame.copy()
            tracked = self.tracker.track(self.frame)
            for tr in tracked:
                cv2.polylines(frame, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                for (x, y) in np.int32(tr.p1):
                    cv2.circle(frame, (x, y), 2, (255, 255, 255))
                self.draw_model(frame, tr)
                    
            self.rect.draw(frame)
            cv2.imshow("PlaneTracker", frame)
            ret = cv2.waitKey(1)
            if ret == ord('q'):
                break
            
    def draw_model(self, image, tracked):
        x0, y0, x1, y1 = tracked.target.rect
        quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
        fx = 0.5 + cv2.getTrackbarPos('focal', 'PlaneTracker') / 50.0
        h, w = image.shape[:2]
        K = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0,0.0,      1.0]])
        dist_coef = np.zeros(4)
        _ret, rvec, tvec = cv2.solvePnP(quad_3d, tracked.quad, K, dist_coef)
        verts = ar_verts * [(x1-x0), (y1-y0), -(x1-x0)*0.3] + (x0, y0, 0)
        verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
        for i, j in ar_edges:
            (x0, y0), (x1, y1) = verts[i], verts[j]
            cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)


if __name__ == "__main__":
    player = VideoPlayer()
    player.play()