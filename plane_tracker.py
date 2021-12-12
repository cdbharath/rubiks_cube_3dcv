# Author: Bharath Kumar
# Contact: kumar7bharath@gmail.com
# Reference: opencv.org

import cv2
import numpy as np
from collections import namedtuple

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, 
                   key_size = 12,     
                   multi_probe_level = 1) 

MIN_MATCH_COUNT = 20

PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

class SelectRect:
    def __init__(self, window, cb):
        self.window = window
        self.callback = cb
        self.tp_rect_start = None
        self.tp_rect = None
        cv2.setMouseCallback(window, self.on_mouse)
    
    def on_mouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y])
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tp_rect_start = (x, y)
            return
        
        if self.tp_rect_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xtemp, ytemp = self.tp_rect_start
                x0, y0 = np.minimum([xtemp, ytemp], [x, y])
                x1, y1 = np.maximum([xtemp, ytemp], [x, y])
                if x1-x0 > 0  and y1-y0 >0:
                    self.tp_rect = (x0, y0, x1, y1)
            else:
                rect = self.tp_rect
                self.tp_rect = None
                self.tp_rect_start = None
                if rect:
                    self.callback(rect)
                
    def draw(self, out_frame):
        if not self.tp_rect:
            return False
        x0, y0, x1, y1 = self.tp_rect
        cv2.rectangle(out_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

class planeTracker:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.targets = []
        self.frame_key_points = []
    
    def track(self, frame):
        self. frame_key_points, frame_descriptors = self.detect_features(frame)
    
        if len(self.frame_key_points) < MIN_MATCH_COUNT:
            return []
        
        matches = self.matcher.knnMatch(frame_descriptors, k=2)
        
        matches = [m[0] for m in matches if len(m)==2 and m[0].distance < m[1].distance*0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
                    
        # For multiple reference images 
        matches_by_id = [[] for _ in range(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
            
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_key_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            mask = mask.ravel() != 0
            if mask.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[mask], p1[mask]
            
            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked
                    
    def add_target(self, image, rect):
        x0, y0, x1, y1 = rect
        _key_points, _descriptors = self.detect_features(image)
        key_points = []
        descriptors = []
        
        for kp, des in zip(_key_points, _descriptors):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                key_points.append(kp)
                descriptors.append(des)
        descriptors = np.uint8(descriptors)
        self.matcher.add([descriptors])
        target = PlanarTarget(image=image, rect=rect, keypoints=key_points, descrs=descriptors, data=None)
        self.targets.append(target)

    
    def detect_features(self, frame):
        key_points, descriptors = self.detector.detectAndCompute(frame, None)
        if descriptors is None:
            descriptors = []
        return key_points, descriptors

class VideoPlayer:
    def __init__(self):
        self.cap = cv2.VideoCapture(2)
        self.frame = None
        self.tracker = planeTracker()
        
        cv2.namedWindow("PlaneTracker")
        self.rect = SelectRect("PlaneTracker", self.rect_cb)
    
    def rect_cb(self, rect):
        cv2.imwrite("./data/result.jpg", self.frame)
        print(rect)
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
                    
            self.rect.draw(frame)
            cv2.imshow("PlaneTracker", frame)
            ret = cv2.waitKey(1)
            if ret == ord('q'):
                break

if __name__ == "__main__":
    player = VideoPlayer()
    player.play()