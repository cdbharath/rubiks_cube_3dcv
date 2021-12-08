# Author: Bharath Kumar
# Contact: kumar7bharath@gmail.com
# Reference: opencv.org

import cv2
import numpy as np
from collections import namedtuple
import glob
from kalman_filter import IP_KF

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, 
                   key_size = 12,     
                   multi_probe_level = 1) 

MIN_MATCH_COUNT = 10

PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

class planeTracker:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.targets = []
        self.frame_key_points = []
        self.kalman_p1 = IP_KF(0, 0)
        self.kalman_p2 = IP_KF(0, 0)
        self.kalman_p3 = IP_KF(0, 0)
        self.kalman_p4 = IP_KF(0, 0)
    
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

            x1, y1 = quad[0][0], quad[0][1]
            x2, y2 = quad[1][0], quad[1][1]
            x3, y3 = quad[2][0], quad[2][1]
            x4, y4 = quad[3][0], quad[3][1] 
            print(self.kalman_p1.update(x1, y1))

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
        self.cap = cv2.VideoCapture(-1)
        self.frame = None
        self.tracker = planeTracker()
        
        cv2.namedWindow("PlaneTracker")
        
        train_images = glob.glob("./data/train_images/test/*")
        train_images = sorted(train_images)
        print(train_images)

        with open("./data/train_images/rect.txt") as file:
            lines = file.readlines()
            lines = [line.split(',') for line in lines]

            for index, train_image in enumerate(train_images):
                line = [int(point) for point in lines[index]]
                print(line)
                train_image.split('/')[-1].split('.')[0]
                frame = cv2.imread(train_image)
                x, y, _  = frame.shape

                self.tracker.add_target(frame, (line[0], line[1], line[2], line[3]))
                # self.tracker.add_target(frame, (0, 0, y, x))
    
    def play(self):
        while True:
            ret, frame = self.cap.read()
            self.frame = frame.copy()
            frame = self.frame.copy()
            tracked = self.tracker.track(self.frame)
            for tr in tracked:
                cv2.polylines(frame, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                for (x, y) in np.int32(tr.p1):
                    cv2.circle(frame, (x, y), 2, (255, 255, 255))
                    
            # self.rect.draw(frame)
            cv2.imshow("PlaneTracker", frame)
            ret = cv2.waitKey(1)
            if ret == ord('q'):
                break

if __name__ == "__main__":
    player = VideoPlayer()
    player.play()