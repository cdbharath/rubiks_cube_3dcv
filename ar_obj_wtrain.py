# Author: Bharath Kumar
# Contact: kumar7bharath@gmail.com
# Reference: opencv.org

import cv2
import numpy as np
from plane_tracker import planeTracker, SelectRect
from load_obj import objLoader
import math
import glob

# Simple model of a typical house (prism over cuboid)
ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                       [0, 0.5, 2], [1, 0.5, 2]])
ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]

ar_faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 8), (7, 6, 9), (8, 9, 7, 4), (8, 9, 6, 5)]
color = [(0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (255, 255, 255), (255, 255, 255)]

class VideoPlayer:
    def __init__(self):
        self.cap = cv2.VideoCapture(-1)
        self.frame = None
        self.tracker = planeTracker()
        
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
        
        cv2.namedWindow("PlaneTracker")
        cv2.createTrackbar('focal', 'PlaneTracker', 25, 50, self.empty)
    
    def empty(*arg, **kw):
        pass
    
    def play(self):
        obj = objLoader("./ar_models/fox.obj", swapyz=True)
        
        while True:
            ret, frame = self.cap.read()
            self.frame = frame.copy()
            frame = self.frame.copy()
            tracked = self.tracker.track(self.frame)
            for tr in tracked:
                cv2.polylines(frame, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                for (x, y) in np.int32(tr.p1):
                    cv2.circle(frame, (x, y), 2, (255, 255, 255))
                # self.draw_model(frame, tr)
                frame = self.render_obj(frame, obj, tr)
                    
            # self.rect.draw(frame)
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
        
        # for i, j in ar_edges:
        #     (x0, y0), (x1, y1) = verts[i], verts[j]
        #     cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)
        
        for ind, face in enumerate(ar_faces):
            face_verts = []
            for i in face:
                face_verts.append(verts[i])
            face_verts = np.int32(face_verts)
            cv2.fillConvexPoly(image, face_verts, color[ind])
        
        
    def render_obj(self, image, obj, tracked, color=False):
        vertices = obj.vertices
        scale_matrix = np.eye(3)*1
        h, w = image.shape[:2]
        fx = 0.5 + cv2.getTrackbarPos('focal', 'PlaneTracker') / 50.0
        K = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0,0.0,      1.0]])
        
        # K = np.float64([[800, 0, 320], 
        #                 [0, 800, 240], 
        #                 [0, 0, 1]])

        projection = self.projection_matrix(K, tracked.H)

        x0, y0, x1, y1 = tracked.target.rect
        h, w = y1 - y0, x1 - x0
        
        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            # render model in the middle of the reference surface. To do so,
            # model points must be displaced
            points = np.array([[p[0] + w / 2 + x0, p[1] + h / 2 + y0, p[2]] for p in points])
            # points = np.array([[p[0], p[1], p[2]] for p in points])

            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(image, imgpts, (0, 0, 0))
            else:
                color = self.hex_to_rgb(face[-1])
                color = color[::-1]  # reverse
                cv2.fillConvexPoly(image, imgpts, color)
        return image
            
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.strip('#')
        h_len = len(hex_color)
        return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    
    @staticmethod
    def projection_matrix(camera_parameters, homography):
        # A[R1 R2 T] = H        
        homography = homography * (-1)
        rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
        col_1 = rot_and_transl[:, 0]
        col_2 = rot_and_transl[:, 1]
        col_3 = rot_and_transl[:, 2]
        # Normalize
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        # finally, compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return np.dot(camera_parameters, projection)



if __name__ == "__main__":
    player = VideoPlayer()
    player.play()