from __future__ import annotations
from pathlib import Path
from typing import Sequence

import argparse
import cv2
import datetime
import glob
import logging
import numpy as np
import os
import time
from ultralytics import YOLO


# from jtop import jtop # Use this to monitor compute usage (for Jetson Nano)

logging.getLogger().setLevel(logging.INFO)


# def draw_boxes(image, pred, classes, colors):
#     """Visualize YOLOv8 detection results"""
#     size=0
#     cls = None
#     width = 0
#     for r in pred:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             score = round(float(box.conf[0]), 2)
#             label = int(box.cls[0])

#             color = colors[label].tolist()
#             cls_name = classes[label]

#             cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(image, f"{cls_name} {score}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
#             if abs((x1-x2)*(y1-y2)) > size :
#                 size = abs((x1-x2)*(y1-y2))
#                 cls = cls_name
#                 width = abs(x1-x2)
#     return size, cls, width
'''
def draw_center_line(model,image):
    results = model.predict(image[:,50:,:], device = 0, half=True, imgsz = 640, stream = True)
    for r in results:
        pre_image = r.plot(boxes=False)
        point = r.masks.xy
        cls = r.boxes.cls.tolist()
        road_point = []
        car_point = []
        for i,v in enumerate(cls):
            if v == 2.0:
                road_point.append(point[i].astype(np.int32))
            else:
                car_point.append(point[i].astype(np.int32))
    mask_img = np.zeros_like(pre_image[:,:,0])
    w, h = mask_img.shape
    for x in range(len(road_point)):
        cv2.fillPoly(mask_img,[road_point[x]],(255,255,255))
    cv2.fillPoly(mask_img,car_point,(0,0,0))
    center_point = []
    for i in range(1,20):
        idx = w -20*i
        cont = []
        mask_img[idx][0] = 0
        mask_img[idx][h-1] = 0
        for j in range(1,len(mask_img[idx])):
            if mask_img[idx][j-1] != mask_img[idx][j]:
                cont.append(j)
        x_dist = 0
        c_point = 0
        if len(cont) > 1:
            for k in range(len(cont)//2):
                new_dist = cont[2*k+1]-cont[2*k]
                if new_dist > x_dist:
                    x_dist = new_dist
                    c_point = (cont[2*k+1]+cont[2*k])//2
            if x_dist > 100:
                center_point.append([c_point,idx])
    center_point = np.array(center_point).astype(np.int32)
    cv2.polylines(pre_image,[center_point],False,(255,0,0),2)
    return center_point, pre_image, mask_img
'''
def draw_center_line2(model,image):
    results = model.predict(image[:,50:,:], device= 0, half = True,  imgsz = 640, stream = True)
    for r in results:
        pre_image = r.plot(boxes=False)
        point = r.masks.xy
        cls = r.boxes.cls.tolist()
        road_point = []
        car_point = []
        for i,v in enumerate(cls):
            if v == 2.0:
                road_point.append(point[i].astype(np.int32))
            else:
                car_point.append(point[i].astype(np.int32))
    mask_img = np.zeros_like(pre_image[:,:,0])
    w, h = mask_img.shape
    for x in range(len(road_point)):
        cv2.fillPoly(mask_img,[road_point[x]],(255,255,255))
    cv2.fillPoly(mask_img,car_point,(0,0,0))
    center_point = []
    point_x =[]
    point_y =[]
    for i in range(1,20):
        idx = w -20*i
        cont = []
        mask_img[idx][0] = 0
        mask_img[idx][h-1] = 0
        for j in range(1,len(mask_img[idx])):
            if mask_img[idx][j-1] != mask_img[idx][j]:
                cont.append(j)
        x_dist = 0
        c_point = 0
        if len(cont) > 1:
            for k in range(len(cont)//2):
                new_dist = cont[2*k+1]-cont[2*k]
                if new_dist > x_dist:
                    x_dist = new_dist
                    c_point = (cont[2*k+1]+cont[2*k])//2
            if x_dist > 300:
                center_point.append([c_point,idx])
                point_x.append(c_point)
                point_y.append(idx)
    center_point = np.array(center_point).astype(np.int32)

    dim = 3
    poly = np.polyfit(point_y, point_x, dim)
    y_line = np.linspace(min(point_y), max(point_y), max(point_y)-min(point_y))
    x_pred = np.zeros_like(y_line)
    dx_pred = np.zeros_like(y_line)
    for i in range(dim+1): 
        x_pred += y_line ** (dim - i) * poly[i]
        if i == dim:
            pass
        else:
            dx_pred += y_line ** (dim-1 - i) * poly[i] * (dim - i)

        
    
    poly_point = []
    d_poly_point = []
    for i in range(len(y_line)):
        poly_point.append([x_pred[i],y_line[i]])
        d_poly_point.append([dx_pred[i],int(y_line[i])])
    poly_point = np.array(poly_point).astype(np.int32)
    d_poly_point = np.array(d_poly_point).astype(np.float32)

    cv2.polylines(pre_image,[center_point],False,(255,0,0),2)
    cv2.polylines(pre_image,[poly_point],False,(0,255,0),2)
    return center_point, poly_point,d_poly_point, pre_image, mask_img
'''           
def draw_center_line3(model,image):
    results = model.predict(image[:,50:,:], device = 0, half=True, imgsz = 640, stream = True)
    for r in results:
        pre_image = r.plot(boxes=False)
        point = r.masks.xy
        cls = r.boxes.cls.tolist()
        road_point = []
        car_point = []
        for i,v in enumerate(cls):
            if v == 2.0:
                road_point.append(point[i].astype(np.int32))
            else:
                car_point.append(point[i].astype(np.int32))
    mask_img = np.zeros_like(pre_image[:,:,0])
    w, h = mask_img.shape
    for x in range(len(road_point)):
        cv2.fillPoly(mask_img,[road_point[x]],(255,255,255))
    cv2.fillPoly(mask_img,car_point,(0,0,0))
    center_point = []
    for i in range(w//5):
        idx = 5*i
        cont = []
        mask_img[idx][0] = 0
        mask_img[idx][h-1] = 0
        for j in range(1,len(mask_img[idx])):
            if mask_img[idx][j-1] != mask_img[idx][j]:
                cont.append(j)
        x_dist = 0
        c_point = 0
        if len(cont) > 1:
            for k in range(len(cont)//2):
                new_dist = cont[2*k+1]-cont[2*k]
                if new_dist > x_dist:
                    x_dist = new_dist
                    c_point = (cont[2*k+1]+cont[2*k])//2
            if x_dist > 300:
                center_point.append([c_point,idx])
    center_point = np.array(center_point).astype(np.int32)
    cv2.polylines(pre_image,[center_point],False,(255,0,0),2)
    return center_point, pre_image, mask_img
'''
class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 1920,
        height: int = 1080,
        _width: int = 960,
        _height: int = 540,
        # width: int = 1640,
        # height: int = 1232,
        # _width: int = 820,
        # _height: int = 616,
        frame_rate: int = 10,
        flip_method: int = 0,
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = True,
        save: bool = True,
        log: bool = False,
    ) -> None:
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self._width = _width
        self._height = _height
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log
        self.model = None

        # Check if OpenCV is built with GStreamer support
        # print(cv2.getBuildInformation())

        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence) and len(sensor_id) > 1:
            raise NotImplementedError("Multiple cameras are not supported yet")

        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        self.cap = [cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=id, flip_method=0), \
                    cv2.CAP_GSTREAMER) for id in self.sensor_id]

        # Make record directory
        if save:
            assert save_path is not None, "Please provide a save path"
            os.makedirs(self.save_path, exist_ok=True) # if path does not exist, create it
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)

            logging.info(f"Save directory: {self.save_path}")

    def gstreamer_pipeline(self, sensor_id: int, flip_method: int) -> str:
        """
        Return a GStreamer pipeline for capturing from the CSI camera
        """
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                self.width,
                self.height,
                self.frame_rate,
                flip_method,
                self._width,
                self._height,
            )
        )
    def set_model(self, model: YOLO, classes: dict) -> None:
        """
        Set a YOLO model
        """
        self.model = model
        self.classes = classes                
        self.colors = np.random.randint(0,255,size=(len(self.classes), 3))
        self.colors = self.colors.astype(np.uint8)
        self.visualize_pred_fn = lambda img, pred: draw_boxes(img, pred, self.classes, self.colors)

    def run(self) -> None:
        """
        Streaming camera feed
        """
        if self.stream:
            cv2.namedWindow(self.window_title)

        if self.cap[0].isOpened():
            try:
                while True:
                    t0 = time.time()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    _, frame = self.cap[0].read()


                    # frame = cv2.undistort(frame,mtx,dist,None,newcameramtx)
                    frame = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

                    if self.model:
                        # center_points,pre_frame, mask_img = draw_center_line(self.model,frame)
                        center_points, p,dp,pre_frame, mask_img = draw_center_line2(self.model,frame)
                        
                        # center_points,pre_frame, mask_img = draw_center_line3(self.model,frame)


                    if self.save:
                        cv2.imwrite(str(self.save_path / f"{timestamp}.jpg"), pre_frame)

                    if self.log:
                        print(f"FPS: {1 / (time.time() - t0):.2f}")

                        

                        

                    if self.stream:
                        cv2.imshow(self.window_title, pre_frame)

                        if cv2.waitKey(1) == ord('q'):
                            break

            except Exception as e:
                print(e)
            finally:
                self.cap[0].release()
                cv2.destroyAllWindows()

    @property
    def frame(self) -> np.ndarray:
        """
        !!! Important: This method is not efficient for real-time rendering !!!

        [Example Usage]
        ...
        frame = cam.frame # Get the current frame from camera
        cv2.imshow('Camera', frame)
        ...

        """
        if self.cap[0].isOpened():
            return self.cap[0].read()[1]
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--sensor_id',
        type = int,
        default = 0,
        help = 'Camera ID')
    args.add_argument('--window_title',
        type = str,
        default = 'Camera',
        help = 'OpenCV window title')
    args.add_argument('--save_path',
        type = str,
        default = 'record',
        help = 'Image save path')
    args.add_argument('--save',
        action = 'store_true',
        help = 'Save frames to save_path')
    args.add_argument('--stream',
        action = 'store_true',
        help = 'Launch OpenCV window and show livestream')
    args.add_argument('--log',
        action = 'store_true',
        help = 'Print current FPS')
    args = args.parse_args()

    cam = Camera(
        sensor_id = args.sensor_id,
        window_title = args.window_title,
        save_path = args.save_path,
        save = args.save,
        stream = args.stream,
        log = args.log)
        
    model = YOLO("./roadseg.engine", task = 'segment')
    cam.model = model
    # img = cv2.imread('/home/ircv/HYU-2024-Embedded/team2/run/record/000066/20240607100228745708.jpg')
    # results = model(img)
    # print(results[0])
    Cameramtx = np.load('Cameramtx.npz')
    mtx = Cameramtx['mtx']
    dist = Cameramtx['dist']
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx,dist,(960,540),0)
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(960,540),5)
    cam.run()
