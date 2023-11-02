#!/usr/bin/env python3
# -*-coding: utf-8-*-
import argparse
import torch
import numpy as np
import cv2
from time import time, sleep
from pathlib import Path


# Modified from source: Neel7317 https://github.com/ultralytics/yolov5/issues/2045
class OD:

    def __init__(
            self,
            capture_index,
            model_name,
            yolo_path,
            conf_thresh=0.35
    ):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.yolo = yolo_path
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf_thresh = conf_thresh
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            print(f"Attempting to load model {model_name}")
            model = torch.hub.load(self.yolo.resolve(), 'custom', str(model_name.resolve()), source='local')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= self.conf_thresh:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                cv2.putText(frame, f'conf: {row[4]}', (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255) , thickness=4)
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
        try:
            while True:
                # print("Get frame")
                ret, frame = cap.read()
                assert ret
                
                frame = cv2.resize(frame, (640,640))  # Consider removing or modifying this
                
                start_time = time()
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                
                end_time = time()
                fps = 1/np.round(end_time - start_time, 2)
                #print(f"Frames Per Second : {fps}")
                 
                cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                
                cv2.imshow('YOLOv5 Detection', frame)

                key = cv2.waitKey(1)
                if key == ord('q') & 0xFF:
                    break
                # sleep(0.01)
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', default='/home/hiwonder/yolov5', help='Path to YOLOv5 directory')
    parser.add_argument('weights_file', type=str ,help='Path to weights file.')
    args = parser.parse_args()
    
    weights = Path(args.weights_file).resolve()
    yolo = Path(args.yolo).resolve()
    assert weights.is_file()
    assert yolo.is_dir()
    # Create a new object and execute.

    detector = OD(capture_index=0, model_name=weights, yolo_path=yolo)
    detector()
