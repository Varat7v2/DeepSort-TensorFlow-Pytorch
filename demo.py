#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
from myVideoStream import myVideoCapture
from myPersonReIdNew_v2 import PERSON_REID
from myFROZEN_GRAPH_PERSON import FROZEN_GRAPH_INFERENCE
import os
warnings.filterwarnings('ignore')

VIDEO_CAPTURE = 2
writeVideo_flag = True
asyncVideo_flag = False

file_path = './videos/cctv.mp4'
output_filename = './output_fasterrcnn-101-cctv.avi'

def main():

    PATH_TO_CKPT_PERSON = 'models/faster_rcnn_restnet50.pb'

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = 200
    nms_max_overlap = 1.0
    yolo = YOLO()
    reid = PERSON_REID()
    frozen_person = FROZEN_GRAPH_INFERENCE(PATH_TO_CKPT_PERSON)
    
    # # Deep SORT
    # model_filename = 'model_data/mars-small128.pb'
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # file_path = 0
    if VIDEO_CAPTURE == 0 and asyncVideo_flag == True:
        video_capture = VideoCaptureAsync(file_path)
    elif VIDEO_CAPTURE == 1 and asyncVideo_flag == True:
        video_capture = myVideoCapture(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)
        im_width = int(video_capture.get(3))
        im_height = int(video_capture.get(4))

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    boxs = list()
    confidence = list()
    persons = list()
    frame_count = 0
    track_count = 0
    num_files = 0

    while True:
        t1 = time.time()
        ret, frame = video_capture.read()  # frame shape 640*480*3
        frame_org = frame.copy()
        if ret != True:
             break
        frame_count += 1
        # print('Frame count: {}'.format(frame_count))

        # Person detection using Frozen Graph
        persons = frozen_person.run_frozen_graph(frame, im_width, im_height)
        boxs = [[person['left'], person['top'], person['width'], person['height']] for person in persons]
        confidence = [person['confidence'] for person in persons]
        cropped_persons = list(person['cropped'] for person in persons)

        # # Person detection using YOLO - Keras-converted model
        # image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        # boxs = yolo.detect_image(image)[0]
        # confidence = yolo.detect_image(image)[1]
        # cropped_persons = [np.array(frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]) for box in boxs] #[x,y,w,h]

        # features = encoder(frame, boxs)
        if len(cropped_persons) > 0:
            features = reid.extract_feature_imgTensor(cropped_persons)
            # print(features.shape)
            detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxs, confidence, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                directory = os.path.join('output', str(track.track_id))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # file_count = len([name for name in os.listdir(directory+'/') if os.path.isfile(name)])
                file_count = sum([len(files) for root, dirs, files in os.walk(directory)])
                # print(file_count)

                if file_count == 0:
                    cv2.imwrite(directory+'/'+str(file_count+1)+'.jpg', frame_org[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                elif file_count > 0 and track_count % 10 == 0:
                    cv2.imwrite(directory+'/'+str(file_count+1)+'.jpg', frame_org[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                
                track_count += 1

            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2)
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                cv2.putText(frame, score + '%', (int(bbox[0]), int(bbox[3])), 0, 5e-3 * 130, (0,255,0),2)
            
        cv2.imshow('YOLO DeepSort', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        # print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    # print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
