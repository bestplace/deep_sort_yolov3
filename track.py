#! /usr/bin/env python
# -*- coding: utf-8 -*-

from timeit import time
import cv2
import csv
import numpy as np
from PIL import Image
from .sort.sort import Sort

from .preprocessing import non_max_suppression


def dump_tracks(frame_index, tracks, csv_writer):
    rows = []
    for track in tracks:
        track_id, class_ = track[-2], track[-1]
        bbox = map(lambda x: str(int(x)), track[:-2])
        rows.append((str(frame_index),
                     int(track_id),
                     int(class_),
                     *bbox,
                     np.random.randint(7)))
    csv_writer.writerows(rows)


def track(yolo,
          videofile='test.mp4',
          csv_file='tracks.csv',
          out_videofile='output.avi',
          writeVideo_flag=False,
          show_video=False,
          verbose=False,
          skip=0):

    class_names = yolo.class_names

    nms_max_overlap = 1.0

    tracker = Sort(max_age=3, min_hits=1)

    video_capture = cv2.VideoCapture(videofile)

    w, h = int(video_capture.get(3)), int(video_capture.get(4))
    if writeVideo_flag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_videofile, fourcc, 60, (w, h))

    frame_index = -1
    res_file = open(csv_file, 'w')

    csv_writer = csv.writer(res_file)
    csv_writer.writerow(('frame_index',
                         'track_id',
                         'class_id',
                         'left',
                         'top',
                         'right',
                         'bottom',
                         'color'))

    fps = 0.0

    skip += 1

    end = False

    while not end:
        if verbose:
            start_time = time.time()
            t1 = time.time()

        for _ in range(skip):
            ret, frame = video_capture.read()
            if not ret:
                end = True
                break
            frame_index += 1
        if end:
            break

        if verbose:
            lasts = time.time() - start_time
            print('preproc: {0:.2f}s'.format(lasts), end='; ')
            start_time = time.time()
        frame = Image.fromarray(frame)
        bboxes = yolo.detect_image(frame)
        frame = np.array(frame)

        if verbose:
            lasts = time.time() - start_time
            print('inference: {0:.2f}s'.format(lasts), end='; ')
            start_time = time.time()

        # Run non-maxima suppression.
        boxes = np.array([d[2:] for d in bboxes])
        scores = np.array([d[1] for d in bboxes])
        indices = non_max_suppression(boxes, nms_max_overlap, scores)
        detections = np.array([bboxes[i] for i in indices])
        bboxes = []
        for detection in detections:
            bboxes.append(detection)

        if verbose:
            lasts = time.time() - start_time
            print('postproc: {0:.2f}s'.format(lasts), end='; ')
            start_time = time.time()
        frame = np.array(frame)
        if len(bboxes) == 0:
            if show_video:
                cv2.imshow('', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writeVideo_flag:
                out.write(frame)

            if verbose:
                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %f" % (fps), end='; ')
            continue

        bboxes = np.array(bboxes)
        bboxes[:, 4:] = bboxes[:, 4:] + bboxes[:, 2:4]

        tracks = tracker.update(bboxes[:, 2:], bboxes[:, 0])

        if show_video or writeVideo_flag:
            for track in tracks:
                track_id = track[-2]
                text = ':'.join((str(track_id), class_names[int(track[-1])]))
                bbox = track[:-1]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,  (0, 255, 0), 2)

        if show_video:
            cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)

        if len(tracks) > 0:
            dump_tracks(frame_index, tracks, csv_writer)

        if verbose:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %f" % (fps), end='; ')

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if verbose:
            lasts = time.time() - start_time
            print('writing: {0:.2f}s'.format(lasts), end='\n')

    video_capture.release()
    if writeVideo_flag:
        out.release()
    res_file.close()
    cv2.destroyAllWindows()
