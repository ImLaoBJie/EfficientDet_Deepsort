
"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
from timeit import default_timer as timer

import torch
import cv2
import numpy as np
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from deep_sort import build_tracker

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video, xyxy_to_xywh
from config import *


class MOT(object):
    def __init__(self,
                 video_src: str,
                 video_output: str,
                 text_output: str,
                 obj_list: list,
                 input_sizes: list,
                 reid_cpkt: str,
                 compound_coef: int,
                 force_input_size=None,
                 threshold=0.2,
                 iou_threshold=0.2,
                 use_cuda=True,
                 use_float16=False,
                 cudnn_fastest=True,
                 cudnn_benchmark=True,

                 max_dist=0.2,
                 min_confidence=0.3,
                 nms_max_overlap=0.5,
                 max_iou_distance=0.7,
                 max_age=70,
                 n_init=3,
                 nn_budget=100,

                 selected_target=None):

        # I/O
        # Video's path
        self.video_src = video_src  # set int to use webcam, set str to read from a video file
        self.video_output = video_output  # output to the specific position
        # text path
        self.text_output = text_output  # output to the file with the csv format

        # DETECTOR
        self.compound_coef = compound_coef
        self.force_input_size = force_input_size  # set None to use default size

        self.threshold = threshold
        self.iou_threshold = iou_threshold

        self.use_cuda = use_cuda
        self.use_float16 = use_float16
        cudnn.fastest = cudnn_fastest
        cudnn.benchmark = cudnn_benchmark

        # coco_name
        self.obj_list = obj_list

        # input size
        self.input_sizes = input_sizes
        self.input_size = input_sizes[self.compound_coef] if force_input_size is None else force_input_size

        # load detector model
        model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(obj_list))
        model.load_state_dict(torch.load(f'weights/efficientdet-d{self.compound_coef}.pth'))
        model.requires_grad_(False)
        model.eval()

        if self.use_cuda and torch.cuda.is_available():
            self.detector = model.cuda()
        if self.use_float16:
            self.detector = model.half()

        # TRACKER
        self.reid_cpkt = reid_cpkt
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget

        # load tracker model,
        self.trackers = []
        self.selected_target = selected_target
        for num in range(0, len(self.selected_target)):
            self.trackers.append(build_tracker(reid_cpkt,
                                               max_dist,
                                               min_confidence,
                                               nms_max_overlap,
                                               max_iou_distance,
                                               max_age,
                                               n_init,
                                               nn_budget,
                                               use_cuda))
        # video frames
        self.frame_id = 0

    # function for video and text display
    def _display(self, preds, imgs, text_recorder=None, track_result=None):
        self.frame_id += 1

        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

        if len(preds['rois']) == 0:
            return imgs

        # get color
        obj_ids = preds['obj_ids']
        u, indices = np.unique(obj_ids, return_inverse=True)

        for j in range(len(preds['rois'])):
            # bbox
            (x1, y1, x2, y2) = preds['rois'][j].astype(np.int)
            # info
            cls_id = self.obj_list[preds['class_ids'][j]]
            obj_id = int(preds['obj_ids'][j])
            # color
            color = [int((p * (obj_id ** 2 - obj_id + 1)) % 255) for p in palette]
            cv2.rectangle(imgs, (x1, y1), (x2, y2), color, 2)

            cv2.putText(imgs, '{}, {}'.format(cls_id, obj_id),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

            # text recorded
            if text_recorder is not None:
                text_recorder.write(','.join([str(self.frame_id), str(obj_id), cls_id, str(x1), str(y1), str(x2), str(y2)]))
                text_recorder.write("\n")
                track_result[cls_id].add(obj_id)

        return imgs

    def detect_video(self):
        # Box
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        # Video capture
        cap = cv2.VideoCapture(self.video_src)
        if not cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # video recorder
        isVideoOutput = True if (self.video_output != "") or (self.video_output is not None) else False
        if isVideoOutput:
            # print("TYPE:", type(self.video_output), type(video_FourCC), type(video_fps), type(video_size))
            output2video = cv2.VideoWriter(self.video_output, video_FourCC, video_fps, video_size)

        # text recorder
        isTextOutput = True if (self.text_output != "") or (self.text_output is not None) else False
        if isTextOutput:
            output2text = open(self.text_output, 'w', encoding='utf-8')
            output2text.write("Frame,Obj_ID,Type,x1,y1,x2,y2\n")
            track_result = {}
            for obj_cls in self.obj_list:
                track_result[obj_cls] = set([])
        else:
            output2text = None
            track_result = None

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=self.input_size)

            if self.use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)

            # model predict
            with torch.no_grad():
                features, regression, classification, anchors = self.detector(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  self.threshold, self.iou_threshold)

            # detector result
            out = invert_affine(framed_metas, out)
            # out = [{[xyxy], [class], [scores]}, ...]
            bbox_xyxy = out[0]['rois']
            bbox_xywh = xyxy_to_xywh(bbox_xyxy)
            cls_ids = out[0]['class_ids']
            cls_conf = out[0]['scores']

            # tracker results
            # frame, class, conf, object identification
            tracker_out = {'rois': np.empty(shape=(0, 4)), 'class_ids': np.empty(shape=(0,), dtype=np.int),
                           'obj_ids': np.empty(shape=(0,), dtype=np.int)}
            for index, target in enumerate(self.selected_target):
                mask = cls_ids == target
                bbox = bbox_xywh[mask]
                conf = cls_conf[mask]

                outputs = self.trackers[index].update(bbox, conf, frame)

                if len(outputs) > 0:
                    tracker_out['rois'] = np.append(tracker_out['rois'], outputs[:, 0:4], axis=0)
                    tracker_out['class_ids'] = np.append(tracker_out['class_ids'], np.repeat(target, outputs.shape[0]))
                    tracker_out['obj_ids'] = np.append(tracker_out['obj_ids'], outputs[:, -1])

            # show bbox info and results
            img_show = self._display(tracker_out, ori_imgs[0], output2text, track_result)

            # show frame by frame
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            # show FPS
            cv2.putText(img_show, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 255, 0), thickness=2)
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", img_show)

            if isVideoOutput:
                output2video.write(img_show)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        output2text.close()
        for obj_cls in self.obj_list:
            print(obj_cls + ': ' + str(len(track_result[obj_cls])))


if __name__ == "__main__":
    detector = MOT(
                   video_src,
                   video_output,
                   text_output,
                   obj_list,
                   input_sizes,
                   REID_CKPT,
                   compound_coef,
                   force_input_size,
                   threshold,
                   iou_threshold,
                   use_cuda,
                   use_float16,
                   cudnn_fastest,
                   cudnn_benchmark,

                   MAX_DIST,
                   MIN_CONFIDENCE,
                   NMS_MAX_OVERLAP,
                   MAX_IOU_DISTANCE,
                   MAX_AGE,
                   N_INIT,
                   NN_BUDGET,

                   selected_target,)

    detector.detect_video()
