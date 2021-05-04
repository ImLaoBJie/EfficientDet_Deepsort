import torch


# I/O ##################################################################################################################
# Video's path
video_src = "E:/workplace/EfficientDet_Deepsort/test/TESTVIDIODATA/demo001.mp4"  # set int to use webcam, set str to read from a video file
video_output = "E:/workplace/EfficientDet_Deepsort/test/TESTVIDIODATA/demo001_output.mp4"  # output to the specific position

text_output = "F:/video/00030_d0_output.csv"  # output to the file with the csv format

# DETECTOR #############################################################################################################
compound_coef = 4
force_input_size = None  # set None to use default size

threshold = 0.2
iou_threshold = 0.5

use_cuda = torch.cuda.is_available()
# use_cuda = True
use_float16 = False
cudnn_fastest = True
cudnn_benchmark = True

# coco_name
obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
# input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# TRACKER ##############################################################################################################
REID_CKPT = "./weights/ckpt.t7"
MAX_DIST = 0.2
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 0.5
MAX_IOU_DISTANCE = 0.7
MAX_AGE = 50  # as low as possible
N_INIT = 3
NN_BUDGET = 100


# TARGETS
selected_target = [obj_list.index('person'),
                   obj_list.index('bicycle'),
                   obj_list.index('car'),
                   obj_list.index('motorcycle'),
                   obj_list.index('bus'),
                   obj_list.index('truck')]
