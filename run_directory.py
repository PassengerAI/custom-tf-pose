import argparse
import logging
import time
import glob
import ast
import os
import dill
import sys
sys.path.append('/home/nvidia/tf-pose-estimation/tf_pose')
import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import tensorflow as tf
logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='./images/')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), tf_config=config)

    files_grabbed = glob.glob(os.path.join(args.folder, '*.jpg'))
    all_humans = dict()
    for i, file in enumerate(files_grabbed):
        # estimate human poses from a single image !
        image = common.read_imgfile(file, None, None)
        t = time.time()
        humans = e.inference(image)
        elapsed = time.time() - t

        logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey(5)

        all_humans[file.replace(args.folder, '')] = humans

    with open(os.path.join(args.folder, 'pose.dil'), 'wb') as f:
        dill.dump(all_humans, f, protocol=dill.HIGHEST_PROTOCOL)
