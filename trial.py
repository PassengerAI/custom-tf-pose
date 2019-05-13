import logging
import time
import glob
import cv2
import numpy as np
import sys
from InferenceEngine import InferenceEngine
import tensorflow as tf

import coco_datatypes.common

sys.path.append('/home/nvidia/tf-pose-estimation')
from tf_pose.tensblur.smoother import Smoother
from tf_pose.estimator import PoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common
logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def create_video(out_file_path, images_list, fps):
    frame_width = images_list[-1].shape[1]
    frame_height = images_list[-1].shape[0]
    out = cv2.VideoWriter(out_file_path, 
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          fps, (frame_width,frame_height))
    for frame in images_list:
        out.write(frame)
    out.release()

class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, target_size=(320, 240), tf_config=None):
        self.target_size = target_size

        # load graph
        logger.info('loading graph from %s(default size=%dx%d)' % (graph_path, target_size[0], target_size[1]))
        #with tf.gfile.GFile(graph_path, 'rb') as f:
        #    graph_def = tf.GraphDef()
        #    graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        #tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)

        # for op in self.graph.get_operations():
        #     print(op.name)
        # for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        #     print(ts)

        #self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        #self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.tensor_output = tf.placeholder(dtype=tf.float32, shape=[1, 34, 20, 57])
        self.tensor_heatMat = self.tensor_output[:, :, :, 38:]
        self.tensor_pafMat = self.tensor_output[:, :, :, :38]
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, 38:], self.upsample_size,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :38], self.upsample_size,
                                                     align_corners=False, name='upsample_pafmat')
        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                     tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None
        logger.info('Done loading graph, warming up now')
        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in
                                      self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        self.engine = InferenceEngine('/home/nvidia/cmu.engine')
        logger.info('1 warm ups done')
        #self.persistent_sess.run(
        #    [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
        #    feed_dict={
        #        self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
        #        self.upsample_size: [target_size[1], target_size[0]]
        #    }
        #)
        #logger.info('2 warm ups done')
        #self.persistent_sess.run(
        #    [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
        #    feed_dict={
        #        self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
        #        self.upsample_size: [target_size[1] // 2, target_size[0] // 2]
        #    }
        #)
        #logger.info('3 warm ups done')
        #self.persistent_sess.run(
        #    [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
        #    feed_dict={
        #        self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
        #        self.upsample_size: [target_size[1] // 4, target_size[0] // 4]
        #    }
        #)
        logger.info('all warm ups done')
        # logs
        #if self.tensor_image.dtype == tf.quint8:
        #    logger.info('quantization mode enabled.')

    def __del__(self):
        # self.persistent_sess.close()
        pass

    def get_flops(self):
        flops = tf.profiler.profile(self.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        return flops.total_float_ops

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            # draw point
            for i in range(coco_datatypes.common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

        return npimg

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1],
                                  window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, resize_to_default=True, upsample_size=1.0):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        #if self.tensor_image.dtype == tf.quint8:
        #    # quantize input image
        #    npimg = TfPoseEstimator._quantize_img(npimg)
        #    pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]
        logger.debug('img shape : {}'.format(np.shape(img)))
        #output = self.persistent_sess.run(
        #    [self.tensor_output], feed_dict={
        #        self.tensor_image: [img]
        #    }
        #)
        
        output = self.engine.infer(np.expand_dims(img, axis=1))
        logger.info('generated output {}'.format(np.shape(output)))
        output = output[0].reshape([57, 20, 34, 1]).transpose([3, 1, 2, 0])
        
        logger.info('generated output {}'.format(np.shape(output)))
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                self.tensor_output: output,  self.upsample_size: upsample_size
            })
        logger.info('generated peaks')
        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]
        
        for i in range(19):
            arr_ = np.squeeze(heatMat_up[0][:,:,i])
            logger.debug('MAX : {} MIN : {}, MEDIAN : {}'.format(np.max(arr_), np.min(arr_), np.median(arr_)))
            cv2.imwrite('/home/nvidia/Downloads/heatmap_{}.jpg'.format(i), arr_)
        
        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
            self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        logger.debug('estimate time=%.5f' % (time.time() - t))
        return humans
if __name__ == "__main__":
    resolution = '272x160'
    model = 'cmu'
    showBG = True
    logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    w, h = model_wh(resolution)
    logger.info('W and H : {} {}'.format(w, h))
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    count = 0
    limit = 100
    images = []
    fps_times = []
    for img_path in sorted(glob.glob('/home/nvidia/Downloads/frame_*.jpg')):
#         print(img_path)
            image = common.read_imgfile(img_path, None, None)
            for i in range(1):
                t = time.time()
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
                fps_time = time.time() - t
                fps_times.append(fps_time)
            if not showBG:
                image = np.zeros(image.shape)
        
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            images.append(image)
            count += 1
            if count >= limit:
                break
    logger.debug('finished+ {}'.format(1.0 / np.median(fps_times)))
    logger.debug(1.0 / np.median(fps_times))
    for i in range(len(images)):
        cv2.imwrite('/home/nvidia/Downloads/output_{}.jpg'.format(i), images[i])

