import sys
import cv2
import glob
import time
import tensorflow as tf
from PIL import Image
import numpy as np
sys.path.append('/home/nvidia/custom-tf-pose')
from InferenceEngine import InferenceEngine

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.

    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]
    
def analyze_prediction(detection_out, pred_start_idx):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    if confidence > 0.5:
        # TODO: It should come from a mapping provided by THE VINIT
        class_name = label
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
#        boxes_utils.draw_bounding_boxes_on_image(
#            img_pil, np.array([[ymin, xmin, ymax, xmax]]),
#            display_str_list=["{}: {}".format(
#                class_name, confidence_percentage)],
#            color=coco_utils.COCO_COLORS[label]
#        )

def load_img(image_path):
        image = Image.open(image_path)
        model_input_width = 300
        model_input_height = 300
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        
        return img_np

def load_cv_image(image_path, width=None, height=None):
    val_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image

def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, 3)
        ).astype(np.uint8)

if __name__ == "__main__":
	img_paths = '/home/nvidia/frame*.jpg'
	images = [[load_img(img_path), load_cv_image(img_path)] for img_path in glob.glob(img_paths)]
	times = []
	
	# TensorFlow Pose Estimation
	w, h = model_wh('272x160')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_options)
	pose_estimator = TfPoseEstimator(get_graph_path('mobilenet_v2_large'),
                             target_size=(
        w, h), tf_config=config)
	time.sleep(1)
	object_detector = InferenceEngine('/home/nvidia/ssd_new.engine')

	# TensorRT object detection
	for img in images:
		for i in range(5):
			t1 = time.time()

            # Detect humans using the object detector
			[detection_out, keep_count_out] = object_detector.infer(img[0])
			prediction_fields = len(TRT_PREDICTION_LAYOUT)
			for det in range(int(keep_count_out[0])):
				analyze_prediction(detection_out, det * prediction_fields)

		    # Estimate the pose
            # XXX/ upsample_size upsamples the heatmap and vectors in the
            # model, prior to passing them to the post processor for mapping
            # keypoints to humans. The output of the post-processors need to
            # be parsed to get to the useful format
			humans = pose_estimator.inference(img[1],
                                              resize_to_default=(w > 0 and h >0),
                                              upsample_size=4.0)
			print('Humans {}'.format(humans))
			print('---------------------------------------------------------')
			t2 = time.time()
			times.append(t2-t1)
			#print(out)
	print(np.median(times))
	print(np.mean(times))
