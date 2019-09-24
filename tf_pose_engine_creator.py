import logging
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt
import time


logger = logging.getLogger('TfPoseEstimator')
logger.handlers.clear()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def create_engine(graph_path, target_size=(320, 240), tf_config=None):

    # load graph
    logger.info('loading graph from %s(default size=%dx%d)' % (
        graph_path, target_size[0], target_size[1]))
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.get_default_graph()
    tf.import_graph_def(graph_def, name='TfPoseEstimator')
    output_nodes = ["Openpose/concat_stage7"]
    graph_def = trt.create_inference_graph(
        graph_def,
        output_nodes,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 20,
        precision_mode="FP16",
        minimum_segment_size=3,
        maximum_cached_engines=int(1e3),
    )

    with open('open_pose_trt.pb', 'wb') as f:
        f.write(graph_def .SerializeToString())


if __name__ == '__main__':
    graph_path = 'models/graph/cmu/graph_opt.pb'
    target_size = (272, 160)
    create_engine(graph_path, target_size)
