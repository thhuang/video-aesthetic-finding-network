import os
import sys
sys.path.insert(0, '/app')
import argparse
import tensorflow as tf
import numpy as np
import skimage.transform
import vfn.network as nw
from vfn.vfn_eval import str2bool
from utils.time_counters import time_counters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', help='Embedding dimension before mapping to one-dimensional score', type=int, default = 1000)
    parser.add_argument('--initial_parameters', help="Path to initial parameter file", type=str, default='/app/vfn/alexnet.npy')
    parser.add_argument('--ranking_loss', help='Type of ranking loss', type=str, choices=['ranknet', 'svm'], default='svm')
    parser.add_argument('--snapshot', help='Name of the checkpoint files', type=str, default='/app/data/downloads/models/model-spp-max/model-spp-max')
    parser.add_argument('--spp', help='Whether to use spatial pyramid pooling in the last layer or not', type=str, default='True')
    parser.add_argument('--pooling', help='Which pooling function to use', type=str, choices=['max', 'avg'], default='max')
    parser.add_argument('--input', help='Path to the file with images for aesthetics score evaluation', type=str, default='/app/data/input/images')
    parser.add_argument('--input_video', help='Path to the video for aesthetics score evaluation', type=str)
    parser.add_argument('--report_path', help='Path to score report', type=str, default='/app/data/output/images')

    args = parser.parse_args()

    embedding_dim = args.embedding_dim
    ranking_loss = args.ranking_loss
    snapshot = args.snapshot
    net_data = np.load(args.initial_parameters, encoding = 'latin1').item()
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=[1,227,227,3])
    var_dict = nw.get_variable_dict(net_data)
    SPP = str2bool(args.spp)
    pooling = args.pooling
    images_dir = args.input
    video_dir = args.input_video
    report_dir = args.report_path

    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
        #score_func = nw.score(feature_vec)
        score_func = nw.score_1000(feature_vec)


    # load pre-trained model
    t0, _ = time_counters()
    print('---load pre-trained model---')
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, snapshot)
    t0, _ = time_counters(t0, '>>> load pre-trained model', print_time=True)

    # evaluate aesthetics
    print('---load images---')
    scores = list()
    index = np.array([])
    image_names = os.listdir(images_dir)
    for image_name in image_names:
        print('Evaluating {}'.format(image_name))
        image_dir = os.path.join(images_dir, image_name)
        print('Loading {}'.format(image_dir))
        img = skimage.io.imread(image_dir)
        img = img.astype(np.float32) / 255
        img_resize = skimage.transform.resize(img, (227, 227)) - 0.5
        if not len(img_resize.shape) == 3:
            continue
        img_resize = np.expand_dims(img_resize, axis=0)
        index = np.append(index, image_name)
        scores.append(sess.run([score_func], feed_dict={image_placeholder: img_resize}))
        #print(image_name, scores[-1])
    scores = np.squeeze(np.concatenate(scores, axis=1))
    print(scores.shape)
    t0, _ = time_counters(t0, '>>> evaluate aesthetics', print_time=True)

    # write score file
    print('---write score file---')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    report_fullpath = os.path.join(report_dir, 'score')
    np.save(report_fullpath, scores)
    print('scores is written to {}'.format(report_fullpath))
    t0, _ = time_counters(t0, '>>> write score file', print_time=True)