import os
import sys
sys.path.insert(0, '/app')
import argparse
import tensorflow as tf
import numpy as np
import skimage.transform
import skvideo.io
import vfn.network as nw
from vfn.vfn_eval import str2bool
from utils.time_counters import time_counters
from tqdm import tqdm
from datetime import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', help='Embedding dimension before mapping to one-dimensional score', type=int, default = 1000)
    parser.add_argument('--initial_parameters', help="Path to initial parameter file", type=str, default='/app/vfn/alexnet.npy')
    parser.add_argument('--ranking_loss', help='Type of ranking loss', type=str, choices=['ranknet', 'svm'], default='svm')
    parser.add_argument('--snapshot', help='Name of the checkpoint files', type=str, default='/app/data/downloads/models/model-spp-max/model-spp-max')
    parser.add_argument('--spp', help='Whether to use spatial pyramid pooling in the last layer or not', type=str, default='True')
    parser.add_argument('--pooling', help='Which pooling function to use', type=str, choices=['max', 'avg'], default='max')
    parser.add_argument('--input', help='Path to the file with images for aesthetics score evaluation', type=str, default='/app/data/input/images')
    parser.add_argument('--input_videos', help='Path to the file with videos for aesthetics score evaluation', type=str, default='/app/data/input/videos')
    parser.add_argument('--report_path', help='Path to score report', type=str, default='/app/data/output/videos')

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
    videos_dir = args.input_videos
    report_dir = args.report_path

    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
        #score_func = nw.score(feature_vec)
        score_func = nw.score_1000(feature_vec)

    # Load pre-trained model
    t0, _ = time_counters()
    print('---Load Pre-trained Model---')
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session(config=tf.ConfigProto())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, snapshot)
    time_counters(t0, '>>> Load pre-trained model', print_time=True)
    print()

    # Evaluate aesthetics
    print('---Extract Aesthetics Features---')
    start_time = datetime.now()

    video_names = os.listdir(videos_dir)
    num_videos = len(video_names)
    count = 1
    for video_name in video_names:
        t0, _ = time_counters()
        print('{:0>2}/{:0>2} Evaluating {}'.format(count, num_videos, video_name))
        count += 1
        video_dir = os.path.join(videos_dir, video_name)
        metadata = skvideo.io.ffprobe(video_dir)
        video_gen = skvideo.io.vreader(video_dir)
        features = list()
        count = 1

        for frame in tqdm(video_gen, total=int(metadata['video']['@nb_frames']), unit='frames'):
            #print('Evaluating frame {:0>7}'.format(count))
            img = frame.astype(np.float32) / 255
            img_resize = skimage.transform.resize(img, (227, 227), mode='reflect') - 0.5
            img_resize = np.expand_dims(img_resize, axis=0)
            features.append(sess.run([score_func], feed_dict={image_placeholder: img_resize}))
            count += 1
        features = np.squeeze(np.concatenate(features, axis=1))
        # write feature file
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        report_fullpath = os.path.join(report_dir, video_name)
        np.save(report_fullpath, features)
        print('({} X {}) features are written to {}'.format(features.shape[0], features.shape[1], report_fullpath))

    end_time = datetime.now()
    print('Training complete!')
    print('Time taken: {}'.format(end_time - start_time))




