#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

caffe_root = '../../../../../01_DepthRegression/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

parser = argparse.ArgumentParser()
parser.add_argument('--imagesdir', type=str, required=True,
        help='Directory path with all the images to process')
parser.add_argument('--outputdir', type=str, required=True,
        help='Output directory')
parser.add_argument('--feature', type=str, default='prediction',
        help='could be prediction/fc7')

args = parser.parse_args()
IMGS_DIR = args.imagesdir
OUT_DIR = os.path.join(args.outputdir, args.feature)
FEAT = args.feature

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
        channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

net.set_phase_test()
net.set_mode_cpu()

files = [f for f in os.listdir(IMGS_DIR) if os.path.isfile(
            os.path.join(IMGS_DIR, f))];

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)
for fname in files:
    fpath = os.path.join(IMGS_DIR, fname)
    input_image = caffe.io.load_image(fpath)
    prediction = net.predict([input_image])
    if FEAT == 'prediction':
        feature = prediction.flat
    else:
        feature = net.blobs[FEAT].data[4]; # index 4 is the center crop
        feature = feature.flat
    fileBaseName, fext = os.path.splitext(fname)
    out_fpath = os.path.join(OUT_DIR, fileBaseName + '.txt')
    np.savetxt(out_fpath, feature, '%.7f')
    print 'Done for %s' % (fileBaseName)
