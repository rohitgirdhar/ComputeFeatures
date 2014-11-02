#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

caffe_root = '/exports/cyclops/software/vision/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imagesdir', type=str, required=True,
        help='Directory path with all the images to process')
parser.add_argument('-o', '--outputdir', type=str, required=True,
        help='Output directory')
parser.add_argument('-f', '--feature', type=str, default='prediction',
        help='could be prediction/fc7/pool5 etc')

args = parser.parse_args()
IMGS_DIR = args.imagesdir
OUT_DIR = os.path.join(args.outputdir, args.feature)
FEAT = args.feature

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = os.path.join('/exports/cyclops/work/001_Selfies/001_ComputeFeatures/Features/CNN/deploy.prototxt')
PRETRAINED = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
        channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))

net.set_phase_test()
net.set_mode_cpu()

files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(IMGS_DIR) for f in filenames]

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)
for fname in files:
    fpath = os.path.join(IMGS_DIR, fname)
    input_image = caffe.io.load_image(fpath)
    prediction = net.predict([input_image])
    if FEAT == 'prediction':
        feature = prediction.flat
    else:
        feature = net.blobs[FEAT].data[1]; # Computing only 1 crop, by def is center crop
        feature = feature.flat
    fileBaseName, fext = os.path.splitext(fname)
    fileBasePath, _ = os.path.splitext(fileBaseName)
    os.makedirs(fileBasePath)
    out_fpath = os.path.join(OUT_DIR, fileBaseName + '.txt')
    np.savetxt(out_fpath, feature, '%.7f')
    print 'Done for %s' % (fileBaseName)
