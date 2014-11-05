#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, errno
import sys

def main():
    caffe_root = '/exports/cyclops/software/vision/caffe/'
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

    pwd = os.getcwd()
    os.chdir(IMGS_DIR)
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames]
    os.chdir(pwd)

    if not os.path.isdir(OUT_DIR):
        mkdir_p(OUT_DIR)

    count = 0
    for frpath in files:
        fpath = os.path.join(IMGS_DIR, frpath)
        fileBaseName, fext = os.path.splitext(frpath)
        fileBasePath, _ = os.path.split(fileBaseName)
        out_fpath = os.path.join(OUT_DIR, fileBaseName + '.txt')
        lock_fpath = os.path.join(OUT_DIR, fileBaseName + '.lock')

        # create the subdir to save output in
        outRelDir = os.path.join(OUT_DIR, fileBasePath)
        if not os.path.exists(outRelDir):
            mkdir_p(outRelDir)

        if os.path.exists(lock_fpath) or os.path.exists(out_fpath):
            print('Some other working on/done for %s\n' % fpath)
            continue
        
        mkdir_p(lock_fpath)
        input_image = caffe.io.load_image(fpath)
        prediction = net.predict([input_image])
        if FEAT == 'prediction':
            feature = prediction.flat
        else:
            feature = net.blobs[FEAT].data[0]; # Computing only 1 crop, by def is center crop
            feature = feature.flat

        np.savetxt(out_fpath, feature, '%.7f')
        count += 1
        
        os.rmdir(lock_fpath)
        print 'Done for %s (%d / %d)' % (fileBaseName, count, len(files))

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

if __name__ == '__main__':
    main()

