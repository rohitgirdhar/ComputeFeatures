#!/usr/bin/python3.4

import numpy as np
import argparse
import os
import scipy.spatial

parser = argparse.ArgumentParser()
parser.add_argument('--featuresdir', type=str, required=True,
        help='Directory with all the features as files')
parser.add_argument('--outputdir', type=str, required=True,
        help='Directory where ouput will be stored')
args = parser.parse_args()
FEAT_DIR = args.featuresdir
OUT_DIR = args.outputdir

files = [f for f in os.listdir(FEAT_DIR) if os.path.isfile(os.path.join(FEAT_DIR, f))]
files = sorted(files)
nFiles = len(files)
scores = np.zeros([nFiles, nFiles])

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

for fname in files:
    fpath = os.path.join(FEAT_DIR, fname)
    feat = np.loadtxt(fpath, dtype=float, delimiter='\n')
    ds = np.empty(0)
    for fname2 in files:
        fpath2 = os.path.join(FEAT_DIR, fname2)
        feat2 = np.loadtxt(fpath2, dtype=float, delimiter='\n')
        dist = scipy.spatial.distance.cosine(feat, feat2)
        print(dist)
        np.append(ds, dist)
    np.savetxt(ds, os.path.join(OUT_DIR, fname), '\n')
    
