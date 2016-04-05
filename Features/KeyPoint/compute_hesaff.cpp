/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include "hesaff_base.hpp"

int main(int argc, char **argv)
{
  if (argc < 3) {
    cerr << "Usage: ./prog <indir> <fileslist> <outdir>" << endl;
    return -1;
  }
  string indir = argv[1];
  string fileslist = argv[2];
  string outdir = argv[3];

  // read all images into list
  vector<string> imnames;
  ifstream fin(fileslist.c_str());
  string line;
  while (getline(fin, line)) {
    imnames.push_back(line);
  }
  fin.close();

  #pragma omp parallel for
  for (size_t i = 1; i <= imnames.size(); i++) { // IMP: 1 indexed
    Mat tmp = imread(indir + "/" + imnames[i - 1]);
    string outfpath = outdir + "/" + imnames[i - 1] + ".hesaff";
    if (!Locker::lock(outfpath)) {
      continue;
    }

    Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));

    float *out = image.ptr<float>(0);
    unsigned char *in  = tmp.ptr<unsigned char>(0); 

    for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
    {
      *out = (float(in[0]) + in[1] + in[2])/3.0f;
      out++;
      in+=3;
    }

    HessianAffineParams par;
    double t1 = 0;
    {
      // copy params 
      PyramidParams p;
      p.threshold = par.threshold;

      AffineShapeParams ap;
      ap.maxIterations = par.max_iter;
      ap.patchSize = par.patch_size;
      ap.mrSize = par.desc_factor;

      SIFTDescriptorParams sp;
      sp.patchSize = par.patch_size;

      AffineHessianDetector detector(image, p, ap, sp);
      t1 = getTime(); g_numberOfPoints = 0;
      detector.detectPyramidKeypoints(image);
      cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfAffinePoints << " affine shapes in " << getTime()-t1 << " sec." << endl;

      ofstream out(outfpath.c_str());
      detector.exportKeypoints(out);
    }
    Locker::unlock(outfpath);
  }
}
