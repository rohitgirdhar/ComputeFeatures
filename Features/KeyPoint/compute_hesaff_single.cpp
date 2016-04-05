// For a single image

#include "hesaff_base.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
  if (argc < 2 || argc >= 3) {
    cout << "./a.out <imgpath>" << endl;
    return -1;
  }
  Mat tmp = imread(argv[1]);
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
    // cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfAffinePoints << " affine shapes in " << getTime()-t1 << " sec." << endl;

    ostream out(std::cout.rdbuf());
    detector.exportKeypoints(out);
  }
  return 0;
}

