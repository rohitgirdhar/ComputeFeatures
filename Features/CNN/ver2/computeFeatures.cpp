/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include "caffe/caffe.hpp"
#include "utils.hpp"
//#include "external/DiskVector/DiskVector.hpp"
#include "external/DiskVector/DiskVectorLMDB.hpp"
#include "lock.hpp"

using namespace std;
using namespace std::chrono;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

#define MAXFEATPERIMG 10000
#define PRINT_INTERVAL 20
// output type
#define OUTTYPE_TEXT 1
#define OUTTYPE_LMDB 2

void dumpFeature(FILE*, const vector<float>&);
long long hashCompleteName(long long, int);
template<typename Dtype>
void computeFeaturesPipeline(Net<Dtype>& caffe_test_net,
    const vector<Mat>& Is,
    const vector<string>& layers,
    int BATCH_SIZE,
    vector<vector<vector<Dtype>>>& output,
    bool verbose,
    const string& POOLTYPE,
    bool NORMALIZE);

int
main(int argc, char *argv[]) {
  ::google::InitGoogleLogging(argv[0]);
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  LOG(INFO) << "Extracting Features in CPU mode";
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("network-path,n", po::value<string>()->required(),
     "Path to the prototxt file")
    ("model-path,m", po::value<string>()->required(),
     "Path to corresponding caffemodel")
    ("outdir,o", po::value<string>()->default_value("output"),
     "Output directory")
    ("layer,l", po::value<string>()->default_value("pool5"),
     "CNN layer to extract features from")
    ("imgsdir,i", po::value<string>()->required(),
     "Input directory of images")
    ("imgslst,q", po::value<string>()->required(),
     "List of images relative to input directory")
    ("windir,w", po::value<string>()->default_value(""),
     "Input directory of all windows in each image (selective search format: y1 x1 y2 x2)." 
     "Defaults to full image features."
     "Ignores sliding window if set.")
    ("sliding,s", po::bool_switch()->default_value(false),
     "Compute features in sliding window fashion")
    ("pool,p", po::value<string>()->default_value(""),
     "Pool the features from different regions into one feature."
     "Supports: <empty>: no pooling. store all features."
     "avg: avg pooling")
    ("segdir,x", po::value<string>()->default_value(""),
     "Directory with images with same filename as in the corpus images dir "
     "but uses it to prune the set of windows. "
     "By default keeps only those overlapping <0.2 with FG")
    ("startimgid,z", po::value<int>()->default_value(1),
     "The image id of the first image in the list."
     "Useful for testing parts of dataset because the selsearch boxes" 
     "etc use the image ids. Give 1 indexed")
    ("output-type,t", po::value<string>()->default_value("lmdb"),
     "Output format [txt/lmdb]")
    ("normalize,y", po::bool_switch()->default_value(false),
     "Enable feature L2 normalization")
    ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help")) {
    LOG(INFO) << desc;
    return -1;
  }
  try {
    po::notify(vm);
  } catch(po::error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  fs::path NETWORK_PATH = fs::path(vm["network-path"].as<string>());
  fs::path MODEL_PATH = 
    fs::path(vm["model-path"].as<string>());
  string LAYERS = vm["layer"].as<string>();
  vector<string> layers;
  boost::split(layers, LAYERS, boost::is_any_of(","));
  fs::path OUTDIR = fs::path(vm["outdir"].as<string>());
  fs::path IMGSDIR = fs::path(vm["imgsdir"].as<string>());
  fs::path IMGSLST = fs::path(vm["imgslst"].as<string>());
  fs::path WINDIR = fs::path(vm["windir"].as<string>());
  fs::path SEGDIR = fs::path(vm["segdir"].as<string>());
  string POOLTYPE = vm["pool"].as<string>();
  bool NORMALIZE = vm["normalize"].as<bool>();
  int START_IMGID = vm["startimgid"].as<int>();
  int OUTTYPE = -1;
  if (vm["output-type"].as<string>().compare("text") == 0) {
    OUTTYPE = OUTTYPE_TEXT;
  } else if (vm["output-type"].as<string>().compare("lmdb") == 0) {
    OUTTYPE = OUTTYPE_LMDB;
  } else {
    LOG(FATAL) << "Unknown output-type " << vm["output-type"].as<string>();
  }

  if (SEGDIR.string().length() > 0 && fs::exists(SEGDIR)) {
    LOG(INFO) << "Will be pruning the bounding boxes using "
              << "segmentation information";
  } else {
    SEGDIR = fs::path(""); // so that I don't need to check existance again
  }
  if (POOLTYPE.length() > 0) {
    LOG(INFO) << "Will be pooling with " << POOLTYPE;
  }

  Net<float> caffe_test_net(NETWORK_PATH.string(), caffe::TEST);
  caffe_test_net.CopyTrainedLayersFrom(MODEL_PATH.string());
  int BATCH_SIZE = caffe_test_net.blob_by_name("data")->num();

  // Get list of images in directory
  vector<fs::path> imgs;
  readList<fs::path>(IMGSLST, imgs);
  
  std::shared_ptr<DiskVectorLMDB<vector<float>>> dv;
  if (OUTTYPE == OUTTYPE_LMDB) {
    dv = std::shared_ptr<DiskVectorLMDB<vector<float>>>(
        new DiskVectorLMDB<vector<float>>(OUTDIR));
  }
  high_resolution_clock::time_point begin = high_resolution_clock::now();
  for (long long imgid = START_IMGID; imgid <= START_IMGID + imgs.size(); imgid++) {
    high_resolution_clock::time_point start = high_resolution_clock::now();
    fs::path imgpath = imgs[imgid - 1];

    if (imgid % PRINT_INTERVAL == 0) {
      cout << "Doing for " << imgpath << " (" << imgid << "/"
           << imgs.size() << ")...";
    }

    vector<Mat> Is;
    Mat I = imread((IMGSDIR / imgpath).string());
    Mat S; // get the segmentation image as well, used in debugging
    if (!I.data) {
      LOG(ERROR) << "Unable to read " << imgpath;
      continue;
    }
    vector<Rect> bboxes;
    if (WINDIR.string().size() > 0) {
      readBBoxesSelSearch<float>((WINDIR / (to_string((long long)imgid) + ".txt")).string(), bboxes);
    } else if (vm["sliding"].as<bool>()) {
      genSlidingWindows(I.size(), bboxes);
    } else {
      bboxes.push_back(Rect(0, 0, I.cols, I.rows)); // full image
    }

    // check if segdir defined. If so, then prune the list of bboxes
    if (SEGDIR.string().length() > 0) {
      fs::path segpath = SEGDIR / imgpath;
      if (!fs::exists(segpath)) {
        LOG(ERROR) << "Segmentation information not found for " << segpath;
      } else {
        pruneBboxesWithSeg(I.size(), segpath, bboxes, S);
      }
    }

    LOG(INFO) << "Computing over " << bboxes.size() << " subwindows";
    // push in all subwindows
    for (int i = 0; i < bboxes.size(); i++) {
      Mat Itemp  = I(bboxes[i]);
      resize(Itemp, Itemp, Size(256, 256)); // rest of the transformation will run
                                            // like mean subtraction etc
      Is.push_back(Itemp);
    }
    // Uncomment to see the windows selected
    // DEBUG_storeWindows(Is, fs::path("temp/") / imgpath, I, S);

    // [layer[image[feature]]]
    vector<vector<vector<float>>> output;
    /**
     * Separately computing features for either case of text/lmdb because 
     * can using locking (and run parallel) for text output
     */
    if (OUTTYPE == OUTTYPE_TEXT) {
      fs::path outFile = fs::change_extension(OUTDIR / imgpath, ".txt");
      if (output.size() > 1) {
        outFile = fs::change_extension(OUTDIR / fs::path(layers[0]) / imgpath, ".txt");
      }
      if (!lock(outFile)) {
        continue;
      }
      computeFeaturesPipeline(caffe_test_net, Is, layers, 
          BATCH_SIZE, output, false, POOLTYPE, NORMALIZE);
      for (int l = 0; l < output.size(); l++) {
        fs::path thisoutFile = fs::change_extension(OUTDIR / imgpath, ".txt");
        if (output.size() > 1) {
          thisoutFile = fs::change_extension(OUTDIR / fs::path(layers[l]) / imgpath, ".txt");
        }
        fs::create_directories(thisoutFile.parent_path());
        FILE* fout = fopen(thisoutFile.string().c_str(), "w");
        for (int i = 0; i < output[l].size(); i++) {
          dumpFeature(fout, output[l][i]);
        }
        fclose(fout);
      }
      unlock(outFile);
    } else if (OUTTYPE == OUTTYPE_LMDB) {
      LOG(FATAL) << "Multiple layer output is not suppported with lmdb output.";
      computeFeaturesPipeline(caffe_test_net, Is, layers, 
          BATCH_SIZE, output, false, POOLTYPE, NORMALIZE);
      // output into a DiskVector
      for (int i = 0; i < output[0].size(); i++) {
        dv->Put(hashCompleteName(imgid, i), output[0][i]);
      }
    }
    if (imgid % PRINT_INTERVAL == 0) {
      high_resolution_clock::time_point end = high_resolution_clock::now();
      cout << "Done in " << duration_cast<milliseconds>(end - start).count()
           << "ms" << endl
           << "Average taking " 
           << duration_cast<milliseconds>(end - begin).count() * 1.0f / 
              (imgid - START_IMGID + 1) << "ms" << endl;
    }
  }

  return 0;
}

inline void dumpFeature(FILE* fout, const vector<float>& feat) {
  for (int i = 0; i < feat.size(); i++) {
    if (feat[i] == 0) {
      fprintf(fout, "0 ");
    } else {
      fprintf(fout, "%f ", feat[i]);
    }
  }
  fprintf(fout, "\n");
}

inline long long hashCompleteName(long long imgid, int id) {
  return (imgid - 1) * MAXFEATPERIMG + id;
}

template<typename Dtype>
void computeFeaturesPipeline(Net<Dtype>& caffe_test_net,
    const vector<Mat>& Is,
    const vector<string>& layers,
    int BATCH_SIZE,
    vector<vector<vector<Dtype>>>& output,
    bool verbose,
    const string& POOLTYPE,
    bool NORMALIZE) {
  computeFeatures(caffe_test_net, Is, layers, BATCH_SIZE, output, verbose);
  if (! POOLTYPE.empty()) {
    // assuming all layers need to be pooled
    for (int l = 0; l < output.size(); l++) {
      poolFeatures(output[l], POOLTYPE);
    }
  }
  if (NORMALIZE) {
    // assuming all layers need to be normalized
    for (int i = 0; i < output.size(); i++) {
      l2NormalizeFeatures(output[i]);
    }
  }
}

