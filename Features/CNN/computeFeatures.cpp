/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

void genImgsList(const fs::path&, vector<fs::path>&);
template<typename Dtype>
void computeFeatures(Net<Dtype>&, const vector<Mat>&, string, int, vector<vector<Dtype>>&);
void dumpFeature(fs::path, const vector<float>&);

int
main(int argc, char *argv[]) {
    #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
    LOG(INFO) << "Extracting Features in CPU mode";
    #else
    Caffe::set_mode(Caffe::GPU);
    #endif
    Caffe::set_phase(Caffe::TEST); // important, else will give random features
    
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
    string LAYER = vm["layer"].as<string>();
    fs::path OUTDIR = fs::path(vm["outdir"].as<string>());
    fs::path IMGSDIR = fs::path(vm["imgsdir"].as<string>());
 
    NetParameter test_net_params;
    ReadProtoFromTextFile(NETWORK_PATH.string(), &test_net_params);
    Net<float> caffe_test_net(test_net_params);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(MODEL_PATH.string(), &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
    int BATCH_SIZE = caffe_test_net.blob_by_name("data")->num();

    // Get list of images in directory
    vector<fs::path> imgs;
    genImgsList(IMGSDIR, imgs);

    // Create output directory
    fs::path FEAT_OUTDIR = OUTDIR / fs::path(LAYER.c_str());
    fs::create_directories(FEAT_OUTDIR);

    vector<Mat> Is;
    for (auto imgpath : imgs) {
        Mat I = imread((IMGSDIR / imgpath).string());
        if (!I.data) {
            LOG(ERROR) << "Unable to read image " << imgpath;
            return -1;
        }
        resize(I, I, Size(256, 256));
        Is.push_back(I);
    }
    LOG(INFO) << "read images";
    vector<vector<float>> output;
    computeFeatures<float>(caffe_test_net, Is, LAYER, BATCH_SIZE, output);

    // Dump output
    for (int i = 0; i < imgs.size(); i++) {
        fs::path imgpath = imgs[i];
        fs::path outFile = fs::change_extension(FEAT_OUTDIR / imgpath, ".txt");
        fs::create_directories(outFile.parent_path());
        dumpFeature(outFile, output[i]);
    }

    return 0;
}

template<typename Dtype>
void computeFeatures(Net<Dtype>& caffe_test_net,
        const vector<Mat>& imgs,
        string LAYER,
        int BATCH_SIZE,
        vector<vector<Dtype>>& output) {
    int nImgs = imgs.size();
    int nBatches = ceil(nImgs * 1.0f / BATCH_SIZE);
    for (int batch = 0; batch < nBatches; batch++) {
        int actBatchSize = min(nImgs - batch * BATCH_SIZE, BATCH_SIZE);
        vector<Mat> imgs_b;
        if (actBatchSize >= BATCH_SIZE) {
            imgs_b.insert(imgs_b.end(), imgs.begin() + batch * BATCH_SIZE, 
                    imgs.begin() + (batch + 1) * BATCH_SIZE);
        } else {
            imgs_b.insert(imgs_b.end(), imgs.begin() + batch * BATCH_SIZE, imgs.end());
            for (int j = actBatchSize; j < BATCH_SIZE; j++)
                imgs_b.push_back(imgs[0]);
        }
        vector<int> dvl(BATCH_SIZE, 0);
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype>>(
                caffe_test_net.layers()[0])->AddMatVector(imgs_b, dvl);
        vector<Blob<Dtype>*> dummy_bottom_vec;
        Dtype loss = 0.0f;
        caffe_test_net.ForwardPrefilled(&loss);
        const boost::shared_ptr<Blob<Dtype>> feat = caffe_test_net.blob_by_name(LAYER);
        for (int i = 0; i < actBatchSize; i++) {
            Dtype* feat_data = feat->mutable_cpu_data() + feat->offset(i);
            output.push_back(vector<Dtype>(feat_data, feat_data + feat->count() / feat->num()));
        }
        LOG(INFO) << "Batch " << batch << " (" << actBatchSize << " images) done";
    }
}

/**
  * Function to return list of images in a directory (searched recursively).
  * The output paths are w.r.t. the path imgsDir
  */
void genImgsList(const fs::path& imgsDir, vector<fs::path>& list) {
    if(!fs::exists(imgsDir) || !fs::is_directory(imgsDir)) return;
    vector<string> imgsExts = {".jpg", ".png", ".jpeg", ".JPEG", ".PNG", ".JPG"};

    fs::recursive_directory_iterator it(imgsDir);
    fs::recursive_directory_iterator endit;
    while(it != endit) {
        if(fs::is_regular_file(*it) && 
                find(imgsExts.begin(), imgsExts.end(), 
                    it->path().extension()) != imgsExts.end())
            // write out paths but clip out the initial relative path from current dir 
            list.push_back(fs::path(it->path().relative_path().string().
                    substr(imgsDir.relative_path().string().length())));
        ++it;
    }
    LOG(INFO) << "Found " << list.size() << " image file(s) in " << imgsDir;
}

void dumpFeature(fs::path outpath, const vector<float>& feat) {
    FILE* fout = fopen(outpath.string().c_str(), "w");
    for (auto feati : feat) {
        fprintf(fout, "%f\n", feati);
    }
    fclose(fout);
}

