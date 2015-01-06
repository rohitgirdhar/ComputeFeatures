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
void computeFeatures(Net<Dtype>&, const vector<Mat>&, string, vector<vector<Dtype>>&);
void dumpFeature(fs::path, const vector<float>&);

int
main(int argc, char *argv[]) {
    #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
    LOG(INFO) << "Extracting Features in CPU mode";
    #else
    Caffe::set_mode(Caffe::GPU);
    #endif
    
    po::options_description desc("Allowed options:");
    desc.add_options()
        ("help", "Show help")
    ;
    // TODO: Add the arguments
    fs::path NETWORK_PATH = fs::path("deploy.prototxt");
    fs::path MODEL_PATH = 
        fs::path("/home/rgirdhar/work/03_temp/caffe_dev/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel");
    string LAYER = "pool5";
    fs::path OUTDIR = fs::path("output");
    fs::path IMGSDIR = fs::path(".");
 
    NetParameter test_net_params;
    ReadProtoFromTextFile(NETWORK_PATH.string(), &test_net_params);
    Net<float> caffe_test_net(test_net_params);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(MODEL_PATH.string(), &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

    // Get list of images in directory
    vector<fs::path> imgs;
    genImgsList(IMGSDIR, imgs);

    // Create output directory
    fs::path FEAT_OUTDIR = OUTDIR / fs::path(LAYER.c_str());
    fs::create_directories(FEAT_OUTDIR);

    vector<Mat> Is;
    for (auto imgpath : imgs) {
        Mat I = imread((IMGSDIR / imgpath).string());
        resize(I, I, Size(256, 256));
        Is.push_back(I);
    }
    vector<vector<float>> output;
    computeFeatures<float>(caffe_test_net, Is, LAYER, output);

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
        vector<vector<Dtype>>& output) {
    vector<int> dvl(imgs.size(), 0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype>>(
            caffe_test_net.layers()[0])->AddMatVector(imgs, dvl);
    vector<Blob<Dtype>*> dummy_bottom_vec;
    Dtype loss = 0.0f;
    caffe_test_net.Forward(dummy_bottom_vec, &loss);
    const boost::shared_ptr<Blob<Dtype>> feat = caffe_test_net.blob_by_name(LAYER);
    output.push_back(vector<Dtype>(feat->cpu_data(), feat->cpu_data() + feat->count()));
}

void genImgsList(const fs::path& imgsDir, vector<fs::path>& list) {
    if(!fs::exists(imgsDir) || !fs::is_directory(imgsDir)) return;
    vector<string> imgsExts = {".jpg", ".png", ".jpeg", ".JPEG", ".PNG", ".JPG"};

    fs::recursive_directory_iterator it(imgsDir);
    fs::recursive_directory_iterator endit;
    while(it != endit) {
        if(fs::is_regular_file(*it) && 
                find(imgsExts.begin(), imgsExts.end(), 
                    it->path().extension()) != imgsExts.end())
            list.push_back(it->path().relative_path());
        ++it;
    }
    LOG(INFO) << "Found " << list.size() << " image file(s) in " << imgsDir;
}

void dumpFeature(fs::path outpath, const vector<float>& feat) {
    ofstream fout(outpath.string().c_str(), ios::out);
    for (auto feati : feat) {
        fout << feati << endl;
    }
    fout.close();
}

