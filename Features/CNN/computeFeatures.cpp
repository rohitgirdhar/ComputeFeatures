/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
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
    string NETWORK_PATH = 
        "/exports/cyclops/software/vision/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
    string MODEL_PATH = 
        "/exports/cyclops/software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";

    NetParameter test_net_params;
    ReadProtoFromBinaryFile(NETWORK_PATH, &test_net_params);
    Net<float> caffe_test_net(test_net_params);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(MODEL_PATH, &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

    vector<Blob<float>*> blob_input_vec;
    I = imread("/exports/cyclops/work/005_BgMatches/dataset/PeopleAtLandmarks/corpus/AbuSimbel/1.jpg");
    
    return 0;
}

