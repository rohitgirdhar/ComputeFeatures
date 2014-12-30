/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
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
    string NETWORK_PATH = "deploy2.prototxt";
    string MODEL_PATH = 
        "/exports/cyclops/software/vision/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";

    NetParameter test_net_params;
    ReadProtoFromTextFile(NETWORK_PATH, &test_net_params);
    Net<float> caffe_test_net(test_net_params);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(MODEL_PATH, &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

    LOG(INFO) << "here" << endl;
    vector<Blob<float>*> blob_input_vec;
    string fname = "/exports/cyclops/work/005_BgMatches/dataset/PeopleAtLandmarks/corpus/AbuSimbel/1.jpg";
    Mat I = imread(fname);
    vector<Datum> datum_vector;
    Datum datum;
    ReadImageToDatum(fname, 0, I.rows, I.cols, 1, &datum);
    datum_vector.push_back(datum);

    const boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer =
        boost::static_pointer_cast<MemoryDataLayer<float> >(
        caffe_test_net.layer_by_name("data2"));
    return 0;
}

