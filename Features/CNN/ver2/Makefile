# Set the following:
#CAFFE_INSTALL_DIR = /IUS/homes4/rohytg/software/caffe_install
CAFFE_ROOT := ../external/caffe
CUDA_ROOT := /usr/local/cuda
#INC += -I$(CAFFE_INSTALL_DIR)/gflags_install/include -I$(CAFFE_INSTALL_DIR)/glog_install/include -I$(CAFFE_INSTALL_DIR)/mdb_install/include -I ~/software/boost/include
#LIBS += -L$(CAFFE_INSTALL_DIR)/gflags_install/lib -L$(CAFFE_INSTALL_DIR)/glog_install/lib -L$(CAFFE_INSTALL_DIR)/mdb_install/lib -L ~/software/boost/libs
CPU_ONLY := 0

CXXFLAGS += -std=c++0x -fopenmp -O2 -g
LIBS += -L$(CAFFE_ROOT)/build/lib -L/usr/local/lib/ -L/srv2/rgirdhar/Software/cpp/zeromq/install/lib/ -L/srv2/rgirdhar/Software/vision/opencv3_install/lib/
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -llmdb `pkg-config --libs /srv2/rgirdhar/Software/vision/opencv3_install/lib/pkgconfig/opencv.pc` -lhdf5 -lzmq
INC += -I$(CAFFE_ROOT)/include/ -I$(CAFFE_ROOT)/build/src/ -I/srv2/rgirdhar/Software/vision/opencv3_install/include/ -I../ -I/srv2/rgirdhar/Software/cpp/zeromq/cppzmq/ -I /srv2/rgirdhar/Software/cpp/zeromq/install/include/
ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CXXFLAGS += -DCPU_ONLY
endif
CXXFLAGS += -DUSE_OPENCV

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile utils.hpp ../external/DiskVector/DiskVectorLMDB.hpp Config.hpp
	$(CXX)  \
		$(CXXFLAGS) \
    $(INC) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) 

clean:
	rm -rf *.bin
