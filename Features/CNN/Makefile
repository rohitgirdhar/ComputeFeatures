# Set the following:
CAFFE_INSTALL_DIR = /IUS/homes4/rohytg/software/caffe_install
CAFFE_ROOT := external/caffe_dev_MemLayerWithMat
CUDA_ROOT := /usr/local/cuda
INC += -I$(CAFFE_INSTALL_DIR)/gflags_install/include -I$(CAFFE_INSTALL_DIR)/glog_install/include -I$(CAFFE_INSTALL_DIR)/mdb_install/include
LIBS += -L$(CAFFE_INSTALL_DIR)/gflags_install/lib -L$(CAFFE_INSTALL_DIR)/glog_install/lib -L$(CAFFE_INSTALL_DIR)/mdb_install/lib
CPU_ONLY := 1

CXX += -std=c++0x
LIBS += -L$(CAFFE_ROOT)/build/lib `pkg-config --libs opencv`
LDFLAGS += -lcaffe -lglog -lboost_program_options -lboost_filesystem -lboost_system
CPPFLAGS += `pkg-config --cflags opencv` -g
INC += -I$(CAFFE_ROOT)/include/ -I$(CAFFE_ROOT)/build/src/
ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CPPFLAGS += -DCPU_ONLY
endif

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile utils.hpp
	$(CXX) -std=c++0x \
        $(INC) \
		$(CPPFLAGS) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2
clean:
	rm -rf *.bin
