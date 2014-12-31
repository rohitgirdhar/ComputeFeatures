# Set the following:
CAFFE_ROOT := /home/rgirdhar/work/03_temp/caffe_dev
CUDA_ROOT := /usr/local/cuda
CPU_ONLY := 0

CXX += -std=c++0x
LIBS += -L$(CAFFE_ROOT)/build/lib `pkg-config --libs opencv`
LDFLAGS += -lcaffe -lglog -lboost_program_options
CPPFLAGS += `pkg-config --cflags opencv`
INC += -I$(CAFFE_ROOT)/include/ -I$(CAFFE_ROOT)/build/src/
ifneq ($(CPU_ONLY),1)
    LIBS += -L$(CUDA_ROOT)/lib64
    INC += -I$(CUDA_ROOT)/include
    LDFLAGS += -lcudart -lcublas -lcurand
else
	CPPFLAGS += -DCPU_ONLY
endif

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile
	$(CXX) -std=c++0x \
        $(INC) \
		$(CPPFLAGS) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2
clean:
	rm -rf *.bin
