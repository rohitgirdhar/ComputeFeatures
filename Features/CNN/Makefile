# Set the following:
CAFFE_ROOT:=/exports/cyclops/software/vision/caffe/
CPU_ONLY := 1

CXX += -std=c++0x
LIBS += -L$(CAFFE_ROOT)/build/lib `pkg-config --libs opencv`
LDFLAGS += -lcaffe -lglog -lboost_program_options
CPPFLAGS += `pkg-config --cflags opencv`
ifeq ($(CPU_ONLY),1)
	CPPFLAGS+=-DCPU_ONLY
endif

all: $(patsubst %.cpp, %.bin, $(wildcard *.cpp))

%.bin: %.cpp Makefile
	$(CXX) -std=c++0x \
		-I$(CAFFE_ROOT)/include/ \
		-I$(CAFFE_ROOT)/build/src/ \
		$(CPPFLAGS) \
		$< -o $@ \
		$(LIBS) \
		$(LDFLAGS) \
		-O2
clean:
	rm -rf *.bin
