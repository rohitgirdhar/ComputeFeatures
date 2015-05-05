all: *.cpp
	g++ -O3 -Wall -I/srv2/rgirdhar/Software/vision/opencv_install/include/ -o hesaff pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp hesaff.cpp -L /srv2/rgirdhar/Software/vision/opencv_install/lib/ `pkg-config --libs /srv2/rgirdhar/Software/vision/opencv_install/lib/pkgconfig/opencv.pc` -lrt 
