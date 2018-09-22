# OpticalFlow-SpeedTest
This is a script that test the speed of multiple optical flow extraction methods, supporting both single-thread and multi-thread

Optical flows plays a significant role in video understanding. This script helps you test multiple methods, including c++ api: [DenseFlow](https://github.com/wanglimin/dense_flow), and python api:[pydenseflow](https://github.com/qijiezhao/py-denseflow)，cpu version(farnback algorithm), gpu version(tvl1 algorithm), and single-thread, multi-thread。 Optionally, you can add other methods in code to jointly test.

### Requirements

- numpy
- opencv 2+ (cv2)
- PIL
- multiprocess
- scikit-video
- scipy

### DenseFlow Installation
	
	cd dense_flow
	[[ -d build ]] || mkdir build
	cd build
	OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
	make -j


### PyDenseFlow Installation

	pip install -r requirements.txt


### Usage

1. Install DenseFlow.
2. open **main.py**, re-define the get_videos() function to parse test videos that you are going to test.
2. set Denseflow path and output path.
3. set the parameter of Pool, 8 is the default number. actually, it should be smaller than the number of your machine's cores.
3. run the code: 
	
		python main.py

### My result

1. As is well known, TVL1 algorithm is proved to be better when represents short temporal information compared with Farnback , But TVL1 always requires GPU, so this tradeoff should be considered.
2. If I run more than one processes on a single GPU to extract TVL1 opticalflows, the efficiency will not improve. Multi-process is only useful when extractor are implemented on multi-GPUs, each GPU runs a process to extract opticalflows.
3. Farnback on CPU is little slow than TVL1 on GPU(TITAN X), but 8 threads accelerate the speed with more than 6x. As a comparison, 8 GPUs is not easy toget.
4. C++ api runs a little faster than python api. This is not strange. But python api is convenient to install and use.
5. If you have enough computation resource, C++ api + tvl1 is suggested. Otherwise, with limited resource, you should process a large amount of data, C++ api + farnback + multi-process is suggested.  If you are going to finish a little demo, or some toy code, python api + farnback + multi-process is suggested.

 
