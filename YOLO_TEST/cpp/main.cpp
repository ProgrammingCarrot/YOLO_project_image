#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include <cmath>

std::vector<std::string> classes = {"Human"};
bool debug = false;

struct input_factor{
	const int input_w = 640; // also is model input rows and cols
	const int input_h = 480;
	const float x_factor = 640.0/640.0f;
	const float y_factor = 480.0/640.0f;
	const int y_padding = (input_w - input_h) / 2;
}screen_output;

struct nms{
	std::vector<int> class_ids;
	std::vector<float> confidence;
	std::vector<cv::Rect> boxes;
	std::vector<int> indices;
	std::vector<cv::Point> center;
};

// load Model
cv::dnn::Net model(const std::string& path){
	std::cout << "Loading Model from:" << path << std::endl;
	cv::dnn::Net net;
	try{
		net = cv::dnn::readNet(path);
		std::cout << "loading Model Success!" << std::endl;

		// set dnn run on GPU
		// net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		// net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		return net;
	}
	catch(std::exception& e){
		std::cerr << e.what() << std::endl;
		return net;
	}	
}

float get_distance(const int x,const int y,rs2::depth_frame depth,rs2::depth_sensor sensor){
	// get point
	int width = screen_output.input_w;
	int height = screen_output.input_h;
	
	// get depth and scale
	float scale = sensor.get_depth_scale();
	uint16_t* data = (uint16_t*)depth.get_data();
	uint16_t pixel = data[width*y + x];

	if(debug){ // mode:1 = debug mode
		std::cout << "scale : " << scale << std::endl;
		std::cout << "pixel : " << pixel << std::endl;
	}

	// calculate distance
	float distance = pixel*scale;
	return distance;
}


// debug
cv::Mat convert_blob(cv::Mat blob){

	std::vector<cv::Mat> images_from_blob;
	cv::Mat image;
	try{
		cv::dnn::imagesFromBlob(blob,images_from_blob);
		image = images_from_blob[0];
		return image;
	}
	catch(std::exception& e){
		std::cerr << e.what() << std::endl;
		return image;
	}
}

// resize image size to match model input
cv::UMat image_resize(const cv::UMat& image,const cv::Size& target,const cv::Scalar& color= cv::Scalar(114,114,114)){
	int top = (target.height - screen_output.input_h) / 2;
	int bottom = target.height - screen_output.input_h - top;
	// because x = 640, so no need to fill right and left
	cv::UMat padding_image;
	cv::copyMakeBorder(image,padding_image,top,bottom,0,0,cv::BORDER_CONSTANT,color);
	
	return padding_image;
}

int main(int argc,char *argv[]) try{
	cv::ocl::setUseOpenCL(true);
	// Detect if in debug mode
	if(std::string(argv[1]) == "debug"){
		std::cout << "THIS IS DEBUG MODE!" << std::endl;
		debug = true;
	}
	// loading Camera
	rs2::pipeline pipe;
	rs2::config cfg;

	// loading Model
	std::string path = "best.onnx";
	cv::dnn::Net yolo_net = model(path);
	if(yolo_net.empty()){
		std::cout << "Cannot load Model! Please try again" << std::endl;
		return 1;
	}

	// Set Color and Depth Stream
	cfg.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
	cfg.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);

	rs2::pipeline_profile profile = pipe.start(cfg);
	rs2::align align_color(RS2_STREAM_COLOR);
	rs2::device device = profile.get_device();
	rs2::depth_sensor d_sensor = device.query_sensors().front().as<rs2::depth_sensor>();
	
	const auto name = "Display";
	cv::Scalar font_color = (0,0,0);
	cv::namedWindow(name,cv::WINDOW_AUTOSIZE);
	cv::Mat blob;
	const cv::Size target_size(640,640);
	while(cv::waitKey(1)!=27 && cv::getWindowProperty(name,cv::WND_PROP_AUTOSIZE) >= 0){
		rs2::frameset frame = pipe.wait_for_frames();
		frame = align_color.process(frame);

		rs2::depth_frame depth = frame.get_depth_frame();
		rs2::frame color = frame.get_color_frame();

		const int w = color.as<rs2::video_frame>().get_width();
		const int h = color.as<rs2::video_frame>().get_height();

		if(!depth || !color){
			continue;
		}
		
		cv::Mat cpu_image(cv::Size(w,h),CV_8UC3,(void*)color.get_data(),cv::UMat::AUTO_STEP);
		cv::UMat image = cpu_image.getUMat(cv::ACCESS_READ);
		cv::UMat input_image = image_resize(image,target_size);
		
		// Model input Handle
                blob = cv::dnn::blobFromImage(input_image,1/255.0,target_size,cv::Scalar(0,0,0),true,false);
		yolo_net.setInput(blob);
		cv::Mat preds = yolo_net.forward();

		// output result
		cv::Mat det_output(preds.size[1],preds.size[2],CV_32F,preds.ptr<float>());
		struct nms NMS;
		for(int i = 0;i < det_output.cols;i++){
			
			double final_confidence = det_output.at<float>(4,i);
			if(final_confidence < 0.6f){
				continue;
			}
			// draw detect boxes
			float cx = det_output.at<float>(0,i);
			float cy = det_output.at<float>(1,i);
			float ow = det_output.at<float>(2,i);
			float oh = det_output.at<float>(3,i);

			int x = static_cast<int>(cx-0.5*ow);
			int y = static_cast<int>(cy-0.5*oh - screen_output.y_padding);	
			int width = static_cast<int>(ow);
			int height = static_cast<int>(oh);

			// don't let width and height bigger then screen
			width = std::min(width, w);
			height = std::min(height, h);

			NMS.boxes.push_back(cv::Rect(x,y,width,height));
			NMS.confidence.push_back(final_confidence);
			NMS.center.push_back(cv::Point(cx,cy));
		}
		
		// set NMS boxes
		cv::dnn::NMSBoxes(NMS.boxes,NMS.confidence,0.45f,0.7f,NMS.indices);
		for(int i = 0; i < NMS.indices.size(); ++i){
			int index = NMS.indices[i];
			int y_offset = 0;
			cv::Rect NMSbox = NMS.boxes[index];
			if(debug) std::cout << "box origin:" << "(" << NMS.boxes[i].x << "," << NMS.boxes[i].y << ")" << std::endl;
			if(NMSbox.y < 50){
				y_offset = -30;
				NMSbox.y = 0;	
			}
			else{
				y_offset = 0;
			}
			if(debug) std::cout << "center:" << "(" << NMS.center[i].x << "," << NMS.center[i].y << ")" << std::endl;
			// get distance
			float instance_distance = get_distance(NMS.center[i].x,NMS.center[i].y,depth,d_sensor);
			std::stringstream label_stream;
			label_stream << classes[0] << std::setprecision(2) << instance_distance << "meters" << NMS.confidence[i] << std::setprecision(2);
			std::string label = label_stream.str();

			cv::rectangle(image,cv::Point(NMSbox.x,NMSbox.y),cv::Point(NMSbox.x+NMSbox.width, NMSbox.y+NMSbox.height),cv::Scalar(0,0,255),2,8);
			cv::putText(image,label.c_str(), cv::Point(NMSbox.x,NMSbox.y-y_offset),cv::FONT_HERSHEY_SIMPLEX,0.7,font_color,3);
		}

		imshow(name,image);
	}
	cv::destroyAllWindows();
	return EXIT_SUCCESS;
}
catch(const rs2::error& e){
	std::cerr << "Realsense error:" << e.get_failed_function() << "(" << e.get_failed_args() \
		  << ")" << "\n" << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch(const std::exception& e){
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
