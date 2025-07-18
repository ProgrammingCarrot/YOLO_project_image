#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include <cmath>

std::vector<std::string> classes = {"Human"};

struct input_factor{
	const int input_w = 640;
	const int input_h = 480;
	const float x_factor = 640/640.0f;
	const float y_factor = 480/640.0f;
}model_input;

struct nms{
	std::vector<int> class_ids;
	std::vector<float> confidence;
	std::vector<cv::Rect> boxes;
	std::vector<int> indices;
};

float sigmoid(double logit){
	float score = 1.0f/(1.0f + exp(-logit));
	return score;
}

// load Model
cv::dnn::Net model(const std::string& path){
	std::cout << "Loading Model from:" << path << std::endl;
	cv::dnn::Net net;
	try{
		net = cv::dnn::readNetFromONNX(path);
		std::cout << "loading Model Success!" << std::endl;
		return net;
	}
	catch(std::exception& e){
		std::cerr << e.what() << std::endl;
		return net;
	}	
}

int main() try{
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
	float scale = d_sensor.get_depth_scale();
	
	const auto name = "Display";
	cv::namedWindow(name,cv::WINDOW_AUTOSIZE);
	while(cv::waitKey(1)!=27 && cv::getWindowProperty(name,cv::WND_PROP_AUTOSIZE) >= 0){
		rs2::frameset frame = pipe.wait_for_frames();
		frame = align_color.process(frame);

		rs2::depth_frame depth = frame.get_depth_frame();
		rs2::frame color = frame.get_color_frame();

		const int w = color.as<rs2::video_frame>().get_width();
		const int h = color.as<rs2::video_frame>().get_height();

		// get distance test
		int x = w/2;
		int y = h/2;
		
		uint16_t* data = (uint16_t*)depth.get_data();
		uint16_t pixel = data[(y*w)+x];
		float distance = pixel*scale;

		std::stringstream info_stream;
		info_stream << "(" << x << "," << y << ") distance : " << distance;
		std::string distance_info = info_stream.str();
		cv::Point position(30,450);
		cv::Scalar font_color(0,0,0);

		if(!depth || !color){
			continue;
		}
		
		cv::Mat image(cv::Size(w,h),CV_8UC3,(void*)color.get_data(),cv::Mat::AUTO_STEP);

		cv::Mat blob;
		// Model input Handle
                blob = cv::dnn::blobFromImage(image,1/255.0,cv::Size(640,640),cv::Scalar(0,0,0),true,false);
		yolo_net.setInput(blob);
		cv::Mat preds = yolo_net.forward();

		// output result
		cv::Mat det_output(preds.size[1],preds.size[2],CV_32F,preds.ptr<float>());
		struct nms NMS;
		for(int i = 0;i < det_output.rows;i++){
			float confidence = det_output.at<float>(i,4);
			cv::Mat classes_score = det_output.row(i).colRange(5,5+classes.size());
			
			// record Max value Position
			cv::Point class_id_point;
			double max_classes_score;
			float max_classes_score_sigmoid;
			minMaxLoc(classes_score,NULL,&max_classes_score,NULL,&class_id_point);
			int class_id = class_id_point.x;

			confidence = sigmoid(confidence);
			max_classes_score_sigmoid = sigmoid(confidence);
			float final_confidence = max_classes_score_sigmoid * confidence;
			if(final_confidence < 0.5) continue;
			// draw detect boxes
			float cx = det_output.at<float>(i,0);
			float cy = det_output.at<float>(i,1);
			float ow = det_output.at<float>(i,2);
			float oh = det_output.at<float>(i,3);

			int x = static_cast<int>((cx-ow/2)*model_input.x_factor);
			int y = static_cast<int>((cy-oh/2)*model_input.y_factor);
			int width = static_cast<int>(ow*model_input.x_factor);
			int height = static_cast<int>(oh*model_input.y_factor);

			NMS.boxes.push_back(cv::Rect(x,y,width,height));
			NMS.confidence.push_back(final_confidence);
			NMS.class_ids.push_back(class_id);
		}
		// set NMS boxes
		cv::dnn::NMSBoxes(NMS.boxes,NMS.confidence,0.5,0.5,NMS.indices);
		for(int i = 0; i < NMS.indices.size(); ++i){
			int index = NMS.indices[i];
			std::string label = classes[NMS.class_ids[index]] + ":" + std::to_string(NMS.confidence[index]);
			cv::Rect NMSbox = NMS.boxes[index];
			cv::rectangle(image,cv::Point(NMSbox.x,NMSbox.y),cv::Point(NMSbox.x+NMSbox.width, NMSbox.y+NMSbox.height),cv::Scalar(0,0,255),2,8);
			cv::putText(image,label.c_str(), cv::Point(NMSbox.x,NMSbox.y-10),cv::FONT_HERSHEY_SIMPLEX,0.7,font_color,3);
		}
		cv::circle(image,cv::Point(x,y),1,cv::Scalar(0,255,255),-1);
		cv::putText(image,distance_info,position,cv::FONT_HERSHEY_SIMPLEX,0.7, font_color,3);
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
