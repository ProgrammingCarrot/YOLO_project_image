#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <string.h>

int main() try{
	rs2::colorizer color_map;
	rs2::pipeline pipe;

	pipe.start();
	const auto name = "Display";
	cv::namedWindow(name,cv::WINDOW_AUTOSIZE);

	while(cv::waitKey(1)<0 && cv::getWindowProperty(name,cv::WND_PROP_AUTOSIZE) >= 0){
		rs2::frameset frame = pipe.wait_for_frames();

		rs2::frame depth = frame.get_depth_frame();
		rs2::frame color = frame.get_color_frame();
		depth = depth.apply_filter(color_map);

		const int w = depth.as<rs2::video_frame>().get_width();
		const int h = depth.as<rs2::video_frame>().get_height();

		cv::Mat image(cv::Size(w,h),CV_8UC3,(void*)depth.get_data(), \
			      cv::Mat::AUTO_STEP);
		imshow(name,image);
	}

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
