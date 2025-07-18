cmake_minimum_required(VERSION 3.10)

project(HelloWorld CXX)

find_package(realsense2 REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(test main.cpp)
set_property(TARGET test PROPERTY CXX_STANDARD 11)
target_link_libraries(test PRIVATE 
		      realsense2::realsense2
		      ${OpenCV_LIBS}
		      )

set_target_properties(test PROPERTIES
		      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
