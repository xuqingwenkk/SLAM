#add the dependency of opencv
FIND_PACKAGE(OpenCV 2.4 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_library(calibbase calib.cpp)
target_link_libraries(calibbase ${OpenCV_LIBS})

add_executable(Fisheye_Calib FisheyeCalib.cpp)
target_link_libraries(Fisheye_Calib calibbase ${OpenCV_LIBS})
