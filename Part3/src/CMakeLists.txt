find_package(PCL REQUIRED COMPONENTS common io visualization filters)

add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})

#add the dependency of opencv
FIND_PACKAGE(OpenCV 2.4 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_library(SFMbase sfm.cpp)
target_link_libraries(SFMbase ${OpenCV_LIBS} ${PCL_LIBRARIES})

#add_executable(Fisheye_Calib FisheyeCalib.cpp)
#target_link_libraries(Fisheye_Calib SFMbase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(Feature feature.cpp)
target_link_libraries(Feature SFMbase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(FeaMatch match.cpp)
target_link_libraries(FeaMatch SFMbase ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(VisualOdometry VO.cpp)
target_link_libraries(VisualOdometry SFMbase ${OpenCV_LIBS} ${PCL_LIBRARIES})
