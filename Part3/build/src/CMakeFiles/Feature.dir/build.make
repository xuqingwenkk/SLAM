# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robotqw/robotqw/SLAM/Project/Part3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robotqw/robotqw/SLAM/Project/Part3/build

# Include any dependencies generated for this target.
include src/CMakeFiles/Feature.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/Feature.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/Feature.dir/flags.make

src/CMakeFiles/Feature.dir/feature.cpp.o: src/CMakeFiles/Feature.dir/flags.make
src/CMakeFiles/Feature.dir/feature.cpp.o: ../src/feature.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/robotqw/robotqw/SLAM/Project/Part3/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/Feature.dir/feature.cpp.o"
	cd /home/robotqw/robotqw/SLAM/Project/Part3/build/src && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Feature.dir/feature.cpp.o -c /home/robotqw/robotqw/SLAM/Project/Part3/src/feature.cpp

src/CMakeFiles/Feature.dir/feature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Feature.dir/feature.cpp.i"
	cd /home/robotqw/robotqw/SLAM/Project/Part3/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/robotqw/robotqw/SLAM/Project/Part3/src/feature.cpp > CMakeFiles/Feature.dir/feature.cpp.i

src/CMakeFiles/Feature.dir/feature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Feature.dir/feature.cpp.s"
	cd /home/robotqw/robotqw/SLAM/Project/Part3/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/robotqw/robotqw/SLAM/Project/Part3/src/feature.cpp -o CMakeFiles/Feature.dir/feature.cpp.s

src/CMakeFiles/Feature.dir/feature.cpp.o.requires:
.PHONY : src/CMakeFiles/Feature.dir/feature.cpp.o.requires

src/CMakeFiles/Feature.dir/feature.cpp.o.provides: src/CMakeFiles/Feature.dir/feature.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/Feature.dir/build.make src/CMakeFiles/Feature.dir/feature.cpp.o.provides.build
.PHONY : src/CMakeFiles/Feature.dir/feature.cpp.o.provides

src/CMakeFiles/Feature.dir/feature.cpp.o.provides.build: src/CMakeFiles/Feature.dir/feature.cpp.o

# Object files for target Feature
Feature_OBJECTS = \
"CMakeFiles/Feature.dir/feature.cpp.o"

# External object files for target Feature
Feature_EXTERNAL_OBJECTS =

../bin/Feature: src/CMakeFiles/Feature.dir/feature.cpp.o
../bin/Feature: src/CMakeFiles/Feature.dir/build.make
../bin/Feature: ../lib/libSFMbase.a
../bin/Feature: /usr/local/lib/libopencv_videostab.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_video.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_ts.a
../bin/Feature: /usr/local/lib/libopencv_superres.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_stitching.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_photo.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_ocl.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_objdetect.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_nonfree.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_ml.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_legacy.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_imgproc.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_highgui.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_gpu.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_flann.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_features2d.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_core.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_contrib.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_calib3d.so.2.4.10
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/Feature: /usr/lib/libpcl_common.so
../bin/Feature: /usr/lib/libpcl_octree.so
../bin/Feature: /usr/lib/libOpenNI.so
../bin/Feature: /usr/lib/libOpenNI2.so
../bin/Feature: /usr/lib/libvtkCommon.so.5.8.0
../bin/Feature: /usr/lib/libvtkFiltering.so.5.8.0
../bin/Feature: /usr/lib/libvtkImaging.so.5.8.0
../bin/Feature: /usr/lib/libvtkGraphics.so.5.8.0
../bin/Feature: /usr/lib/libvtkGenericFiltering.so.5.8.0
../bin/Feature: /usr/lib/libvtkIO.so.5.8.0
../bin/Feature: /usr/lib/libvtkRendering.so.5.8.0
../bin/Feature: /usr/lib/libvtkVolumeRendering.so.5.8.0
../bin/Feature: /usr/lib/libvtkHybrid.so.5.8.0
../bin/Feature: /usr/lib/libvtkWidgets.so.5.8.0
../bin/Feature: /usr/lib/libvtkParallel.so.5.8.0
../bin/Feature: /usr/lib/libvtkInfovis.so.5.8.0
../bin/Feature: /usr/lib/libvtkGeovis.so.5.8.0
../bin/Feature: /usr/lib/libvtkViews.so.5.8.0
../bin/Feature: /usr/lib/libvtkCharts.so.5.8.0
../bin/Feature: /usr/lib/libpcl_io.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/Feature: /usr/lib/libpcl_kdtree.so
../bin/Feature: /usr/lib/libpcl_search.so
../bin/Feature: /usr/lib/libpcl_visualization.so
../bin/Feature: /usr/lib/libpcl_sample_consensus.so
../bin/Feature: /usr/lib/libpcl_filters.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/Feature: /usr/lib/libOpenNI.so
../bin/Feature: /usr/lib/libOpenNI2.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/Feature: /usr/lib/libvtkCommon.so.5.8.0
../bin/Feature: /usr/lib/libvtkFiltering.so.5.8.0
../bin/Feature: /usr/lib/libvtkImaging.so.5.8.0
../bin/Feature: /usr/lib/libvtkGraphics.so.5.8.0
../bin/Feature: /usr/lib/libvtkGenericFiltering.so.5.8.0
../bin/Feature: /usr/lib/libvtkIO.so.5.8.0
../bin/Feature: /usr/lib/libvtkRendering.so.5.8.0
../bin/Feature: /usr/lib/libvtkVolumeRendering.so.5.8.0
../bin/Feature: /usr/lib/libvtkHybrid.so.5.8.0
../bin/Feature: /usr/lib/libvtkWidgets.so.5.8.0
../bin/Feature: /usr/lib/libvtkParallel.so.5.8.0
../bin/Feature: /usr/lib/libvtkInfovis.so.5.8.0
../bin/Feature: /usr/lib/libvtkGeovis.so.5.8.0
../bin/Feature: /usr/lib/libvtkViews.so.5.8.0
../bin/Feature: /usr/lib/libvtkCharts.so.5.8.0
../bin/Feature: /usr/local/lib/libopencv_nonfree.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_ocl.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_gpu.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_photo.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_objdetect.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_legacy.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_video.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_ml.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_calib3d.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_features2d.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_highgui.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_imgproc.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_flann.so.2.4.10
../bin/Feature: /usr/local/lib/libopencv_core.so.2.4.10
../bin/Feature: /usr/lib/libvtkViews.so.5.8.0
../bin/Feature: /usr/lib/libvtkInfovis.so.5.8.0
../bin/Feature: /usr/lib/libvtkWidgets.so.5.8.0
../bin/Feature: /usr/lib/libvtkVolumeRendering.so.5.8.0
../bin/Feature: /usr/lib/libvtkHybrid.so.5.8.0
../bin/Feature: /usr/lib/libvtkParallel.so.5.8.0
../bin/Feature: /usr/lib/libvtkRendering.so.5.8.0
../bin/Feature: /usr/lib/libvtkImaging.so.5.8.0
../bin/Feature: /usr/lib/libvtkGraphics.so.5.8.0
../bin/Feature: /usr/lib/libvtkIO.so.5.8.0
../bin/Feature: /usr/lib/libvtkFiltering.so.5.8.0
../bin/Feature: /usr/lib/libvtkCommon.so.5.8.0
../bin/Feature: /usr/lib/libvtksys.so.5.8.0
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/Feature: /usr/lib/libpcl_common.so
../bin/Feature: /usr/lib/libpcl_octree.so
../bin/Feature: /usr/lib/libOpenNI.so
../bin/Feature: /usr/lib/libOpenNI2.so
../bin/Feature: /usr/lib/libpcl_io.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/Feature: /usr/lib/libpcl_kdtree.so
../bin/Feature: /usr/lib/libpcl_search.so
../bin/Feature: /usr/lib/libpcl_visualization.so
../bin/Feature: /usr/lib/libpcl_sample_consensus.so
../bin/Feature: /usr/lib/libpcl_filters.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/Feature: /usr/lib/libpcl_common.so
../bin/Feature: /usr/lib/libpcl_octree.so
../bin/Feature: /usr/lib/libOpenNI.so
../bin/Feature: /usr/lib/libOpenNI2.so
../bin/Feature: /usr/lib/libpcl_io.so
../bin/Feature: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/Feature: /usr/lib/libpcl_kdtree.so
../bin/Feature: /usr/lib/libpcl_search.so
../bin/Feature: /usr/lib/libpcl_visualization.so
../bin/Feature: /usr/lib/libpcl_sample_consensus.so
../bin/Feature: /usr/lib/libpcl_filters.so
../bin/Feature: src/CMakeFiles/Feature.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../bin/Feature"
	cd /home/robotqw/robotqw/SLAM/Project/Part3/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Feature.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/Feature.dir/build: ../bin/Feature
.PHONY : src/CMakeFiles/Feature.dir/build

src/CMakeFiles/Feature.dir/requires: src/CMakeFiles/Feature.dir/feature.cpp.o.requires
.PHONY : src/CMakeFiles/Feature.dir/requires

src/CMakeFiles/Feature.dir/clean:
	cd /home/robotqw/robotqw/SLAM/Project/Part3/build/src && $(CMAKE_COMMAND) -P CMakeFiles/Feature.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/Feature.dir/clean

src/CMakeFiles/Feature.dir/depend:
	cd /home/robotqw/robotqw/SLAM/Project/Part3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robotqw/robotqw/SLAM/Project/Part3 /home/robotqw/robotqw/SLAM/Project/Part3/src /home/robotqw/robotqw/SLAM/Project/Part3/build /home/robotqw/robotqw/SLAM/Project/Part3/build/src /home/robotqw/robotqw/SLAM/Project/Part3/build/src/CMakeFiles/Feature.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/Feature.dir/depend
