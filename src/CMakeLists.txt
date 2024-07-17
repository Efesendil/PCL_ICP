cmake_minimum_required(VERSION 2.9 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 17)
project(ICP_PROJECT)

# Set the PCL directory
set(PCL_DIR "/home/efesendil/Libraries/pcl-pcl-1.14.0/build/PCLConfig.cmake")
find_package(PCL 1.14.0 REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/home/efesendil/Projects/ICP_PCL)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add executable for pipeline
add_executable(pipeline main.cpp)

# Link libraries to the pipeline target
target_link_libraries(pipeline ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FEATURES_LIBRARIES})

# Debugging messages
message(STATUS "Pipeline folder: ${Pipeline_folder}")
message(STATUS "Pipeline source file: ${Pipeline_folder}/main.cpp")
