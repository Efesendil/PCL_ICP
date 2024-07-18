cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 17)
project(ICP_PROJECT)

# Set the PCL directory
set(PCL_DIR "/opt/Libraries/pcl-pcl-1.14.0/build/PCLConfig.cmake")

# Find PCL package
find_package(PCL 1.14.0 REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Add executable
add_executable(pipeline src/main.cpp src/pipeline.cpp)

# Link libraries to the executable
target_link_libraries(pipeline ${PCL_LIBRARIES})

# Debugging messages
message(STATUS "PCL include directories: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL library directories: ${PCL_LIBRARY_DIRS}")

