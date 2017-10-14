cmake_minimum_required(VERSION 2.8)
project(apriltag4cmake)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 2.4.3 QUIET)


find_package(Threads REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/common
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/tag36h11.c
src/tag36h10.c
src/tag36artoolkit.c
src/tag25h9.c
src/tag25h7.c
src/tag16h5.c
src/apriltag_quad_thresh.c
src/apriltag.c
)

add_library(common SHARED
common/zmaxheap.c
common/zhash.c
common/zarray.c
common/workerpool.c
common/unionfind.c
common/time_util.c
common/svd22.c
common/string_util.c
common/pnm.c
common/pjpeg-idct.c
common/pjpeg.c
common/pam.c
common/matd.c
common/image_u8x4.c
common/image_u8x3.c
common/image_u8.c
common/image_f32.c
common/homography.c
common/getopt.c
common/g2d.c
)

target_link_libraries(${PROJECT_NAME}
common
${OpenCV_LIBS}
${CMAKE_THREAD_LIBS_INIT}
)


# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples)

add_executable(opencv_demo
Examples/opencv_demo.cc)
target_link_libraries(opencv_demo ${PROJECT_NAME})

add_executable(main
Examples/main.cc)
target_link_libraries(main ${PROJECT_NAME})

