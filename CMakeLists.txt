cmake_minimum_required(VERSION 3.17)
project(geometry_utils VERSION 0.1
                       DESCRIPTION "Some geometry utils."
                       LANGUAGES CXX)
enable_testing()

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${PROJECT_SOURCE_DIR}/cmake
)

# Find the eigen package.
find_package(Eigen REQUIRED)
if (${Eigen_FOUND})
  MESSAGE("Eigen Includes: ${EIGEN_INCLUDE_DIRS}")
endif()

# Main project library
add_library(${PROJECT_NAME} INTERFACE)

# Turn on C++14
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)

# Specify both build and install include location
# TODO(gareth): Not sure this is correct
target_include_directories(${PROJECT_NAME} 
  INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN_INCLUDE_DIRS}
)

# Unit tests
add_subdirectory(test)
