cmake_minimum_required(VERSION 3.17)
project(geometry_utils VERSION 0.1
    DESCRIPTION "Some geometry utils."
    LANGUAGES CXX)
enable_testing()
include(FetchContent)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
    )

# Maybe download eigen, if this option is specified.
option(GEO_UTILS_FETCH_EIGEN "Clone eigen instead of searching on the system for it" OFF)
if (${GEO_UTILS_FETCH_EIGEN})
  message(STATUS "Cloning Eigen from gitlab")
  FetchContent_Declare(
      eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.3.7
  )
  FetchContent_GetProperties(eigen)
  if (NOT eigen_POPULATED)
    # Turn off eigen tests.
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Populate(eigen)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif ()
endif ()

# Find the eigen package.
if ((NOT TARGET eigen) AND (NOT GEO_UTILS_FETCH_EIGEN))
  find_package(Eigen REQUIRED)
  if (${EIGEN_FOUND})
    message("Eigen Includes: ${EIGEN_INCLUDE_DIRS}")
  else ()
    message(FATAL_ERROR "Failed to find Eigen")
  endif ()
endif ()

# Main project library
add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE eigen)

# Turn on C++14
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)

# Specify both build and install include location
target_include_directories(${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

# Unit tests
option(GEO_UTILS_BUILD_TESTING "Build tests for geometry_utils" ON)
if (${GEO_UTILS_BUILD_TESTING})
  add_subdirectory(test)
endif ()

# make cache variables for install destinations
include(GNUInstallDirs)

# Specify public headers
FILE(GLOB_RECURSE ${PROJECT_NAME}_HEADERS "include/*.hpp")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${${PROJECT_NAME}_HEADERS}")

# Setup install target
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    # Install headers to sub-directory of /usr/local/include (or whatever the prefix is).
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
)

# Check if we want to install Eigen as well:
option(GEO_UTILS_INSTALL_EIGEN "Add a install target for Eigen public headers" OFF)
if (${EIGEN_FOUND})
  if (${GEO_UTILS_INSTALL_EIGEN})
    # Install eigen as well. We use DIRECTORY since INTERFACE target doesn't work w/ INSTALL
    message(STATUS "Will install eigen headers to \"${CMAKE_INSTALL_INCLUDEDIR}\"")
    install(DIRECTORY "${EIGEN_INCLUDE_DIRS}/Eigen" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  endif ()
endif ()
