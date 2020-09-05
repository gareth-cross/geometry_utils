include(FetchContent)

# Download google test.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

# # Build testing helper library.
# add_library(${PROJECT_NAME}_test_utils test_utils.cc)
# target_link_libraries(${PROJECT_NAME}_test_utils STATIC ${PROJECT_NAME})
# target_include_directories(${PROJECT_NAME}_test_utils PUBLIC ${GTEST_INCLUDE_DIRS})

# # Find all the test files.
# FILE(GLOB ${PROJECT_NAME}_TEST_SOURCES "*_test.cc")

# foreach(test_file ${${PROJECT_NAME}_TEST_SOURCES})
#   # get just the filename w/o extension
#   get_filename_component(test_name ${test_file} NAME_WE)
#   # add test executable for every file
#   add_executable(${test_name} ${test_file})
#   target_link_libraries(${test_name}
#     ${PROJECT_NAME}
#     ${PROJECT_NAME}_test_utils
#     ${GTEST_BOTH_LIBRARIES}
#     )

#   # turn on maximum warnings
#   if(MSVC)
#     target_compile_options(${test_name} PRIVATE /W4 /WX)
#   else()
#     target_compile_options(${test_name} PRIVATE -Wall -Wextra -pedantic -Werror)
#   endif()

#   if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
#     message("Building ${test_name} with -DNDEBUG")
#     target_compile_definitions(${test_name} PUBLIC NDEBUG)
#   endif()

#   add_test(${test_name} ${test_name})
# endforeach(test_file)
