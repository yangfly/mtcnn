cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project("mtcnn")

### ------------------- Find Packages --------------------
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} REQUIRED)

list(APPEND CMAKE_PREFIX_PATH ${Caffe_ROOT})
find_package(Caffe REQUIRED)
if (Caffe_FOUND)
	message (STATUS "Caffe_LIBRARIES=${Caffe_LIBRARIES}")
endif()
### --------------------------------------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
file(GLOB PROJECT_SRCS ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB PROJECT_HEADERS ${PROJECT_SOURCE_DIR}/*.h)

add_library(${PROJECT_NAME} SHARED ${PROJECT_SRCS} ${PROJECT_HEADERS}) 

target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${Caffe_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${Caffe_LIBRARIES}
        )

#---------------------------   INSTALLATION    -----------------------------------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES
                      PREFIX "_"
                      INSTALL_RPATH_USE_LINK_PATH TRUE)
#install(TARGETS ${PROJECT_NAME} 
#        RUNTIME DESTINATION .
#        LIBRARY DESTINATION .
#        ARCHIVE DESTINATION .)
# install(FILES ${CMAKE_BUILD_DIR}/_${PROJECT_NAME}.so DESTINATION .)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/ DESTINATION .)
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION .)

