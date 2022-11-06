CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(FlowEngineInfer LANGUAGES C CXX)

FILE(GLOB INFER_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
FILE(GLOB INFER_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

INCLUDE_DIRECTORIES(
    ${3RDPARTY_DIR}/spdlog/include
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/infer
)

LINK_LIBRARIES(module_logger)
ADD_LIBRARY(infer SHARED ${INFER_LIB_SRC})
SET(INFER_LIBS infer)

IF(TARGET_HARDWARE STREQUAL jetson)
    LINK_DIRECTORIES(
        ${CUDA_LIBRARY_DIR}
    )
    FILE(GLOB NN_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.h)
    FILE(GLOB NN_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.c)
    FILE(GLOB NN_LIB_CU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cu)
    ADD_LIBRARY(jetson_infer SHARED ${NN_LIB_SRC})
    TARGET_INCLUDE_DIRECTORIES(jetson_infer PUBLIC 
                                ${OpenCV_INCLUDE_DIRS}
                                ${CUDA_INCLUDE_DIRS}
    )
    CUDA_ADD_LIBRARY(jetson_infer_cu SHARED ${NN_LIB_CU_SRC})
    TARGET_INCLUDE_DIRECTORIES(jetson_infer_cu PUBLIC 
                                ${OpenCV_INCLUDE_DIRS}
                                ${CUDA_INCLUDE_DIRS}
    )
    TARGET_LINK_LIBRARIES(jetson_infer ${INFER_DEPENDENT_LIBS} ${OpenCV_LIBS} ${CUDA_LIBS} jetson_infer_cu  nvinfer nvparsers nvonnxparser nvinfer_plugin infer)
    LIST(APPEND INFER_LIBS jetson_infer jetson_infer_cu)
ELSEIF(TARGET_HARDWARE STREQUAL x3)
    LINK_DIRECTORIES(
        ${DNN_LIB_PATH}
        ${APPSDK_PATH}/lib
        ${APPSDK_PATH}/lib/hbbpu
        ${APPSDK_PATH}/lib/hbmedia
    )
    INCLUDE_DIRECTORIES(
        ${DNN_PATH}/include
        ${APPSDK_PATH}/include
        ${APPSDK_PATH}/include/vio
        ${APPSDK_PATH}/include/libmm
    )
    FILE(GLOB_RECURSE NN_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.h)
    FILE(GLOB_RECURSE NN_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.c)
    ADD_LIBRARY(x3_infer SHARED ${NN_LIB_SRC} ${NN_LIB_HEAD})
    TARGET_LINK_LIBRARIES(x3_infer ${BPU_libs} ${HB_MEDIA_libs} rt dl infer)
    LIST(APPEND INFER_LIBS x3_infer)
ELSE()
    MESSAGE(WARNING "No platform specified!")
ENDIF()

# transfer all headers to the include directory 
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/infer)
FOREACH(include ${INFER_LIB_HEAD}) 
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/infer COPYONLY)
ENDFOREACH()

# transfer all headers to the include directory 
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/infer/${TARGET_HARDWARE})
FOREACH(include ${NN_LIB_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/infer/${TARGET_HARDWARE} COPYONLY)
ENDFOREACH()

# INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/infer
#     FILES_MATCHING
#     PATTERN "*.h"
#     PATTERN "*.hpp" EXCLUDE
#     PATTERN "CMakeLists.txt" EXCLUDE
# )

INSTALL(TARGETS ${INFER_LIBS}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
