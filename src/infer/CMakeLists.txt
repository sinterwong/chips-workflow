CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FILE(GLOB INFER_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

IF(TARGET_HARDWARE STREQUAL jetson)
    # 0: no-simd; 1: sse-Intel; 2: neon-TX2;
    SET(SIMD 2)
    MESSAGE(STATUS "use neon-TX2 to compile")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
    ADD_DEFINITIONS(-DUSE_SIMD)
    ADD_DEFINITIONS(-DUSE_JETSON)
    # if(SIMD EQUAL 1)
    # message(STATUS "use sse-Intel to compile")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4")
    # add_definitions(-DUSE_SIMD)
    # elseif(SIMD EQUAL 2)
    
    # elseif(SIMD EQUAL 0)
    #     message(STATUS "no-simd to compile")
    # endif()
    
    LINK_DIRECTORIES(
        ${CUDA_LIBRARY_DIR}
        # ${FMT_LIBRARY_DIR}
    )
    FILE(GLOB NN_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.h)
    FILE(GLOB NN_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.c)
    FILE(GLOB NN_LIB_CU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cu)
    ADD_LIBRARY(infer SHARED  ${NN_LIB_SRC})
    CUDA_ADD_LIBRARY(infer_cu SHARED  ${NN_LIB_CU_SRC})
    TARGET_INCLUDE_DIRECTORIES(infer_cu PUBLIC 
                                ${3RDPARTY_DIR}/spdlog/include
                                ${PROJECT_INCLUDE_DIR}
                                ${PROJECT_INCLUDE_DIR}/infer 
                                ${OpenCV_INCLUDE_DIRS}
                                ${CUDA_INCLUDE_DIRS}
    )
    TARGET_INCLUDE_DIRECTORIES(infer PUBLIC 
                                ${3RDPARTY_DIR}/spdlog/include
                                ${PROJECT_INCLUDE_DIR}
                                ${PROJECT_INCLUDE_DIR}/infer
                                ${OpenCV_INCLUDE_DIRS}
                                ${CUDA_INCLUDE_DIRS}
    )
    TARGET_LINK_LIBRARIES(infer ${OpenCV_LIBS} ${CUDA_LIBS} module_logger infer_cu nvinfer nvparsers nvonnxparser nvinfer_plugin)
ELSE()
    MESSAGE(FATAL_ERROR "No platform specified!")
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

INSTALL(TARGETS infer infer_cu
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
