CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FILE(GLOB SWMODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
FILE(GLOB SWMODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

IF(TARGET_HARDWARE STREQUAL jetson)
    # 0: no-simd; 1: sse-Intel; 2: neon-TX2;
    SET(SIMD 2)
    MESSAGE(STATUS "use neon-TX2 to compile")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
    ADD_DEFINITIONS(-DUSE_SIMD)
    ADD_DEFINITIONS(-DUSE_JETSON)

    LINK_DIRECTORIES(
        ${CUDA_LIBRARY_DIR}
        ${CURL_LIBRARY_DIR}
    )
    FILE(GLOB_RECURSE MODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.h)
    FILE(GLOB_RECURSE MODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.c)
    ADD_LIBRARY(module SHARED  ${SWMODULE_LIB_SRC} ${MODULE_LIB_SRC})
    TARGET_INCLUDE_DIRECTORIES(module PUBLIC 
                                ${3RDPARTY_DIR}/spdlog/include
                                ${PROJECT_INCLUDE_DIR}
                                ${PROJECT_INCLUDE_DIR}/module
                                ${PROJECT_INCLUDE_DIR}/core
                                ${PROJECT_INCLUDE_DIR}/infer
                                ${PROJECT_INCLUDE_DIR}/jetson-utils
                                ${OpenCV_INCLUDE_DIRS}
                                ${Boost_INCLUDE_DIRS}
                                ${Protobuf_INCLUDE_DIR}
                                ${CUDA_INCLUDE_DIRS}
                                ${CURL_INCLUDE_DIR}
    )
    TARGET_LINK_LIBRARIES(module stdc++fs core infer module_logger jetson-utils utils ${CUDA_LIBS} ${CURL_LIBS})
ELSEIF(TARGET_HARDWARE STREQUAL rv1126)
    # 0: no-simd; 1: sse-Intel; 2: neon-TX2;
    LINK_DIRECTORIES(
        ${CURL_LIBRARY_DIR}
    )
    FILE(GLOB_RECURSE MODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/rockchip/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.h)
    FILE(GLOB_RECURSE MODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/rockchip/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.c)
    ADD_LIBRARY(module SHARED  ${SWMODULE_LIB_SRC} ${MODULE_LIB_SRC})
    TARGET_INCLUDE_DIRECTORIES(module PUBLIC 
                                ${3RDPARTY_DIR}/spdlog/include
                                ${PROJECT_INCLUDE_DIR}
                                ${PROJECT_INCLUDE_DIR}/module
                                ${PROJECT_INCLUDE_DIR}/core
                                ${PROJECT_INCLUDE_DIR}/infer
                                ${OpenCV_INCLUDE_DIRS}
                                ${Protobuf_INCLUDE_DIR}
                                ${CUDA_INCLUDE_DIRS}
                                ${CURL_INCLUDE_DIR}
    )
    TARGET_LINK_LIBRARIES(module stdc++fs core infer module_logger ${CURL_LIBS})
ELSE()
    MESSAGE(WARNING "No valid hardware is specified!")
ENDIF()


# transfer all headers to the include directory 
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/module)
FOREACH(include ${SWMODULE_LIB_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/module COPYONLY)
ENDFOREACH()

# transfer all headers to the include directory 
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/module/${TARGET_HARDWARE})
FOREACH(include ${MODULE_LIB_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/module/${TARGET_HARDWARE} COPYONLY)
ENDFOREACH()

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/module
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp" EXCLUDE
    PATTERN "CMakeLists.txt" EXCLUDE
)

INSTALL(TARGETS module
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
