CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(FlowEngineModule LANGUAGES C CXX)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")

FILE(GLOB SWMODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
FILE(GLOB SWMODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

IF(TARGET_HARDWARE STREQUAL jetson)
    FILE(GLOB_RECURSE MODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.h)
    FILE(GLOB_RECURSE MODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.c)
    LIST(APPEND SWMODULE_LIB_SRC ${MODULE_LIB_SRC})
    ADD_LIBRARY(module SHARED ${SWMODULE_LIB_HEAD} ${MODULE_LIB_HEAD} ${SWMODULE_LIB_SRC})
    TARGET_INCLUDE_DIRECTORIES(module PRIVATE 
                                ${PROJECT_INCLUDE_DIR}
                                ${PROJECT_INCLUDE_DIR}/module
                                ${PROJECT_INCLUDE_DIR}/core
                                ${PROJECT_INCLUDE_DIR}/infer
                                ${PROJECT_INCLUDE_DIR}/jetson-utils
                                ${CURL_INCLUDE_DIR}
                                ${OpenCV_INCLUDE_DIRS}
                                ${Protobuf_INCLUDE_DIR}
                                ${CUDA_INCLUDE_DIRS}
    )
    TARGET_LINK_LIBRARIES(module jetson-utils module_logger)
ELSEIF(TARGET_HARDWARE STREQUAL x3)
    FILE(GLOB MODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.h)
    FILE(GLOB MODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.c)
    LIST(APPEND SWMODULE_LIB_SRC ${MODULE_LIB_SRC})
    ADD_LIBRARY(module SHARED ${SWMODULE_LIB_HEAD} ${MODULE_LIB_HEAD} ${SWMODULE_LIB_SRC})
    TARGET_INCLUDE_DIRECTORIES(module PRIVATE 
                                ${PROJECT_INCLUDE_DIR}
                                ${PROJECT_INCLUDE_DIR}/module
                                ${PROJECT_INCLUDE_DIR}/core
                                ${PROJECT_INCLUDE_DIR}/infer
                                ${CURL_INCLUDE_DIR}
                                ${OpenCV_INCLUDE_DIRS}
    )
    TARGET_LINK_LIBRARIES(module infer module_logger ${CURL_LIBRARIES})
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

# INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/module
#     FILES_MATCHING
#     PATTERN "*.h"
#     PATTERN "*.hpp" EXCLUDE
#     PATTERN "CMakeLists.txt" EXCLUDE
# )

INSTALL(TARGETS module
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
