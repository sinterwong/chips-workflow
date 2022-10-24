CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FILE(GLOB SWMODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
FILE(GLOB SWMODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
LINK_DIRECTORIES(
    ${CURL_LIBRARY_DIR}
)
ADD_LIBRARY(module SHARED ${SWMODULE_LIB_SRC})
SET(MODULE_LIBS module)
SET(MODULE_DEPENDENT_INCLUDE ${3RDPARTY_DIR}/spdlog/include
                        ${PROJECT_INCLUDE_DIR}
                        ${PROJECT_INCLUDE_DIR}/module
                        ${PROJECT_INCLUDE_DIR}/core
                        ${PROJECT_INCLUDE_DIR}/infer
                        ${CURL_INCLUDE_DIR})

SET(MODULE_DEPENDENT_LIBS stdc++fs core module_logger utils ${CURL_LIBS})
TARGET_INCLUDE_DIRECTORIES(module PUBLIC ${MODULE_DEPENDENT_INCLUDE})
TARGET_LINK_LIBRARIES(module ${MODULE_DEPENDENT_LIBS})

IF(TARGET_HARDWARE STREQUAL jetson)
    LINK_DIRECTORIES(
        ${CUDA_LIBRARY_DIR}
    )
    FILE(GLOB_RECURSE MODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.h)
    FILE(GLOB_RECURSE MODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/jetson/*.c)
    ADD_LIBRARY(jetson_module SHARED ${MODULE_LIB_SRC})
    TARGET_INCLUDE_DIRECTORIES(jetson_module PUBLIC 
                                ${MODULE_DEPENDENT_INCLUDE}
                                ${PROJECT_INCLUDE_DIR}/jetson-utils
                                ${OpenCV_INCLUDE_DIRS}
                                ${Protobuf_INCLUDE_DIR}
                                ${CUDA_INCLUDE_DIRS}
    )
    TARGET_LINK_LIBRARIES(jetson_module jetson_infer jetson_infer_cu jetson-utils ${CUDA_LIBS} ${MODULE_DEPENDENT_LIBS})
    LIST(APPEND MODULE_LIBS jetson_module)
ELSEIF(TARGET_HARDWARE STREQUAL x3)
    LINK_DIRECTORIES(
        ${DNN_LIB_PATH}
        ${APPSDK_PATH}/lib
        ${APPSDK_PATH}/lib/hbbpu
        ${APPSDK_PATH}/lib/hbmedia
    )
    FILE(GLOB_RECURSE MODULE_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.h)
    FILE(GLOB_RECURSE MODULE_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/x3/*.c)
    ADD_LIBRARY(x3_module SHARED ${MODULE_LIB_SRC})
    TARGET_INCLUDE_DIRECTORIES(x3_module PUBLIC 
                                ${MODULE_DEPENDENT_INCLUDE}
                                ${DNN_PATH}/include
                                ${APPSDK_PATH}/include
                                ${APPSDK_PATH}/include/vio
                                ${APPSDK_PATH}/include/libmm
    )
    TARGET_LINK_LIBRARIES(x3_module ${BPU_libs} ${HB_MEDIA_libs} rt dl x3_infer ${MODULE_DEPENDENT_LIBS})
    LIST(APPEND MODULE_LIBS x3_module)
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

INSTALL(TARGETS ${MODULE_LIBS}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
