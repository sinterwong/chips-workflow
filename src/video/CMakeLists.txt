CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(FlowEngineModule LANGUAGES C CXX)

FILE(GLOB VIDEO_LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.c ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.c)
FILE(GLOB VIDEO_LIB_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_HARDWARE}/*.h)
ADD_LIBRARY(video SHARED ${VIDEO_LIB_HEAD} ${VIDEO_LIB_SRC})

IF(TARGET_HARDWARE STREQUAL jetson)
  ADD_SUBDIRECTORY(jetson-utils)
  MESSAGE("++++++++++++++++++++++++++" ${PROJECT_INCLUDE_DIR}/jetson-utils)
  TARGET_INCLUDE_DIRECTORIES(video PRIVATE 
                            ${PROJECT_INCLUDE_DIR}
                            ${PROJECT_INCLUDE_DIR}/jetson-utils
                            ${PROJECT_INCLUDE_DIR}/core
                            ${PROJECT_INCLUDE_DIR}/video
                            ${OpenCV_INCLUDE_DIRS}
                            ${CUDA_INCLUDE_DIRS}
                            ${FFMPEG_INCLUDE_DIR}
  )
  TARGET_LINK_LIBRARIES(video jetson-utils ${OpenCV_LIBS} module_logger)
ELSEIF(TARGET_HARDWARE STREQUAL x3)
  LINK_DIRECTORIES(
    ${APPSDK_PATH}/lib/hbmedia
  )
  TARGET_INCLUDE_DIRECTORIES(video PRIVATE 
      ${PROJECT_INCLUDE_DIR}
      ${PROJECT_INCLUDE_DIR}/core
      ${PROJECT_INCLUDE_DIR}/video
      ${OpenCV_INCLUDE_DIRS}
      ${APPSDK_PATH}/include
      ${APPSDK_PATH}/include/vio
  )
  TARGET_LINK_LIBRARIES(video ${OpenCV_LIBS} ${HB_MEDIA_libs} module_logger)
ELSEIF(TARGET_HARDWARE STREQUAL rockchip)
ELSE()
    MESSAGE(WARNING "No platform specified!")
ENDIF()

# transfer all headers to the include directory 
FILE(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/video)
FOREACH(include ${VIDEO_LIB_HEAD})
	MESSAGE("-- Copying ${include}")
	CONFIGURE_FILE(${include} ${PROJECT_INCLUDE_DIR}/video COPYONLY)
ENDFOREACH()

INSTALL(TARGETS video
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
