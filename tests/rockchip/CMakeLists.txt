CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Rockchip-Tests)
MESSAGE(STATUS "Build rockchip test unit" ${3RDPARTY_DIR} ${MPP_LIB})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/infer
    ${PROJECT_INCLUDE_DIR}/video
    ${OpenCV_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIR}
    ${RKNN_INCLUDE_DIR}
    ${MPP_INCLUDE_DIR}
    ${3RDPARTY_DIR}/gflags/include
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
)

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    ${RKNN_API_LIB}
    ${RGA_LIB}
    ${MPP_LIB}
    gflags
    avcodec
    avformat
    infer
    video
    pthread
    module_logger
)

FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/rockchip)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/rockchip)
FILE(GLOB APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")

FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE(${testname} ${sourcefile})
    # INSTALL(TARGETS ${testname}
    #     ARCHIVE DESTINATION lib
    #     LIBRARY DESTINATION lib
    #     RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
