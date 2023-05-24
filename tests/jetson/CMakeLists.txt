CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Jetson-Tests)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/core
    ${PROJECT_INCLUDE_DIR}/infer
    ${PROJECT_INCLUDE_DIR}/video
    ${PROJECT_INCLUDE_DIR}/jetson-utils
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
)

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    infer
    video
    gflags::gflags
    pthread
    spdlog
)

SET(LINK "-Wl,--no-as-needed")
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/jetson)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/jetson)
FILE(GLOB APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")

FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE(${testname} ${sourcefile})

    # TARGET_LINK_LIBRARIES(${testname} PRIVATE ${LINK} core module module_logger stdc++fs)
    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
