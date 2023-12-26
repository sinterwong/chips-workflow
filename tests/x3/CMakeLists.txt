CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(X3-Tests)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/infer
    ${PROJECT_INCLUDE_DIR}/video
    ${OpenCV_INCLUDE_DIRS}
    ${X3_INCLUDE}
    ${3RDPARTY_DIR}/gflags/include
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
    ${X3_DIRECTORIES}
)

LINK_LIBRARIES(
    gflags
    pthread
    video 
    infer
    module_logger
    ${X3_LIBS}
    ${OpenCV_LIBS}
    ${FFMPEG_LIBS}
    stdc++fs
)

SET(LINK "-Wl,--no-as-needed")
FILE(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/x3)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/x3)
FILE(GLOB APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
