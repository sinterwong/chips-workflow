CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(X3-Tests)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-error=unused-function")
INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_INCLUDE_DIR}/core
    ${PROJECT_INCLUDE_DIR}/infer
    ${PROJECT_INCLUDE_DIR}/video
    ${OpenCV_INCLUDE_DIRS}
    ${X3_INCLUDE}
)

LINK_DIRECTORIES(
    ${OpenCV_LIBRARY_DIR}
    ${X3_DIRECTORIES}
)

LINK_LIBRARIES(
    ${OpenCV_LIBS}
    gflags::gflags
    pthread
    spdlog
    ${X3_LIBS}
)

SET(LINK "-Wl,--no-as-needed")
FILE(GLOB APP_HARDWARE_SOURCES *.cpp)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin/x3)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    # TARGET_LINK_LIBRARIES(${testname} PRIVATE ${LINK} core module module_logger stdc++fs)

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
