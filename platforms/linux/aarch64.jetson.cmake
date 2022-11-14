MESSAGE(STATUS "Configure Cross Compiler")

# SET(CMAKE_SYSTEM_NAME Linux)  # 加上之后找不到CUDA 我也不知道为啥
SET(CMAKE_SYSTEM_PROCESSOR aarch64)
SET(TARGET_OS linux)
SET(TARGET_ARCH aarch64)
SET(TARGET_HARDWARE jetson)

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag for cross-compiled process
SET(CROSS_COMPILATION_ARM jetson)
SET(CROSS_COMPILATION_ARCHITECTURE aarch64)

SET(CMAKE_C_COMPILER       ${TOOLCHAIN_ROOTDIR}/gcc)
SET(CMAKE_CXX_COMPILER     ${TOOLCHAIN_ROOTDIR}/g++)

# 0: no-simd; 1: sse-Intel; 2: neon-TX2;
SET(SIMD 2)
MESSAGE(STATUS "use neon-TX2 to compile")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ffast-math -flto -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
ADD_DEFINITIONS(-DUSE_SIMD)
ADD_DEFINITIONS(-DUSE_JETSON)

# set searching rules for cross-compiler
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

SET(CMAKE_SKIP_BUILD_RPATH TRUE)
SET(CMAKE_SKIP_RPATH TRUE)

# set g++ param
# -fopenmp link libgomp
# SET(CMAKE_CXX_FLAGS "-std=c++17 -march=armv8-a -mfloat-abi=softfp -mfpu=neon-vfpv4 \
#     -ffunction-sections \
#     -fdata-sections -O2 -fstack-protector-strong -lm -ldl -lstdc++\
#     ${CMAKE_CXX_FLAGS}")
