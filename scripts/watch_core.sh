#!/bin/bash

# 检查参数个数
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 WATCH_DIR CORE_PREFIX APP_PATH"
    exit 1
fi

# 获取参数
WATCH_DIR="$1"  # 要监控的目录
APP_PATH="$2"  # 应用程序的路径
LOG_FILE="$3"  # 日志文件名

# 使用 inotifywait 进行文件系统监控
inotifywait -m -e close_write $WATCH_DIR | while read path action file; do
    # 检查文件名是否符合 core dump 文件的命名模式
    if [[ "$file" == "core."* ]]; then
        # 添加时间戳到日志
        echo "------------------------" >> $LOG_FILE
        echo "Timestamp: $(date)" >> $LOG_FILE
        echo "Core dump detected at $path$file" >> $LOG_FILE
        
        # 添加文件信息到日志
        echo "File details:" >> $LOG_FILE
        ls -lh "$path$file" >> $LOG_FILE
        
        # 应用程序资源使用情况
        echo "Application resource usage:" >> $LOG_FILE
        pmap $(pgrep $(basename $APP_PATH)) >> $LOG_FILE
        lsof -p $(pgrep $(basename $APP_PATH)) >> $LOG_FILE
        
        # 近期操作系统级别的日志
        echo "Recent OS logs:" >> $LOG_FILE
        dmesg | tail -n 10 >> $LOG_FILE
        
        # 使用 gdb 提取基本信息
        echo "GDB Backtrace:" >> $LOG_FILE
        gdb -q $APP_PATH $path$file <<EOF >> $LOG_FILE 2>&1
          set pagination 0
          thread apply all bt
          info registers
          quit
EOF
    fi
done