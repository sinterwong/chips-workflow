#!/bin/bash

# 定义要测试的端口范围
ports="80 8080 19999 9888"

# 定义要测试的服务器列表，以空格分隔
servers="172.17.0.11 192.168.0.188 192.168.31.227"

timeout=2

# 定义输出文件的路径
output_file="closed_ports.txt"

# 循环遍历服务器列表和端口范围
for server in ${servers}; do
  for port in ${ports}; do
    # 使用 telnet 进行端口测试，并设置超时时间
    {
      (echo >/dev/tcp/"$server"/"$port") & telnet_pid=$!
      sleep $timeout
    } &> /tmp/telnet_output

    if kill -0 $telnet_pid >/dev/null 2>&1; then
      # telnet 进程仍在运行，连接超时
      echo "Connection to $server:$port timed out"
      kill $telnet_pid >/dev/null 2>&1
    else
      # telnet 进程已退出，连接成功
      output=$(cat /tmp/telnet_output)
      if [[ "$output" != *refused* ]]; then
        echo "Port $port is open on $server"
      else
        echo "Port $port is closed on $server"
        echo "$server:$port" >> "$output_file"
      fi
    fi
  done
done
