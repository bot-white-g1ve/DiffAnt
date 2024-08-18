#!/bin/bash

# 远程服务器信息
REMOTE_USER="xthu"   # 替换为你的远程服务器用户名
REMOTE_HOST="xuchang-lab3.staff.sydney.edu.au"   # 替换为远程服务器IP或域名
REMOTE_DIR="/hdd/xthu/CholecT45/data/VID10"   # 替换为远程服务器上的目标文件夹路径

REMOTE_TRIPLET_FILE="/hdd/xthu/CholecT45/triplet/VID10.txt"

# 本地目标目录
LOCAL_DIR="./VID10"  # 当前目录

# 循环从000000.png到000100.png并转移到本地
for i in $(seq -f "%06g" 0 100)
do
  FILE_NAME="${i}.png"
  REMOTE_FILE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${FILE_NAME}"
  scp $REMOTE_FILE $LOCAL_DIR
done



echo "Files have been transferred successfully!"
