#!/bin/bash

# 检查requirements.txt文件是否存在
if [[ ! -f image-build-requirements.txt ]]; then
    echo "image-build-requirements.txt 文件不存在！"
    exit 1
fi

# 读取requirements.txt文件中的所有行
mapfile -t lines < image-build-requirements.txt

# 计算每个分文件中的行数
total_lines=${#lines[@]}
lines_per_file=$(( (total_lines + 3) / 4 ))

# 分割文件
for i in {0..3}; do
    start=$(( i * lines_per_file + 1 ))
    end=$(( (i + 1) * lines_per_file ))
    if (( end > total_lines )); then
        end=$total_lines
    fi
    sed -n "${start},${end}p" image-build-requirements.txt > "image-build-requirements${i}.txt"
done

echo "分割完成：image-build-requirements0.txt - image-build-requirements3.txt"

