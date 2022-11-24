#!/bin/bash
while getopts i:o: OPT; do
  case ${OPT} in
    i) INPUT_ROOT=${OPTARG} # 输入目录
       ;;
    o) OUTPUT_ROOT=${OPTARG} # 输出目录
       ;;
    \?)
       echo "[Usage] `date '+%F %T'` `basename $0` -s your input root -e your output root\n" >&2
       exit 1
  esac
done

mkdir -p ${OUTPUT_ROOT}

# for file in $(ls ${INPUT_ROOT}/* | sed "s:^:${INPUT_ROOT}/:")
for file in $(ls ${INPUT_ROOT}/*.onnx)
do
    filename=$(basename ${file})
    old_suffix=".onnx"
    new_suffix=".engine"
    out_filename=${filename/${old_suffix}/${new_suffix}}
    echo "converting the ${file}"
    outfile=${OUTPUT_ROOT}/${out_filename}
    $(trtexec --onnx=${file} --saveEngine=${outfile} --fp16)
    echo "${outfile} has converted!"
done
