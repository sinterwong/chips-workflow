#!/bin/bash
while getopts i: OPT; do
  case ${OPT} in
    i) INPUT_ROOT=${OPTARG} # 输入目录
       ;;
    \?)
       echo "[Usage] `date '+%F %T'` `basename $0` -s your input root\n" >&2
       exit 1
  esac
done

# for file in $(ls ${INPUT_ROOT}/* | sed "s:^:${INPUT_ROOT}/:")
for file in $(ls ${INPUT_ROOT}/*.engine)
do
   # filename=$(basename ${file})
   output=$(trtexec --loadEngine=${file})
   dimension=$(echo "$output" | sed -n 's/.*with dimensions \([0-9]*x[0-9]*x[0-9]*x[0-9]*\).*/\1/p')
   echo "Shapes are: $dimension"
   len=`expr length "$output"`
   info=${output:$len-200:$len}
   if [ "$(echo "$info" | grep "PASSED")" ]; then
      echo "Engine $file is passed"
   else
      echo "Engine $file is failed"
   fi
done
