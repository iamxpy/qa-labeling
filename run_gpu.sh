#!/bin/sh

while true
do
  count=$(ps -ef | grep -c kk_test)
  if [ $count -lt 2 ]
    then
     # 改动项 查询第1块gpu的容量--2p 第2块3p--2  第三块--4p  第四块--5p
     stat2=$(gpustat | awk '{print $9}' | sed -n '3p')
     # echo $stat2
     if [ "$stat2" -lt 1000 ]
       then
         echo 'run 2'
         #改动项 前面输入占用的gpu id 后面是运行代码
         CUDA_VISIBLE_DEVICES=0,1 nohup python -u swa_train.py > exp53_log &

         sleep 500

         exit 0
     fi
  fi
  sleep 2
done
