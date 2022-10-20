#!/bin/sh

export PREDICT_TIME=1017
export TFLITE_MODEL_ID=forceconcat_20221017_0_default
export TFLITE_MODEL_PATH=/deepiano_data/zhaoliang/project/BGMcloak/tflite_model_files


echo "预测开始"
sh ./predict_union_40_0411.sh &
sh ./predict_union_40_0414.sh &
sh ./predict_union_40_0420.sh &
sh ./predict_union_40_0428.sh &
sh ./predict_union_40_0429.sh &
sh ./predict_union_40_0823.sh &
sh ./predict_union_40_0824.sh &
sh ./predict_union_40_0825.sh &
sh ./predict_union_40_high-20.sh &
sh ./predict_union_40_peilian.sh &
sh ./predict_union_40_single-20.sh &
sh ./predict_union_40_specail.sh &
sh ./predict_union_40_0721-22.sh &
sh ./predict_union_40_0725.sh &
sh ./predict_union_40_0726.sh &
sh ./predict_union_40_0727.sh &
sh ./predict_union_40_0728.sh &
sh ./predict_union_40_aitagging_mix.sh &
sh ./predict_union_40_maestro-v3.0.0_mix.sh &
sh ./predict_union_40_xuanran.sh &


wait
echo 'done'


