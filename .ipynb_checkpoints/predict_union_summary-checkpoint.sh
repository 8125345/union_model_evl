#!/bin/sh

export PREDICT_TIME=909
export TFLITE_MODEL_ID=forceconcat_20220908_0_default
export TFLITE_MODEL_PATH=/data/projects/test_models

echo "预测开始"
sh ./predict_union_40_0721-22.sh &
sh ./predict_union_40_0725.sh &
sh ./predict_union_40_0726.sh &
sh ./predict_union_40_0727.sh &
sh ./predict_union_40_0728.sh &
sh ./predict_union_40_aitagging_mix.sh &
sh ./predict_union_40_maestro-v3.0.0_mix.sh &
sh ./predict_union_160_0721-22.sh &
sh ./predict_union_160_0725.sh &
sh ./predict_union_160_0726.sh &
sh ./predict_union_160_0727.sh &
sh ./predict_union_160_0728.sh &
sh ./predict_union_160_aitagging_mix.sh &
sh ./predict_union_160_maestro-v3.0.0_mix.sh &
wait
echo 'done'


