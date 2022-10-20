#!/bin/sh

PYTHON=python

CHUNK_PADDING=3
CLIENT_THRESHOLD=0.01
DATASET_DIR=/deepiano_data/dataset
MIDI_DATASET_DIR=/deepiano_data/dataset

#需要修改的地方
#PREDICT_TIME=819
#TFLITE_MODEL_ID=forceconcat_20220819_8_default
#TFLITE_MODEL_PATH=/data/projects/test_models

TFLITE_MODEL_DIR=${TFLITE_MODEL_PATH}/${TFLITE_MODEL_ID}.tflite
OUT_PREDICT_DIR=/data/zhaoliang/out/${PREDICT_TIME}/predict/union_delay_40/${TFLITE_MODEL_ID}
OUT_PREDICT_CLIENT_DIR=/data/zhaoliang/out/${PREDICT_TIME}/predict-client/union_delay_40/${TFLITE_MODEL_ID}
OUT_ANALYSER_DIR=/data/zhaoliang/out/${PREDICT_TIME}/analyser/union_delay_40/${TFLITE_MODEL_ID}



echo "bgm_specail"
${PYTHON} tflite_union_predict.py \
--chunk_padding ${CHUNK_PADDING} \
--model_path ${TFLITE_MODEL_DIR}  \
--client_threshold ${CLIENT_THRESHOLD}  \
--input_dirs ${DATASET_DIR}/bgm_specail   \
--input_mid_dirs ${MIDI_DATASET_DIR}/bgm_specail  \
--data_type test \
--output_dir ${OUT_PREDICT_DIR}/bgm_specail \
--output_client_dir ${OUT_PREDICT_CLIENT_DIR}/bgm_specail \
--output_json_dir ${OUT_PREDICT_CLIENT_DIR}/json-bgm_specail  &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.0001 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.0001 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.001 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.001 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.1 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.1 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.2 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.2 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.3 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.3 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.4 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.4 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.5 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.5 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.6 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.6 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.7 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.7 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.8 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.8 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.9 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.9 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.01 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.01 &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/bgm_specail_0.05 \
--output_dir ${OUT_ANALYSER_DIR}/bgm_specail_0.05 &&
wait &&

echo "done"
