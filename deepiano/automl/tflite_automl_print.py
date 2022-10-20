"""
图形显示分析结果
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/Users/xyz/results/',
                           'Directory where the predicted midi & labels are')


tf.app.flags.DEFINE_string('ppea_prod_result_csv_file', './ppea_prod_result.csv',
                           'ppea evaluation csv result for prod env')

tf.app.flags.DEFINE_string('ppea_predict_result_csv_file', './ppea_predict_result.csv',
                           'predicted evaluation csv result')


tf.app.flags.DEFINE_enum(
    'mode', 'ppea_self', ['normal', 'ppea', 'ppea_self'],
    'which mode to convert wav file')


input_dirs = [
  # 'ai_tagging_test_results_0909/analyse/ppea/for-exported_AI_tagging_for_test-dataset-server-maestro-v1-and-flat-midi-std-dataset-2020-09-09-hop-length-256',
'ai_tagging_test_results_0909/analyse/ppea/for-exported_AI_tagging_for_test-dataset-server-maestro-v1-and-flat-midi-std-dataset-2020-09-09-hop-length-256-split_spec-3750',
  # 'ai_tagging_test_results_0909/analyse/ppea/for-exported_AI_tagging_for_test-dataset-server-maestro-v1-and-flat-midi-std-dataset-2020-09-09-hop-length-512',

  # 'ai_tagging_test_results_0908_client_predict/analyse/predict/predict_for_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-16frames-4paddings-onset_threshold-0.3-num_steps-140000',
  # 'ai_tagging_test_results_0908_client_predict/analyse/predict/predict_for_additional_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-16frames-4paddings-onset_threshold-0.3-num_steps-140000',
  # 'ai_tagging_test_results_0908_client_predict/analyse/predict/predict_for_maestro-v2.0.0-16k-no-pitch-shift-test-dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-16frames-4paddings-onset_threshold-0.3-num_steps-140000',
  #
  # 'ai_tagging_test_results_0908_client_predict/analyse/ppea/for_maestro-v2.0.0-16k-no-pitch-shift-test-dataset-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-300000',

  # 'ai_tagging_test_results_0608/ppea_analyse/for_AI_tagging_for_test_dataset_2020_0608_theone_model_devocal',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_AI_tagging_for_test_dataset_2020_0608_server-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-16-model.ckpt-50000',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_AI_tagging_for_test_dataset_2020_0608_server-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-16-model.ckpt-65600',
  #
  # 'ai_tagging_test_results_0608/ppea_analyse/for_maestro-v1.0.0-16000-test-dataset-theone_model_devocal',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_maestro-v1.0.0-16000-test-dataset-2020-06-18-model.ckpt-50000',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_maestro-v1.0.0-16000-test-dataset-2020-06-18-model.ckpt-65600',
  # #
  # 'ai_tagging_test_results_0608/ppea_analyse/for_maestro-v2.0.0-16k-noised-test-dataset-2020_0608_theone_model_devocal',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_maestro-v2.0.0-16k-noised-test-dataset-2020-06-17-model.ckpt-50000',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_maestro-v2.0.0-16k-noised-test-dataset-2020-06-17-model.ckpt-65600',

  # 'ai_tagging_test_results_0608/ppea_analyse/for_AI_tagging_for_additional_AI_tagging_dataset_2020_0622_theone_model_devocal',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_AI_tagging_for_additional_AI_tagging_dataset_2020_0622_theone_model_devocal-snd-pre-server',
  # 'ai_tagging_test_results_0608/ppea_analyse/for_AI_tagging_for_additional_AI_tagging_dataset_2020_0622_server-maestro-2020-06-16-model.ckpt-50000',


  # 'ai_tagging_test_results_0628/analyse/ppea/for_piano_ai_tagging_test_dataset_server-snd-pre',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_piano_ai_tagging_test_dataset_server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-200000',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_piano_ai_tagging_test_dataset_server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-229700',
  'ai_tagging_test_results_0628/analyse/ppea/for_piano_ai_tagging_test_dataset_server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-300000',


  # 'ai_tagging_test_results_0628/analyse/ppea/for_additional_AI_tagging_dataset_server-snd-pre',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_additional_AI_tagging_dataset_server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-200000',
  #  'ai_tagging_test_results_0628/analyse/ppea/for_additional_AI_tagging_dataset_server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-229700',
  #  'ai_tagging_test_results_0628/analyse/ppea/for_additional_AI_tagging_dataset_server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-300000',


  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v1.0.0-16k-server-snd-pre',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v1.0.0-16k-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-200000',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v1.0.0-16k-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-229700',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v1.0.0-16000-test-dataset-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-300000',

  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v2.0.0-16k-noised-2020-06-09-old-test-dataset-server-snd-pre',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v2.0.0-16k-noised-2020-06-09-old-test-dataset-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-171600',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v2.0.0-16k-noised-2020-06-09-old-test-dataset-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-229700',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v2.0.0-16k-noised-2020-06-09-old-test-dataset-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-251900',
  # 'ai_tagging_test_results_0628/analyse/ppea/for_maestro-v2.0.0-16k-noised-2020-06-09-old-test-dataset-server-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-25-model.ckpt-300000',


  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_theone_model_lstm_denoise_20200219',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-80000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-85000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-90000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-100000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-105000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-110000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-115000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-120000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-125000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-130000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-135000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-140000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-145000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-150000',
  #
  # 'ai_tagging_test_results_0628/analyse/predict/for_additional_AI_tagging_dataset_theone_model_lstm_denoise_20200219',
  # 'ai_tagging_test_results_0628/analyse/predict/for_additional_AI_tagging_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-model.ckpt-120000'

  # 'ai_tagging_test_results_0628/analyse/infer/for_v2.0.0_test_tfrecord_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-90000',
  # 'ai_tagging_test_results_0628/analyse/infer/for_v2.0.0_test_tfrecord_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-120000',
  # 'ai_tagging_test_results_0628/analyse/infer/for_v2.0.0_test_tfrecord_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-150000',

  # 'ai_tagging_test_results_0608/predict_analyse/AI_tagging_for_test_dataset_2020_0608_theone_model_lstm_denoise_20200219',
  # 'ai_tagging_test_results_0608/predict_analyse/for_AI_tagging_for_test_dataset_denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-10-model.ckpt-51300',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-60000',
  # 'ai_tagging_test_results_0628/analyse/predict/for_AI_tagging_for_test_dataset_tflite-lstm-full-test-no-cudnn-maestro-v1-v2-ai-tagging-flat-midi-dataset-2020-06-28-num_steps-65000',
  # 'ai_tagging_test_results_0619/analyse/predict/for_AI_tagging_for_test_dataset_theone_model_lstm_denoise_low_20200225',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-20000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-21000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-22000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-23000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-24000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-25000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-26000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-27000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-28000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-29000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-30000',
  #
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-33000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-34000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-35000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-36000',
  #
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-37000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-38000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-39000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-40000',
  #
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-41000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-42000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-43000',
  # 'ai_tagging_test_results_0619/analyse/predict/predict_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-44000',


  # 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-20000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-21000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-22000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-23000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-24000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-25000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-26000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-27000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-28000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-29000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-41000',
# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-42000',

# 'ai_tagging_test_results_0619/analyse/infer/infer_for_AI_tagging_for_test_dataset_tflite_half-denoice-with-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-15-num_steps-44000',

]

def get_analyse_result_csv(input_dirs):
  analyse_files = []
  logging.info('generate_analyse_set %s' % input_dirs)
  for directory in input_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    logging.info('get_analyse_result_csv! path: %s' % path)
    path = os.path.join(path, '*analyse_results.csv')
    analyse_files.append(glob.glob(path))
  return analyse_files


def get_label(analyse_file, data):
  matched = re.match('.*/(.*)_analyse_results.csv', analyse_file)
  if matched:
    name = matched.group(1)[40:] + ': ' + str(data)
    return name
  return 'Unknown: 0'


def show_ppea_self_csv_results():
  ppea_prod_csv_files = [FLAGS.ppea_predict_result_csv_file]
  for file_path in ppea_prod_csv_files:
    results = pd.read_csv(file_path)
    plt.subplot(3, 1, 1)
    plt.xlabel('Eval no.')
    plt.ylabel('simple_pitch')
    plt.title('simple_pitch')

    name_info = 'client predict' # os.path.splitext(file_path)[0]
    plt.plot(results['simple_pitch'], label=name_info)
    plt.legend(loc='lower left')

    plt.subplot(3, 1, 2)
    plt.ylabel('simple_final')
    plt.xlabel('Eval no.')
    plt.title('simple_final')
    name_info = 'client predict' # os.path.splitext(file_path)[0]
    plt.plot(results['simple_final'], label=name_info)
    plt.legend(loc='lower left')

    plt.subplot(3, 1, 3)
    plt.ylabel('simple_complete')
    plt.xlabel('Eval no.')
    plt.title('simple_complete')
    name_info = 'client predict' #os.path.splitext(file_path)[0]
    plt.plot(results['simple_complete'], label=name_info)
    plt.legend(loc='lower left')

  ppea_predict_csv_files = [FLAGS.ppea_predict_result_csv_file]
  for file_path in ppea_predict_csv_files:
    results = pd.read_csv(file_path)
    plt.subplot(3, 1, 1)
    plt.xlabel('Eval no.')
    plt.ylabel('simple_pitch')
    plt.title('simple_pitch')

    name_info = "server predict" #os.path.splitext(file_path)[0]
    plt.plot(results['prod_simple_pitch'], label=name_info)
    plt.legend(loc='lower left')

    plt.subplot(3, 1, 2)
    plt.ylabel('simple_final')
    plt.xlabel('Eval no.')
    plt.title('simple_final')
    name_info = "server predict" # os.path.splitext(file_path)[0]
    plt.plot(results['prod_simple_final'], label=name_info)
    plt.legend(loc='lower left')

    plt.subplot(3, 1, 3)
    plt.ylabel('simple_complete')
    plt.xlabel('Eval no.')
    plt.title('simple_complete')
    name_info = 'server predict' #os.path.splitext(file_path)[0]
    plt.plot(results['prod_simple_complete'], label=name_info)
    plt.legend(loc='lower left')

  plt.show()

def show_ppea_csv_results():
  ppea_csv_files = [FLAGS.ppea_prod_result_csv_file, FLAGS.ppea_predict_result_csv_file]
  for file_path in ppea_csv_files:
    results = pd.read_csv(file_path)
    plt.subplot(3, 1, 1)
    plt.xlabel('Eval no.')
    plt.ylabel('simple_pitch')
    plt.title('simple_pitch')

    name_info = os.path.splitext(file_path)[0]
    plt.plot(results['simple_pitch'], label=name_info)
    plt.legend(loc='lower left')

    plt.subplot(3, 1, 2)
    plt.ylabel('simple_final')
    plt.xlabel('Eval no.')
    plt.title('simple_final')
    name_info = os.path.splitext(file_path)[0]
    plt.plot(results['simple_final'], label=name_info)
    plt.legend(loc='lower left')

    plt.subplot(3, 1, 3)
    plt.ylabel('simple_complete')
    plt.xlabel('Eval no.')
    plt.title('simple_complete')
    name_info = os.path.splitext(file_path)[0]
    plt.plot(results['simple_complete'], label=name_info)
    plt.legend(loc='lower left')
  plt.show()


def main(argv):
  del argv

  if FLAGS.mode == 'normal':
    analyse_files = get_analyse_result_csv(input_dirs)

    plt.figure()

    for file in analyse_files:
      results = pd.read_csv(file[0])
      plt.subplot(3, 1, 1)
      plt.xlabel('Eval no.')
      plt.ylabel('Recall')
      plt.title('Recall')
      low_num = results['recall'].where(results['recall'] < 0.9).count()
      info = str(results['recall'].mean().round(3)) + ', low: ' + str(low_num)
      plt.plot(results['recall'], label=get_label(file[0], info))
      plt.legend(loc='lower left')

      plt.subplot(3, 1, 2)
      plt.ylabel('Precision')
      plt.xlabel('File no.')
      plt.title('Precision')
      info = str(results['precision'].mean().round(3))
      plt.plot(results['precision'], label=get_label(file[0], info))
      plt.legend(loc='lower left')

      # plt.subplot(3, 1, 3)
      # plt.ylabel('Acc')
      # plt.xlabel('File no.')
      # plt.title('Acc')
      # info = str(results['acc'].mean().round(3))
      # plt.plot(results['acc'], label=get_label(file[0], info))
      # plt.legend(loc='bottom left')

    plt.show()
  elif FLAGS.mode == 'ppea':
    show_ppea_csv_results()
  else:
    show_ppea_self_csv_results()



if __name__ == '__main__':
  tf.app.run(main)
