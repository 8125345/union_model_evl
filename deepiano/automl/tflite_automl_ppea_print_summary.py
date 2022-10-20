"""
图形对比各个训练模型综合recall/precision结果
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import glob
import logging
import collections

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/Users/xyz/results/ai_tagging_test_results_0901/analyse',
                           'Directory where the predicted results are')


tf.app.flags.DEFINE_enum(
    'mode', 'ppea', ['predict', 'infer', 'ppea'],
    'which mode to display')

tf.app.flags.DEFINE_enum(
    'category', 'v1', ['ai_tagging', 'additional_AI_tagging', 'v1', 'v2', 'ai_tagging_train'],
    'which analyse data used to display')


predict_analyse_dir = 'predict/'


infer_analyse_dir = 'infer/'


ppea_analyse_dir = 'ppea/'


predict_test_data_set_analyse_patterns = {
  'ai_tagging': [
    {
      'v1+v2+自标注-(阈值-0.3)':
        'predict_for_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-and-ai-tagging-dataset-2020-08-06-onset_threshold-0.3-num_steps',
    },

    {
      'v1+v2+自标注-(阈值-0.5)':
        'predict_for_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-and-ai-tagging-dataset-2020-08-06-onset_threshold-0.5-num_steps',
    },

    {
      'v1+v2-(阈值-0.3)':
        'predict_for_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-dataset-2020-07-26-onset_threshold-0.3-num_steps',
    },

    {
      'v1+v2-(阈值-0.5)':
        'predict_for_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-dataset-2020-07-26-onset_threshold-0.5-num_steps',
    },

    {
      'v1+v2+flat-midi-(阈值-0.3)':
        'predict_for_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-v1-v2-and-flat-midi-dataset-2020-07-26-onset_threshold-0.3-num_steps',
    },

    {
      'v1+v2+flat-midi-(阈值-0.5)':
        'predict_for_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-v1-v2-and-flat-midi-dataset-2020-07-26-onset_threshold-0.5-num_steps',
    },
  ],
  'additional_AI_tagging': [
    {
      'v1+v2+自标注-(阈值-0.3)':
        'predict_for_additional_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-and-ai-tagging-dataset-2020-08-06-onset_threshold-0.3-num_steps',
    },

    # {
    #   'v1+v2+自标注-(阈值-0.5)':
    #     'predict_for_additionalAI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-and-ai-tagging-dataset-2020-08-06-onset_threshold-0.5-num_steps',
    # },

    {
      'v1+v2-(阈值-0.3)':
        'predict_for_additional_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-dataset-2020-07-26-onset_threshold-0.3-num_steps',
    },
    # {
    #   'v1+v2-(阈值-0.5)':
    #     'predict_for_additionalAI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-only-v1-v2-dataset-2020-07-26-onset_threshold-0.5-num_steps',
    #
    # },

    {
      'v1+v2+flat-midi-(阈值-0.3)':
        'predict_for_additional_AI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-v1-v2-and-flat-midi-dataset-2020-07-26-onset_threshold-0.3-num_steps',
    },

    # {
    #   'v1+v2+flat-midi-(阈值-0.5)':
    #     'predict_for_additionalAI_tagging_for_piano_test_dataset_tflite-lstm-full-test-maestro-v1-v2-and-flat-midi-dataset-2020-07-26-onset_threshold-0.5-num_steps',
    # },

  ]
}


infer_test_data_set_analyse_patterns = {
  'ai_tagging': [
  {
    '原始（自标注数据集）':
      'AI_tagging_for_test-dataset-server-maestro-v1-and-flat-midi-2020-08-21-for-model.ckpt'
  },
  {
    '继续训练（自标注数据集）':
      'AI_tagging_for_test-dataset-server-maestro-v1-and-flat-midi-2020-08-21-model.ckpt-55500-ai-tagging-for-model.ckpt'
  }
],

  'additional_AI_tagging': [
    {
      '原始（线上有问题case）':
        'additional_AI_tagging_dataset-server-maestro-v1-and-flat-midi-2020-08-21-for-model.ckpt',
    },
    {
      '继续训练（线上有问题case）':
        'additional_AI_tagging_dataset-server-maestro-v1-and-flat-midi-2020-08-21-model.ckpt-55500-ai-tagging-for-model.ckpt'
    },
  ],

  'v1':
  [
    {
      '原始（v1测试集）':
        'maestro-v1.0.0-16000-dataset-server-maestro-v1-and-flat-midi-2020-08-21-for-model.ckpt',
    },
    {
      '继续训练（v1测试集）':
        'maestro-v1.0.0-16000-dataset-server-maestro-v1-and-flat-midi-2020-08-21-model.ckpt-55500-ai-tagging-for-model.ckpt'
    },
  ]
}

ppea_test_data_set_analyse_patterns = {
  'v1': [
    {
      'v1测试集(无自标注训练)':
        'maestro-v1.0.0-16000-dataset-server-maestro-v1-and-flat-midi-std-dataset-2020-09-01-num_steps',
    },
    {
      'v1测试集(有自标注训练)':
        'maestro-v1.0.0-16000-dataset-server-maestro-v1-v2-and-flat-midi-std-and-ai-tagging-dataset-2020-09-02-num_steps',
    },
  ],
  'ai_tagging': [
    {
      '自标注测试集(无自标注训练)':
      'AI_tagging_for_test-dataset-server-maestro-v1-and-flat-midi-std-dataset-2020-09-01-num_steps'
    },
    {
      '自标注测试集(有自标注训练)':
        'AI_tagging_for_test-dataset-server-maestro-v1-v2-and-flat-midi-std-and-ai-tagging-dataset-2020-09-02-num_steps',
    },
  ],

  'additional_AI_tagging': [
    {
      '线上有问题case(无自标注训练)':
        'additional_AI_tagging_dataset-server-maestro-v1-and-flat-midi-std-dataset-2020-09-01-num_steps',
    },
    {
      '线上有问题case(有自标注训练)':
        'additional_AI_tagging_dataset-server-maestro-v1-v2-and-flat-midi-std-and-ai-tagging-dataset-2020-09-02-num_steps',
    },
  ]
}


def get_analyse_result_csv_files(analyse_dir):
  logging.info('get_analyse_result_csv_files %s' % analyse_dir)
  path = os.path.join(FLAGS.input_dir, analyse_dir + '*')
  logging.info('get_analyse_result_csv! path: %s' % path)
  path = os.path.join(path, '*analyse_results.csv')

  return glob.glob(path)


def get_label(analyse_file, data):
  matched = re.match('.*/(.*)_analyse_results.csv', analyse_file)
  if matched:
    return matched.group(1) + ': ' + str(data)
  return 'Unknown: 0'


def get_step_num(analyse_file):
  matched = re.match('.*-([0-9]+)_analyse_results.csv', analyse_file)
  if matched:
    return int(matched.group(1))
  return 0


def get_csv_contents_from_results(files):
  all_info = []
  for f in files:
    result = pd.read_csv(f)
    v = {
      'step_num': get_step_num(f),
      'recall': result['recall'].mean().round(3),
      'precision': result['precision'].mean().round(3),
      'low_num': result['recall'].where(result['recall'] < 0.9).count()
    }
    all_info.append(v)
    all_info.sort(key=lambda x: x.get('step_num'))

  # convert to the format which will be easy to display
  resp = {
    'step_num': [],
    'recall': [],
    'precision': [],
    'low_num': []
  }
  for info in all_info:
    resp['step_num'].append(info.get('step_num'))
    resp['recall'].append(info.get('recall'))
    resp['precision'].append(info.get('precision'))
    resp['low_num'].append(info.get('low_num'))

  return resp


def display_category_analyse_results(category, dataset_patterns, analyse_files):
  """
  先将模型预测数据分组
  'category1': {
    'name1': [f1, f2, f3],
    'name2': [f4, f5, f6],
  },
  'category2': {
    'name1': [f7, f8, f9],
    'name2': [f10, f11, f12],
  },
  ...
  """
  analyse_group_results = collections.OrderedDict()

  model_info = {}

  #将模型预测结果进行分组
  for pattern in dataset_patterns:
    for name, v in pattern.items():
      format = r'.+{}-([0-9]+)/.+'.format(v)

      files = []
      for file_info in analyse_files:
        matchObj = re.match(format, file_info, re.M | re.I)
        if matchObj:
          print('match: %s name:%s' % (matchObj.group(1), name))
          files.append(file_info)
      model_info[name] = get_csv_contents_from_results(files)
  print('ok!')

  font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')

  # display results
  plt.figure(figsize=(10, 10))
  for name, info in model_info.items():
    plt.subplot(2, 1, 1)
    plt.xlabel('Step num.', fontproperties=font)
    plt.ylabel('Recall', fontproperties=font)
    plt.title('Recall', fontproperties=font)
    plt.plot(info.get('step_num'), info.get('recall'), label=name)
    plt.legend(loc='lower left',  prop=font)

    plt.subplots_adjust(bottom=0.0)

    plt.subplot(2, 1, 2)
    plt.xlabel('Step num.', fontproperties=font)
    plt.ylabel('Precision', fontproperties=font)
    plt.title('Precision', fontproperties=font)
    plt.plot(info.get('step_num'), info.get('precision'), label=name)
    plt.legend(loc='lower left', prop=font)


  plt.show()


def main(argv):
  del argv

  if FLAGS.mode == 'predict':
    analyse_files = get_analyse_result_csv_files(predict_analyse_dir)
    dataset_patterns = predict_test_data_set_analyse_patterns[FLAGS.category]
  elif FLAGS.mode == 'infer':
    analyse_files = get_analyse_result_csv_files(infer_analyse_dir)
    dataset_patterns = infer_test_data_set_analyse_patterns[FLAGS.category]
  elif FLAGS.mode == 'ppea':
    analyse_files = get_analyse_result_csv_files(ppea_analyse_dir)
    dataset_patterns = ppea_test_data_set_analyse_patterns[FLAGS.category]

  display_category_analyse_results(FLAGS.category, dataset_patterns, analyse_files)


if __name__ == '__main__':
  tf.app.run(main)

