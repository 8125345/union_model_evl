"""
修正自标注数据的midi文件
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import random
import re
import subprocess
from functools import lru_cache
from shutil import copyfile

import requests
import os
import glob
from os.path import join, dirname, isfile
from subprocess import Popen

import mido
import sox

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/Users/xyz/user_special_eval_audio_fixed',
                           'Directory where the fixed wav dataset will be placed.')
tf.app.flags.DEFINE_string('output_dir', '/Users/xyz/piano_dataset',
                           'Directory where the target score will be replaced.')

all_wav_dirs = []

data_set_dirs = [
  'exported_AI_tagging_for_test',
  'exported_AI_tagging_for_train'
]

additional_data_set_dirs = [
  'additional_AI_tagging_dataset'
]


def generate_fixed_midi_list(input_dirs):
  fixed_midi_file_pairs = []
  fixed_2_midi_file_pairs = []
  fixed_3_midi_file_pairs = []

  all_files = []
  for directory in list(set(input_dirs)):
    # path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(directory, '*.fix.MID')
    fixed_midi_files = glob.glob(path)

    # find matching mid files
    for midi_file in fixed_midi_files:
      if not os.path.isfile(midi_file):
        print('empty midi_file:' + midi_file)
        continue

      filepath, file_name = os.path.split(midi_file)
      if file_name in all_files:
        print('dupfile: ' + filepath + '/' + file_name)
      all_files.append(file_name)
      base_name, _ = os.path.splitext(midi_file)
      wav_file = midi_file.replace('.fix.MID', '')
      print(midi_file)
      if os.path.isfile(midi_file) and os.path.isfile(wav_file):
        fixed_midi_file_pairs.append((file_name, midi_file, wav_file))
      else:
        print('empty midi:' + base_name)

    path_2 = os.path.join(directory, '*.fix(2).MID')
    fixed_2_midi_files = glob.glob(path_2)
    # find matching mid files
    for midi_file in fixed_2_midi_files:
      if not os.path.isfile(midi_file):
        print('empty midi_file:' + midi_file)
        continue

      filepath, file_name = os.path.split(midi_file)
      if file_name in all_files:
        print('dupfile: ' + filepath + '/' + file_name)
      all_files.append(file_name)
      base_name, _ = os.path.splitext(midi_file)
      wav_file = midi_file.replace('.fix(2).MID', '')
      print(midi_file)
      if os.path.isfile(midi_file):
        fixed_2_midi_file_pairs.append((file_name, midi_file, wav_file))
      else:
        print('empty midi:' + base_name)

    path_3 = os.path.join(directory, '*.fix（2）.MID')
    fixed_3_midi_files = glob.glob(path_3)
    # find matching mid files
    for midi_file in fixed_3_midi_files:
      if not os.path.isfile(midi_file):
        print('empty midi_file:' + midi_file)
        continue

      filepath, file_name = os.path.split(midi_file)
      if file_name in all_files:
        print('dupfile: ' + filepath + '/' + file_name)
      all_files.append(file_name)
      base_name, _ = os.path.splitext(midi_file)
      wav_file = midi_file.replace('.fix（2）.MID', '')
      print(midi_file)
      if os.path.isfile(midi_file):
        fixed_3_midi_file_pairs.append((file_name, midi_file, wav_file))
      else:
        print('empty midi:' + base_name)

  print(len(list(set(all_files))))
  return list(set(fixed_midi_file_pairs)), list(set(fixed_2_midi_file_pairs)), list(set(fixed_3_midi_file_pairs))


def iter_dirs(rootDir):
  for root, dirs, files in os.walk(rootDir):
    if dirs != []:
      for dirname in dirs:
        full_dirname = os.path.join(root, dirname)
        all_wav_dirs.append(full_dirname)
        iter_dirs(full_dirname)


def get_wav_name(file_full_path):
  file_name = ''
  if os.path.splitext(file_full_path)[1] == '.wav':
    (filepath, tempfilename) = os.path.split(file_full_path)
    file_name = os.path.splitext(tempfilename)[0]
  return file_name


def get_last_path_name(file_full_path):
  last_file_path = ''
  if os.path.splitext(file_full_path)[1] == '.wav':
    (filepath, tempfilename) = os.path.split(file_full_path)
    last_file_path = filepath.split('/')[-1]
  return last_file_path


def fix_score_midi(fixed_midi_file):
  fixed_count = 0
  for midi_dir in data_set_dirs:
    path = os.path.join(FLAGS.output_dir, midi_dir, '*.mid')
    target_midi_files = glob.glob(path)
    for midi_file in target_midi_files:
      midi_filepath, midi_file_name = os.path.split(midi_file)
      file_name_without_suffix = midi_file_name.replace('.mid', '')
      if fixed_midi_file.find(file_name_without_suffix) > 0:
        print('fix_score_midi: %s -> %s' % (fixed_midi_file, midi_file))
        copyfile(fixed_midi_file, midi_file)
        fixed_count = fixed_count + 1

  return fixed_count


def copy_additional_midi_data(fixed_midi_file_pairs):
  count = 0
  found_count = 0
  for _, fixed_midi_file, wav_file in fixed_midi_file_pairs:
    found = False
    for midi_dir in data_set_dirs:
      if found:
        break
      path = os.path.join(FLAGS.output_dir, midi_dir, '*.mid')
      target_midi_files = glob.glob(path)
      for midi_file in target_midi_files:
        midi_filepath, midi_file_name = os.path.split(midi_file)
        file_name_without_suffix = midi_file_name.replace('.mid', '')
        if fixed_midi_file.find(file_name_without_suffix) > 0:
          found = True
          break
    if not found:
      _, dest_wav_file_name = os.path.split(wav_file)
      dest_wav_path = os.path.join(FLAGS.output_dir, additional_data_set_dirs[0], dest_wav_file_name)
      dest_mid_path = os.path.join(FLAGS.output_dir, additional_data_set_dirs[0], dest_wav_file_name.replace('.wav',
                                                                                                             '.mid'))
      print('copy_additional[wav]: %s -> %s' % (wav_file, dest_wav_path))
      copyfile(wav_file, dest_wav_path)

      print('copy_additional[mid]: %s -> %s' % (fixed_midi_file, dest_mid_path))
      copyfile(fixed_midi_file, dest_mid_path)
      count = count + 1
    else:
      found_count = found_count + 1
      # print('found: %s found_count:%d' % (wav_file, found_count))

  return count


def main(argv):
  del argv

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  iter_dirs(FLAGS.input_dir)
  fixed_midi_file_pairs, fixed_2_midi_file_pairs, fixed_3_midi_file_pairs = generate_fixed_midi_list(all_wav_dirs)

  fixed_count = 0
  for _, midi_file, wav_file in fixed_midi_file_pairs:
    fixed_count = fixed_count + fix_score_midi(midi_file)
  print('first fixed count: %d, source count: %d' % (fixed_count, len(fixed_midi_file_pairs)))

  fixed_2_count = 0
  for _, midi_file, wav_file in fixed_2_midi_file_pairs:
    fixed_2_count = fixed_2_count + fix_score_midi(midi_file)
  print('2nd fixed count: %d, source count: %d' % (fixed_2_count, len(fixed_2_midi_file_pairs)))

  fixed_3_count = 0
  for _, midi_file, wav_file in fixed_3_midi_file_pairs:
    fixed_3_count = fixed_3_count + fix_score_midi(midi_file)

  print('3rd fixed count: %d, source count: %d' % (fixed_3_count, len(fixed_3_midi_file_pairs)))

  total_fixed_count = fixed_count + fixed_2_count + fixed_3_count
  total_source_count = len(fixed_midi_file_pairs) + len(fixed_2_midi_file_pairs) + len(fixed_3_midi_file_pairs)
  print('total_fixed_count: %d, total_source_count: %d' % (total_fixed_count, total_source_count))

  count = copy_additional_midi_data(fixed_midi_file_pairs)
  count2 = copy_additional_midi_data(fixed_2_midi_file_pairs)
  count3 = copy_additional_midi_data(fixed_3_midi_file_pairs)
  print('copied: %d, %d, %d, total: %d souce: %d' % (count,count2, count3, (total_fixed_count + count + count2 + count3), total_source_count))



def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
