"""
将背景噪音加入wav文件中
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
tf.app.flags.DEFINE_string('input_dir', '/data/maestro/maestro-v2.0.0',
                           'Directory where the raw train related wav dataset will be placed.')
tf.app.flags.DEFINE_string('output_dir', '/data/maestro/maestro-v2.0.0-16k-noised',
                           'Directory where the noised train dataset will be placed.')

tf.app.flags.DEFINE_string('noise_dir', '/data/noise/vocal/',
                           'Directory where the noised background wav files will be placed.')

all_wav_dirs = []


def generate_data_set(input_dirs):
  wav_file_pairs = []
  all_files = []
  all_eval_ids = []
  for directory in list(set(input_dirs)):
    # path = os.path.join(FLAGS.input_dir, directory)
    path = directory
    path = os.path.join(path, '*_16k.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      if not os.path.isfile(wav_file):
        print('empty wav_file:' + wav_file)
        continue

      filepath, file_name = os.path.split(wav_file)
      if file_name in all_files:
        print('dupfile: ' + filepath + '/' + file_name)
      all_files.append(file_name)
      base_name, _ = os.path.splitext(wav_file)
      mid_file = (base_name + '.midi').replace('_16k', '')
      #mid_file = (base_name).replace('_16k', '')
      print(mid_file)
      if os.path.isfile(mid_file):
        wav_file_pairs.append((wav_file, mid_file))
      else:
        print('empty midi:' + base_name)
  print(len(list(set(all_files))))
  return list(set(wav_file_pairs))


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


def get_audio_duration(fn):
  duration = subprocess.run(['soxi', '-D', fn], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
  return float(duration)


@lru_cache(32)
def read_noise_files(noise_dir):
  result = {}
  for fn in glob.glob(os.path.join(noise_dir, '*.wav')):
    duration = get_audio_duration(fn)
    result[fn] = duration

  return list(result.items())


def add_noise(input_filename, output_filename, noise_dir, noise_vol_range=(-40, -5)):
  noise_file_info = read_noise_files(noise_dir)
  if not noise_file_info:
    print('no noise files, skip')
    return
  print(noise_file_info)
  noise_file, noise_duration = random.choice(noise_file_info)
  noise_vol = random.uniform(*noise_vol_range)
  input_duration = get_audio_duration(input_filename)
  start = random.uniform(0, noise_duration - input_duration)

  command = 'sox {noise_file} -p trim {start} {input_duration} fade q 0.05 {input_duration} 0.05 gain {noise_vol} | sox -m "{input_filename}" - "{output_filename}"'.format(
    **{
      'input_filename': input_filename,
      'output_filename': output_filename,
      'noise_file': noise_file,
      'noise_vol': noise_vol,
      'input_duration': input_duration,
      'start': start,
    })

  print(command)

  process_handle = Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  stdout, stderr = process_handle.communicate()
  errors = stderr.decode().split('\n')
  for line in errors:
    print(line)
  if process_handle.returncode == 0:
    print('{} OK!'.format(output_filename))
  else:
    print('{} ERROR!'.format(output_filename))


def add_noise_to_wav(wav_file, label_midi_file):
  base_file_name = get_wav_name(wav_file)
  last_path_name = get_last_path_name(wav_file)

  target_dir = os.path.join(FLAGS.output_dir, last_path_name)
  if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

  target_wav_path = os.path.join(FLAGS.output_dir, last_path_name, base_file_name + '.wav')
  target_midi_path = os.path.join(FLAGS.output_dir, last_path_name, base_file_name + '.midi')
  if os.path.isfile(target_wav_path) and os.path.isfile(target_midi_path):
    return

  with open(target_wav_path, 'wb') as f:
    add_noise(wav_file, f.name, FLAGS.noise_dir)
  copyfile(label_midi_file, target_midi_path)


def main(argv):
  del argv

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  iter_dirs(FLAGS.input_dir)
  wav_pairs = generate_data_set(all_wav_dirs)
  for wav_file, label_midi_file in wav_pairs:
    add_noise_to_wav(wav_file, label_midi_file)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
