"""
根据自标注的mp3文件下载对应的wav文件，提高该数据集可用性。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import requests
import os

import re

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/Users/xyz/src/ai/piano_ai_tagging_3rd/',
                           'Directory where wav & labels are')
tf.app.flags.DEFINE_string('output_dir', '/Users/xyz/src/ai/piano_ai_tagging_3rd/',
                           'Directory where the analyser results will be placed.')

all_mp3_dir = []

AI_BACKEND_PROD_SND_EVAL_RECORD_URL = 'https://musvg.xiaoyezi.com/api/1.0/eval/record'


def download_snd_file(audio_url, target_wav_file):
  r = requests.get(audio_url)

  with open(target_wav_file, mode='wb') as f:
    f.write(r.content)
    f.flush()


def load_eval_record(base_url, eval_id):
  headers = {
    'TOKEN': 'MAGISTER_USER_1',
    'SERVICEID': '3'
  }

  url = base_url + '/' + str(eval_id)

  response = requests.get(url, headers=headers)
  if response.status_code == 200:
    resp = response.json()
    print(resp)
    return resp
  else:
    print('load_eval_record: error[{code}]'.format(code=response.status_code))
    return None


def get_eval_id(mp3_file):
  matchObj = re.match(r'.*-(.*?).mp3', mp3_file, flags=0)
  if matchObj:
    return matchObj.group(1)
  return None


def iter_dirs(rootDir):
  pattern = r'.*/\..*'
  for root, dirs, files in os.walk(rootDir):
    match = re.match(pattern, root)
    if match:
      continue
    if dirs != []:
      for dirname in dirs:
        if not dirname.startswith('.'):
          full_dirname = os.path.join(root, dirname)
          all_mp3_dir.append((dirname, full_dirname))
          iter_dirs(full_dirname)


def convert_mp3_to_wav(all_mp3_dir):
  for d_name, full_dirname in all_mp3_dir:
    dest_dir = os.path.join(FLAGS.output_dir, d_name)
    if not os.path.isdir(dest_dir):
      os.makedirs(dest_dir)

    mp3_dir = full_dirname
    mp3_files = glob.glob(os.path.join(mp3_dir, '*.mp3'))

    for mp3_file in mp3_files:
      eval_id = get_eval_id(mp3_file)
      if not eval_id:
        print('Bad mp3 file name! [%s]' % mp3_file)
        return

      resp = load_eval_record(AI_BACKEND_PROD_SND_EVAL_RECORD_URL, eval_id)
      if not resp:
        print('Empty eval record!')
        return
      audio_url = resp['data'].get('audio_url')

      mp3_base_name = os.path.basename(mp3_file)
      base_name_no_suffix = os.path.splitext(mp3_base_name)[0]
      target_wav_file = os.path.join(dest_dir, base_name_no_suffix + '.aac')

      download_snd_file(audio_url, target_wav_file)


def main(argv):
  del argv

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  iter_dirs(FLAGS.input_dir)

  convert_mp3_to_wav(all_mp3_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
