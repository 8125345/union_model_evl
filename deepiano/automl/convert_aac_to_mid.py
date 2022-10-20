
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import requests
import os
import glob
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', './test',
                           'Directory where wav & labels are')
tf.app.flags.DEFINE_string('output_dir', './test',
                           'Directory where the analyser results will be placed.')


def convert(audio, midi):
  url = 'http://snd-pre.research.xiaoyezi.com/pt/1.0/wav2mid?test=1'
  # url = 'http://39.104.28.114:8086/pt/1.0/wav2mid'
  resp = requests.post(url, data=open(audio, 'rb'), verify=False)
  print(resp)
  result = resp.json()
  midi_data = base64.decodebytes(result['midi'].encode())
  print(result['preds']['onsets'])
  if len(midi_data) > 0:
    open(midi, 'wb').write(midi_data)


def convert_wav_to_midi(wav_file):
  wav_base_name = os.path.basename(wav_file)
  base_name_no_suffix = os.path.splitext(wav_base_name)[0]
  target_predict_midi_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.predicted.midi')
  convert(wav_file, target_predict_midi_path)


def generate_convert_set():
  path = os.path.join(FLAGS.input_dir, '*.aac')
  wav_files = glob.glob(path)
  return wav_files


def main(argv):
  del argv

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  wav_files = generate_convert_set()
  for wav_file in wav_files:
    convert_wav_to_midi(wav_file)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
