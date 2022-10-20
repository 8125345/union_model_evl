"""
客户端调用(/eval/evaluate_pred)接口，发送概率图及midi至服务端进行预测, 结果生成csv报表
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import collections
import math
import time
import os
import base64
import msgpack

import numpy as np
import pandas as pd

import requests
import tensorflow as tf

from deepiano.music import constants as music_constants
from deepiano.music import midi_io
from deepiano.protobuf import music_pb2
from deepiano.server import wav2spec
from deepiano.wav2mid import configs_tflite as configs
from deepiano.wav2mid import constants
from deepiano.wav2mid import data
from deepiano.server import utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string('model_path', '/Users/xyz/tf_model/ai_taggint_test_model/converted_model-full-140000-16frames.tflite',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'hparams',
    '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'frame_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_float(
    'onset_threshold', 0.3,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_string(
    'input', '../../data/test/record.wav',
    'input wav file path')
tf.app.flags.DEFINE_integer(
  'chunk_padding', 5,
  'chunk_padding')


tf.app.flags.DEFINE_string('input_dir', '../../data/AI_tagging_for_test',
                           'Directory where the wav & midi labels are')

tf.app.flags.DEFINE_string('output_dir', '../../data/test/predict_1',
                           'Directory where the predicted midi & midi labels will be placed.')

# dummy flags
tf.app.flags.DEFINE_string('type', 'evaluate_pred',
                           'Type for evaluate_pred')

tf.app.flags.DEFINE_string('name', 'evaluate_pred',
                           'Name for evaluate_pred')

# for csv
tf.app.flags.DEFINE_string('input_csv_path', './ppea_prod_result.csv',
                           'input ppea csv result from prod')
tf.app.flags.DEFINE_string('output_csv_path', './ppea_predict_result.csv',
                           'output ppea predict csv result')

tf.app.flags.DEFINE_enum(
    'mode', 'new', ['new', 'token'],
    'which mode will be used')

PROD_USER_TOKEN = 'af0b16daf0a7c465d22ec3b19a6fb775'
PROD_EVAL_URL = 'https://musicscore-app.xiaoyezi.com/#/reportNew?scoreId={scoreId}&lessonId={lessonId}&recordId={recordId}&platform=2&version=5.5.0&appToken={appToken}'


PRE_USER_TOKEN = '7c85c73f7f6bae302c134b177d5c4c72'
PRE_EVAL_URL = 'https://musicscore-app-pre.xiaoyezi.com/#/reportNew?scoreId={scoreId}&lessonId={lessonId}&recordId={recordId}&platform=2&version=5.5.0&appToken={appToken}'


AI_BACKEND_PRE_SND_EVALUATE_PRED_URL = 'http://aibackend-pre.1tai.com/api/1.0/eval/evaluate_pred'
# AI_BACKEND_DEV_SND_EVALUATE_PRED_URL = 'http://0.0.0.0:20000/api/1.0/eval/evaluate_pred'


ChunkPrediction = collections.namedtuple(
    'ChunkPrediction',
    ('onset_predictions', 'velocity_values', 'onset_probs_flat'))


all_wav_dir = []


def download_snd_file(audio_url):
  r = requests.get(audio_url)
  return r.content


def pianoroll_to_note_sequence(chunk_func_c,
                               frames_per_second,
                               min_duration_ms,
                               velocity=70,
                               instrument=0,
                               program=0,
                               qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                               min_midi_pitch=music_constants.MIN_MIDI_PITCH,
                               wav_file=None):

  frame_length_seconds = 1 / frames_per_second

  sequence = music_pb2.NoteSequence()
  sequence.tempos.add().qpm = qpm
  sequence.ticks_per_quarter = music_constants.STANDARD_PPQ

  note_duration = frame_length_seconds * 3  # to remove redundant same midi
  total_frames = FLAGS.chunk_padding  # left padding

  onsets = np.zeros((0, 88), dtype='float32')

  last_note = {} # {'pitch': time}

  def unscale_velocity(velocity):
    unscaled = max(min(velocity, 1.), 0) * 80. + 10.
    if math.isnan(unscaled):
      return 0
    return int(unscaled)

  def process_chunk(chunk_prediction):
    nonlocal total_frames
    nonlocal onsets

    onset_predictions = chunk_prediction.onset_predictions
    velocity_values = chunk_prediction.velocity_values

    onset_probs_flat = chunk_prediction.onset_probs_flat

    onsets = np.concatenate((onsets, onset_probs_flat), axis=0)

    for i, onset in enumerate(onset_predictions):
      for pitch, active in enumerate(onset):
        if active:
          time = (total_frames + i) * frame_length_seconds
          pitch = pitch + min_midi_pitch
          if time - last_note.get(pitch, -1) > note_duration:
            note = sequence.notes.add()
            note.start_time = time
            note.end_time = time + note_duration
            note.pitch = pitch
            note.velocity = unscale_velocity(velocity_values[i, pitch] if velocity_values else velocity)
            note.instrument = instrument
            note.program = program

            last_note[note.pitch] = note.start_time
            print('note:', note.pitch)

    total_frames += len(onset_predictions)

  print('begin process chunk')
  for chunk in chunk_func_c(wav_file):
    process_chunk(chunk)

  print('end process chunk')
  sequence.total_time = total_frames * frame_length_seconds
  return sequence, onsets


def get_eval_pred_result(score_id, lesson_id, audio_url, content):
  if not audio_url:
    return None

  headers = {
    'TOKEN': PRE_USER_TOKEN,
    'SERVICEID': '3'
  }

  data = {
    'score_id': score_id,
    'lesson_id': lesson_id,
    'audio_url': audio_url,
    'version': '6.1.0',
    'platform': 'android',
  }

  content_file = '/tmp/midi_content.json"'
  with open(content_file, "wb") as f:
    tf.logging.info('writing audio file: %s', f.name)
    content = msgpack.packb(content)
    c = gzip.compress(content)
    f.write(c)
    f.flush()

  url = AI_BACKEND_PRE_SND_EVALUATE_PRED_URL
  response = requests.post(url, data=data, headers=headers, files={'file': open(content_file, 'rb')})
  if response.status_code == 200:
    resp = response.json()
    print(resp)
    return resp
  else:
    print('load_eval_record: error[{code}]'.format(code=response.status_code))
    return None


def fix_tokens():
  print('fix_tokens!!!')
  predict_csv_file = pd.read_csv(FLAGS.output_csv_path)
  total_count = predict_csv_file.shape[0]
  count = 0
  for index, row in predict_csv_file.iterrows():
    score_id, lesson_id, audio_url = int(row['score_id']), int(row['lesson_id']), row['audio_url']
    prod_eval_url = PROD_EVAL_URL.format(scoreId=score_id, lessonId=lesson_id, recordId=row['prod_eval_id'],
                                         appToken=PROD_USER_TOKEN)
    predict_csv_file.loc[index, 'prod_eval_url'] = prod_eval_url

    eval_url = PRE_EVAL_URL.format(scoreId=score_id, lessonId=lesson_id, recordId=row['eval_id'], appToken=PRE_USER_TOKEN)
    predict_csv_file.loc[index, 'eval_url'] = eval_url
    count = count + 1
    print('doing: %d / %d' % (count, total_count))

  predict_csv_file.to_csv(FLAGS.output_csv_path, columns=['score_id', 'lesson_id', 'audio_url',
                                                          'prod_eval_id', 'prod_simple_speed', 'prod_simple_complete',
                                                          'prod_simple_pitch',
                                                          'prod_simple_final',
                                                          'eval_id', 'simple_speed', 'simple_complete', 'simple_pitch',
                                                          'simple_final',
                                                          'prod_eval_url',
                                                          'eval_url'])
  print('done')


def main(argv):
  del argv

  if not os.path.isfile(FLAGS.input_csv_path):
    print('input csv file empty!!')
    return

  if FLAGS.mode == 'token':
    fix_tokens()
    return

  tf.logging.set_verbosity(FLAGS.log)
  tf.logging.info('init...')

  config = configs.CONFIG_MAP[FLAGS.config]
  hparams = config.hparams

  hparams.use_cudnn = tf.test.is_gpu_available()
  hparams.parse(FLAGS.hparams)
  hparams.truncated_length_secs = 0

  interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  chunk_frames = int(input_details[0]['shape'].tolist()[0] / 229)
  chunk_padding = FLAGS.chunk_padding
  frames_nopadding = chunk_frames - chunk_padding * 2
  assert frames_nopadding > 0

  print(input_details)
  print(output_details)

  def gen_input(wav_filename):
    spec = wav2spec.wav2spec(wav_filename)
    for i in range(chunk_padding, spec.shape[0], frames_nopadding):
      start = i - chunk_padding
      end  = i + frames_nopadding + chunk_padding
      chunk_spec = spec[start:end]
      if chunk_spec.shape[0] == chunk_padding * 2 + frames_nopadding:
        input_item = {
          'spec': chunk_spec.flatten()
        }
        yield input_item


  def chunk_func(wav_filename):
    start_time = time.time()
    print(wav_filename)
    for input_item in gen_input(wav_filename):
      interpreter.set_tensor(input_details[0]['index'], input_item['spec'])
      interpreter.invoke()

      onset_probs_flat = interpreter.get_tensor(output_details[0]['index'])
      # velocity_values_flat = interpreter.get_tensor(output_details[1]['index'])

      if chunk_padding > 0:
        onset_probs_flat = onset_probs_flat[chunk_padding:-chunk_padding]
        # velocity_values_flat = velocity_values_flat[chunk_padding:-chunk_padding]

      onset_predictions = onset_probs_flat > FLAGS.onset_threshold
      # velocity_values = velocity_values_flat

      yield ChunkPrediction(
        onset_predictions=onset_predictions,
        velocity_values=None,
        onset_probs_flat=onset_probs_flat)

  print('create predict csv file...')
  eval_ids = []
  score_ids = []
  lesson_ids = []
  audio_urls = []
  simple_pitch_probs = []
  simple_finals = []
  simple_speeds = []
  simple_completes = []
  eval_urls = []

  prod_eval_ids = []
  prod_simple_pitches = []
  prod_simple_finals = []
  prod_simple_speeds = []
  prod_simple_completes = []
  prod_eval_urls = []

  ppea_prod_csv_file = pd.read_csv(FLAGS.input_csv_path)

  total_count = ppea_prod_csv_file.shape[0]
  count = 0
  for index, row in ppea_prod_csv_file.iterrows():
    score_id, lesson_id, audio_url = int(row['score_id']), int(row['lesson_id']), row['audio_url']
    if pd.isna(score_id) or pd.isna(lesson_id) or pd.isna(audio_url):
      print('error input row info')
      return

    wav_content = download_snd_file(audio_url)
    wav_file_name = '/tmp/record.aac'
    f = open(wav_file_name, mode='wb')
    f.write(wav_content)
    f.flush()
    f.close()
    tf.logging.info('writing audio file: %s', wav_file_name)
    sequence_prediction, onsets = pianoroll_to_note_sequence(
          chunk_func,
          frames_per_second=data.hparams_frames_per_second(hparams),
          min_duration_ms=0,
          min_midi_pitch=constants.MIN_MIDI_PITCH,
          wav_file=wav_file_name
      )

    # save midi files
    # midi_io.sequence_proto_to_midi_file(sequence_prediction, predicted_label_midi_file)

    # create midi probs
    pretty_midi_object = midi_io.note_sequence_to_pretty_midi(sequence_prediction)
    with io.BytesIO() as buf:
      pretty_midi_object.write(buf)
      midi_b64 = base64.b64encode(buf.getvalue()).decode()

    midi_content = {
        'midi': midi_b64,
        'preds': {'onsets': utils.to_sparse_onsets(onsets)},
    }

    resp = get_eval_pred_result(score_id, int(lesson_id), audio_url, midi_content)
    if not resp or len(resp.get('data')) == 0:
      print('error ppea eval pred request: %d score_id: %d lesson_id: %d' % (row['id'], score_id, int(lesson_id)))
      continue

    prod_eval_ids.append(row['id'])
    prod_simple_pitches.append(row['simple_pitch'])
    prod_simple_finals.append(row['simple_final'])
    prod_simple_speeds.append(row['simple_speed'])
    prod_simple_completes.append(row['simple_complete'])

    prod_eval_url = PROD_EVAL_URL.format(scoreId=score_id, lessonId=lesson_id, recordId=row['id'], appToken=PROD_USER_TOKEN)
    prod_eval_urls.append(prod_eval_url)

    score_ids.append(score_id)
    lesson_ids.append(lesson_id)
    audio_urls.append(audio_url)

    eval_id = resp.get('data').get('score').get('eval_id')
    simple_pitch_prob = resp.get('data').get('score').get('simple_pitch')
    simple_final = resp.get('data').get('score').get('simple_final')
    simple_speed = resp.get('data').get('score').get('simple_speed')
    simple_complete = resp.get('data').get('score').get('simple_complete')
    simple_finals.append(simple_final)
    eval_ids.append(eval_id)
    simple_pitch_probs.append(simple_pitch_prob)
    simple_speeds.append(simple_speed)
    simple_completes.append(simple_complete)
    eval_url = PRE_EVAL_URL.format(scoreId=score_id, lessonId=lesson_id, recordId=eval_id, appToken=PRE_USER_TOKEN)
    eval_urls.append(eval_url)

    count = count + 1
    print('doing: %d / %d' % (count, total_count))

  result = pd.DataFrame()
  result['score_id'] = score_ids
  result['lesson_id'] = lesson_ids
  result['audio_url'] = audio_urls
  result['prod_eval_id'] = prod_eval_ids
  result['prod_simple_speed'] = prod_simple_speeds
  result['prod_simple_complete'] = prod_simple_completes
  result['prod_simple_pitch'] = prod_simple_pitches
  result['prod_simple_final'] = prod_simple_finals

  result['eval_id'] = eval_ids
  result['simple_speed'] = simple_speeds
  result['simple_complete'] = simple_completes
  result['simple_pitch'] = simple_pitch_probs
  result['simple_final'] = simple_finals

  result['prod_eval_url'] = prod_eval_urls
  result['eval_url'] = eval_urls

  result.to_csv(FLAGS.output_csv_path, encoding='utf_8_sig')
  print('done')


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()

