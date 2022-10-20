
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import os

import tensorflow as tf

import pandas as pd


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_csv_path', './ppea_prod_result.csv',
                           'input ppea csv result from prod')
tf.app.flags.DEFINE_string('output_csv_path', './ppea_predict_result.csv',
                           'output ppea predict csv result')

tf.app.flags.DEFINE_enum(
    'mode', 'new', ['new', 'token'],
    'which mode will be used')

AI_BACKEND_SELF_SND_EVAL_URL_FOR_PRE = 'http://0.0.0.0:20000/api/1.0/eval/evaluate_snd2'

PROD_USER_TOKEN = '57a44d11bdb7c7034791aac30fe280ac'
PROD_EVAL_URL = 'https://musicscore-app.xiaoyezi.com/#/reportNew?scoreId={scoreId}&lessonId={lessonId}&recordId={recordId}&platform=2&version=5.5.0&appToken={appToken}'


PRE_USER_TOKEN = '6437c43c79e0fada47fd6297ef51dfb6'
PRE_EVAL_URL = 'https://musicscore-app-pre.xiaoyezi.com/#/reportNew?scoreId={scoreId}&lessonId={lessonId}&recordId={recordId}&platform=2&version=5.5.0&appToken={appToken}'


def download_snd_file(audio_url):
  r = requests.get(audio_url)
  return r.content


def get_ppea_midi_result(score_id, lesson_id, audio_url):
  if not audio_url:
    return None

  headers = {
    'TOKEN': 'MAGISTER_USER_1',
    'SERVICEID': '3'
  }

  data = {
    'score_id': score_id,
    'lesson_id': lesson_id,
    'audio_url': audio_url,
    'version': '5.4.0',
    'platform': 'android',
  }

  url = AI_BACKEND_SELF_SND_EVAL_URL_FOR_PRE
  response = requests.post(url, data=data, headers=headers)
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
    if pd.isna(score_id) or pd.isna(lesson_id) or pd.isna(audio_url):
      print('error input row info')
      return
    resp = get_ppea_midi_result(score_id, int(lesson_id), audio_url)
    if not resp:
      print('error ppea request: %d' % row['id'])
      return

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
