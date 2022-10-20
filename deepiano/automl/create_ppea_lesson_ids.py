# -*- coding: utf-8 -*-

"""
将一个包含一系列评测记录的csv文件中，根据score_id补全lesson_id
"""
import re
import subprocess

import os

import pandas as pd
from subprocess import Popen

input_csv_file = './query_result.csv'
log_files = '/usr/local/apps/ai_backend_prod/logs/api/api.*'
output_csv_file = './query_result.csv'


def get_lesson_id(score_id):
  command = 'egrep -o "evaluate_snd2.*\'score_id\': \'{score_id}\'.*\'lesson_id\': \'([0-9]+)\'" {log_files}'.format(
    **{
      'score_id': score_id,
      'log_files': log_files
    }
  )
  print(command)

  lesson_ids=[]
  process_handle = Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  stdout, stderr = process_handle.communicate()
  outputs = stdout.decode().split('\n')
  for line in outputs:
    if len(line) == 0:
      continue
    matchObj = re.match(r'.*\'lesson_id\'\: \'([0-9]+)\'', line, flags=0)
    if matchObj:
      lesson_id = matchObj.group(1)
      lesson_ids.append(int(lesson_id))
  if process_handle.returncode == 0:
    print('{}: {} OK!'.format('found score_id', score_id))
  else:
    print('{}: {} error!'.format('found score_id', score_id))
  lesson_ids = list(set(lesson_ids))
  if len(lesson_ids) >= 1:
    return lesson_ids[0]
  else:
    return -1


def fix_eval_cvs_file():
  eval_contents = pd.read_csv(input_csv_file)
  total_count = eval_contents.shape[0]
  count = 0
  for index, row in eval_contents.iterrows():
    lesson_id = row['lesson_id']
    count = count + 1
    if pd.isna(lesson_id):
      score_id = row['score_id']
      lesson_id = get_lesson_id(score_id)
      if lesson_id > 0:
        print('found score_id: %d, lesson_id: %d, %d / %d' % (score_id, lesson_id, count, total_count))
        eval_contents.loc[index, 'lesson_id'] = int(lesson_id)

  eval_contents.to_csv(output_csv_file, columns=['id', 'score_id', 'audio_url', 'simple_speed', 'simple_complete',
                                                 'simple_pitch', 'simple_final', 'lesson_id'])


def main():
  if not os.path.isfile(input_csv_file):
    print('empty csv file')
    return

  fix_eval_cvs_file()


if __name__ == '__main__':
  main()
