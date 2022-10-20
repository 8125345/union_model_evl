
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import signal
import time
import sys

import requests
import os
import glob
from shutil import copyfile

import mido
import re
import json

import tensorflow as tf

from deepiano.automl.tflite_automl import Base

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '/data/maestro/AI_tagging_for_test_dataset_2020_0608',
                           'Directory where wav & labels are')
tf.app.flags.DEFINE_string('output_dir', '/data/test/ai_tagging_test_results_0608/ppea/for_AI_tagging_for_test_dataset_2020_0608_server-maestro-v2.0.0-16k-noised-and-ai-tagging-and-new-theone-flat-midi-noised-2020-06-16-model.ckpt-24000',
                           'Directory where the analyser results will be placed.')

tf.app.flags.DEFINE_enum(
    'mode', 'raw', ['raw', 'local', 'remote', 'remote_score_correct', 'auto_raw'],
    'which mode to convert wav file')

tf.app.flags.DEFINE_enum(
    'dataset', 'v1', ['v1', 'v2', 'self'],
    'which dataset will be used')

# used for transcribe_worker.py
tf.app.flags.DEFINE_string('model_dir', '../../data/models/maestro',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')


all_mp3_dir = []


AI_BACKEND_PROD_SND_EVAL_RECORD_URL = 'https://musvg.xiaoyezi.com/api/1.0/eval/record'

AI_BACKEND_PRE_SND_EVAL_RECORD_URL = 'http://musvg-pre.xiaoyezi.com/api/1.0/eval/record'

AI_BACKEND_PRE_SND_EVAL_URL = 'http://aibackend-pre.1tai.com/api/1.0/eval/evaluate_snd2'

AI_BACKEND_PRE_USB_EVAL_URL = 'http://aibackend-pre.1tai.com/api/1.0/eval/evaluate2'


# score_id, lesson_id, audio_url or local wav files
all_wav_info = [
  (8438, 8397, 'http://ai-backend.oss-cn-beijing.aliyuncs.com/prod/3/record/32114/8438/20200405170959_73599.aac')
]


# score_id, lesson_id
all_labeled_score_info = [
  (8438, 8397)
]

current_transcribe_worker_pid = None
current_transcribe_worker_parent_pid = None
transcribe_worker_pid_file = '/tmp/transcribe_worker_pid'

class TranscribeWorker(Base):
  def __init__(self, *args, **kwargs):
    super(TranscribeWorker, self).__init__(*args, **kwargs)

  def get_base_command_args(self):
    return ['nohup', 'python', '../server/transcribe_worker.py']

  def name(self):
    return 'transcribe_worker'


def midi_to_notes(midi):
  total_time = 0
  midi = list(midi)

  notes = []

  notesOn = []
  for m in midi:
    total_time = m.time = total_time + m.time * 1000  # from 's' to 'ms'
    if m.type == 'note_on':
      note = {
        'time': m.time,
        'pitch': m.note,
        'velocity': m.velocity,
        'prob': 1,
        'duration': 500
      }

      found = False
      for n in notesOn:
        if n.get('pitch') == m.note:
          found = True
          n['time'] = note.get('time')
          n['pitch'] = note.get('pitch')
          n['velocity'] = note.get('velocity')

      if not found:
        notesOn.append(note)
    elif m.type == 'note_off':
      for n in notesOn:
        if n.get('pitch') == m.note:
          n['duration'] = m.time - n.get('time')

          # update notes
          notes.append(n)

          notesOn.remove(n)
          break
    else:
      print(m)

  return notes


def midi_to_user_play_data(midi_file):
  midi = mido.MidiFile(midi_file)
  midi_notes = midi_to_notes(midi)
  return midi_notes


def send_usb_eval_req(score_id, lesson_id, midi_file):
  user_play_data = midi_to_user_play_data(midi_file)
  if not user_play_data:
    print('Error!!user_play_data is empty!')
    return

  headers = {
    'TOKEN': '6b7b59117b2276bd02b775d220d26a60',
    'SERVICEID': '3'
  }

  payload = {
    'score_id': score_id,
    'lesson_id': lesson_id,
    'version': '4.6.0',
    'platform': 'android',
    'play': {
      'notes': user_play_data
    }
  }

  url = AI_BACKEND_PRE_USB_EVAL_URL
  response = requests.post(url, json=payload, headers=headers)
  if response.status_code == 200:
    resp = response.json()
    print(resp)
    return resp
  else:
    print('load_eval_record: error[{code}]'.format(code=response.status_code))
    return None


def convert_usb_record_for_compare(mp3_file, label_midi_file):
  mp3_base_name = os.path.basename(mp3_file)
  base_name_no_suffix = os.path.splitext(mp3_base_name)[0]
  target_predict_json_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.predicted.json')
  target_label_json_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.label.json')

  if os.path.isfile(target_label_json_path) and os.path.isfile(target_predict_json_path):
    return

  if not os.path.isfile(target_predict_json_path):
    # get snd related eval record data
    eval_id = get_eval_id(mp3_file)
    if not eval_id:
      print('Bad mp3 file name! [%s]' % mp3_file)
      return

    resp = load_eval_record(AI_BACKEND_PROD_SND_EVAL_RECORD_URL, eval_id)
    if not resp:
      print('Empty eval record!')
      return
    target_data = resp.get('data')
    if target_data:
      with open(target_predict_json_path, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False)

  if not os.path.isfile(target_label_json_path):
    # get label related eval record data
    score_id = get_score_id(mp3_file)
    if not score_id:
      print('convert_usb_record_for_compare: failed to get score_id! [%s]' % mp3_file)
      return
    for target_score_id, target_lesson_id in all_labeled_score_info:
      if target_score_id != int(score_id):
        continue
      result = send_usb_eval_req(target_score_id, target_lesson_id, label_midi_file)
      if not result:
        print('send_usb_eval_req error!')
        continue

      eval_id = result.get('data').get('score').get('eval_id')
      resp = load_eval_record(AI_BACKEND_PRE_SND_EVAL_RECORD_URL, eval_id)
      if not resp:
        print('Empty eval record!')
        return
      data = resp.get('data')
      if data:
        with open(target_label_json_path, 'w', encoding='utf-8') as f:
          json.dump(data, f, ensure_ascii=False)

  return


def download_snd_file(audio_url):
  r = requests.get(audio_url)
  return r.content


def get_ppea_midi_result(score_id, lesson_id, audio_url, wav_file_name=None):
  if (wav_file_name and os.path.isfile(wav_file_name)) or not audio_url:
    return None

  if audio_url:
    wav_content = download_snd_file(audio_url)
    wav_file_name = '/tmp/record.aac'
    f = open(wav_file_name, mode='wb')
    f.write(wav_content)
    f.flush()
    f.close()

  headers = {
    'TOKEN': 'e6b80b9ea11c27729cf6d96993fa4dbb',
    'SERVICEID': '3'
  }

  data = {
    'score_id': score_id,
    'lesson_id': lesson_id,
    'version': '4.6.0',
    'platform': 'android',
  }

  request_file = {'file': open(wav_file_name, 'rb')}

  url = AI_BACKEND_PRE_SND_EVAL_URL
  response = requests.post(url, data=data, headers=headers, files=request_file)
  if response.status_code == 200:
    resp = response.json()
    print(resp)
    return resp
  else:
    print('load_eval_record: error[{code}]'.format(code=response.status_code))
    return None


def data_to_midi(play, filename):
  midi = mido.MidiFile(ticks_per_beat=500)
  track = mido.MidiTrack()
  midi.tracks.append(track)
  track.append(mido.MetaMessage('set_tempo', tempo=500000))
  for note in play['notes']:
    if note['pitch'] >= 0:
      track.append(mido.Message('note_on', note=note['pitch'], velocity=note['velocity'], time=note['time']))
      track.append(mido.Message('note_off', note=note['pitch'], velocity=note['velocity'],
                                time=note['time'] + note['duration']))
    elif note['pitch'] == -1:
      track.append(mido.Message('control_change', control=64, value=127, time=note['time']))
      track.append(mido.Message('control_change', control=64, value=0, time=note['time'] + note['duration']))

  track.sort(key=lambda m: m.time)
  last_t = 0
  for m in track:
    m.time, last_t = (m.time - last_t, m.time)
  midi.save(filename)


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


def get_score_id(mp3_file):
  matchObj = re.match(r'.*/(\d+)-.*.mp3', mp3_file, flags=0)
  if matchObj:
    return matchObj.group(1)
  return None


def convert_eval_record_to_midi(mp3_file, label_midi_file):
  eval_id = get_eval_id(mp3_file)
  if not eval_id:
    print('Bad mp3 file name! [%s]' % mp3_file)
    return

  resp = load_eval_record(AI_BACKEND_PROD_SND_EVAL_RECORD_URL, eval_id)
  if not resp:
    print('Empty eval record!')
    return
  user_play = resp.get('data').get('play')
  if user_play:
    mp3_base_name = os.path.basename(mp3_file)
    base_name_no_suffix = os.path.splitext(mp3_base_name)[0]
    target_predict_midi_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.predicted.midi')
    target_label_midi_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.label.midi')
    data_to_midi(user_play, target_predict_midi_path)
    copyfile(label_midi_file, target_label_midi_path)


def convert_snd_result(eval_id, mp3_file, label_midi_file):
  resp = load_eval_record(AI_BACKEND_PRE_SND_EVAL_RECORD_URL, eval_id)
  if not resp:
    print('Empty eval record!')
    return
  user_play = resp.get('data').get('play')
  if user_play:
    mp3_base_name = os.path.basename(mp3_file)
    base_name_no_suffix = os.path.splitext(mp3_base_name)[0]
    target_predict_midi_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.predicted.midi')
    target_label_midi_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.label.midi')
    data_to_midi(user_play, target_predict_midi_path)
    copyfile(label_midi_file, target_label_midi_path)


def convert(audio, midi):
  #url = 'http://snd-pre.research.xiaoyezi.com/pt/1.0/wav2mid'
  url = 'http://0.0.0.0:8086/pt/1.0/wav2mid'
  resp = requests.post(url, data=open(audio, 'rb'), verify=False)
  result = resp.json()
  midi_data = base64.decodebytes(result['midi'].encode())
  print('converted: {}'.format(audio))
  if len(midi_data) > 0:
    open(midi, 'wb').write(midi_data)


def convert_mp3_to_midi(mp3_file, label_midi_file):
  mp3_base_name = os.path.basename(mp3_file)
  base_name_no_suffix = os.path.splitext(mp3_base_name)[0]
  target_predict_midi_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.predicted.midi')
  target_label_midi_path = os.path.join(FLAGS.output_dir, base_name_no_suffix + '.label.midi')
  if not os.path.isfile(target_predict_midi_path):
    convert(mp3_file, target_predict_midi_path)
  copyfile(label_midi_file, target_label_midi_path)


def convert_mp3_by_remote(mp3_file, label_midi_file):
  for score_id, lesson_id, audio_url in all_wav_info:
    result = get_ppea_midi_result(score_id, lesson_id, audio_url)
    if not result:
      print('get_ppea_midi_result error!')
      continue
    eval_id = result.get('data').get('score').get('eval_id')
    convert_snd_result(eval_id, mp3_file, label_midi_file)


def iter_dirs(rootDir):
  for root, dirs, files in os.walk(rootDir):
    if dirs != []:
      for dirname in dirs:
        full_dirname = os.path.join(root, dirname)
        all_mp3_dir.append(full_dirname)
        iter_dirs(full_dirname)


def generate_predict_set(input_dirs):
  mp3_file_pairs = []
  all_files = []
  all_eval_ids = []
  if len(input_dirs) == 0:
    input_dirs.append(FLAGS.input_dir)
  for directory in list(set(input_dirs)):
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    mp3_files = glob.glob(path)
    # find matching mid files
    for mp3_file in mp3_files:
      if not os.path.isfile(mp3_file):
        print('empty mp3_file:' + mp3_file)
        continue

      filepath, file_name = os.path.split(mp3_file)
      m = re.match(r'.*-(.*).mp3', file_name)
      if m:
        if m.group(1) in all_eval_ids:
            print('dup eval id:' + m.group(1))
        all_eval_ids.append(m.group(1))
      if file_name in all_files:
        print('dupfile: ' + filepath + '/' + file_name)
      all_files.append(file_name)
      base_name, _ = os.path.splitext(mp3_file)
      mid_file = base_name + '.mid'
      mid_2nd_file = base_name + '.mp3.fix.MID'
      mid_3nd_file = base_name + '.fix.mid'
      mid_4nd_file = base_name + '.fix.MID'
      mid_5nd_file = base_name + '.fix1.mid'
      if os.path.isfile(mid_file):
        mp3_file_pairs.append((mp3_file, mid_file))
      elif os.path.isfile(mid_2nd_file):
        mp3_file_pairs.append((mp3_file, mid_2nd_file))
      elif os.path.isfile(mid_3nd_file):
        mp3_file_pairs.append((mp3_file, mid_3nd_file))
      elif os.path.isfile(mid_4nd_file):
        mp3_file_pairs.append((mp3_file, mid_4nd_file))
      elif os.path.isfile(mid_5nd_file):
        mp3_file_pairs.append((mp3_file, mid_5nd_file))
      else:
        print('empty midi:' + base_name)
  print(len(list(set(all_files))))
  return list(set(mp3_file_pairs))


def parse_config_json(for_v1):
  if for_v1: 
    json_file_name = os.path.join(FLAGS.input_dir, 'maestro-v1.0.0.json')
  else:
    json_file_name = os.path.join(FLAGS.input_dir, 'maestro-v2.0.0.json')

  train_data_set = {}
  test_data_set = {}
  validation_data_set = {}
  with open(json_file_name) as f:
    configs = json.load(f)
    for config in configs:
      year = config['year']
      midi_filename = config['midi_filename']
      audio_filename = config['audio_filename']
      split = config['split']
      if split == 'train':
        train_data = train_data_set.get(year)
        if not train_data:
          train_data_set[year] = [(year, midi_filename, audio_filename)]
        else:
          train_data.append((year, midi_filename, audio_filename))
      elif split == 'test':
        test_data = test_data_set.get(year)
        if not test_data:
          test_data_set[year] = [(year, midi_filename, audio_filename)]
        else:
          test_data.append((year, midi_filename, audio_filename))
      elif split == 'validation':
        validation_data = validation_data_set.get(year)
        if not validation_data:
          validation_data_set[year] = [(year, midi_filename, audio_filename)]
        else:
          validation_data.append((year, midi_filename, audio_filename))
  return train_data_set, test_data_set, validation_data_set


def generate_test_set(for_v1=False):
  """Generate the train TFRecord."""
  train_data_set, test_data_set, validation_data_set = parse_config_json(for_v1)
  file_pairs = []
  print(len(test_data_set))
  for year, dataset in test_data_set.items():
    for year, midi_filename, audio_filename in dataset:
      if for_v1:
        fixed_midi_filename = midi_filename.replace('.midi', '.midi')
        fixed_audio_filename = audio_filename.replace('.wav', '.wav')
      else:
        fixed_midi_filename = midi_filename.replace('.midi', '_16k.midi')
        fixed_audio_filename = audio_filename.replace('.wav', '_16k.wav')

      fixed_midi_path = os.path.join(FLAGS.input_dir, fixed_midi_filename)
      fixed_audio_path = os.path.join(FLAGS.input_dir, fixed_audio_filename)
      # find matching mid files
      if os.path.isfile(fixed_midi_path) and os.path.isfile(fixed_audio_path):
        file_pairs.append((fixed_audio_path, fixed_midi_path))

  return file_pairs


def start_transcribe_worker():
  info = {
    'model_dir': FLAGS.model_dir,
  }
  if FLAGS.checkpoint_path:
    info.update({'checkpoint_path': FLAGS.checkpoint_path})

  worker = TranscribeWorker(**info)
  trans_worker_handle = worker.start(is_async=True)
  if not trans_worker_handle:
    print('start_transcribe_worker error!')
    return

def kill_trans_worker(pid):
  no_such_pid = False
  if pid is not None:
    try:
      if isinstance(pid, str):
        pid = int(pid)
      os.killpg(pid, signal.SIGUSR1)
    except Exception as e:
      no_such_pid = True
      print(e)
  return no_such_pid


def signal_handler(signum, frame):
  if (os.path.isfile(transcribe_worker_pid_file)):
    with open(transcribe_worker_pid_file, 'r') as f:
      line = f.readline()
      current_transcribe_worker_pid, current_transcribe_worker_parent_pid = line.split(',')
  if current_transcribe_worker_parent_pid is not None:
    kill_trans_worker(current_transcribe_worker_parent_pid)
  print('I received: ', signum, ' parent pid:', current_transcribe_worker_parent_pid, ' quit!!!')
  sys.exit(0)

def main(argv):
  del argv

  print('start {}'. format(FLAGS.mode))


  if FLAGS.mode == 'auto_raw':

    # kill and restart transcribe_worker.py
    transcribe_worker_pid = None

    if (os.path.isfile(transcribe_worker_pid_file)):
      with open(transcribe_worker_pid_file, 'r') as f:
        line = f.readline()
        transcribe_worker_pid, transcribe_worker_parent_pid = line.split(',')
        print('transcribe_worker pid: {}, kill it!'.format(transcribe_worker_pid))
        # pid for nohup mode
        no_such_pid = kill_trans_worker(transcribe_worker_parent_pid)
        while not no_such_pid and os.path.isfile(transcribe_worker_pid_file):
          print('wait transcribe_worker to quit!')
          time.sleep(1)
        if os.path.isfile(transcribe_worker_pid_file):
          os.remove(transcribe_worker_pid_file)
    else:
      time.sleep(5)

    print('start_transcribe_worker!')
    start_transcribe_worker()

    while not os.path.isfile(transcribe_worker_pid_file):
      print('waiting for transcribe_worker')
      time.sleep(2)
    with open(transcribe_worker_pid_file, 'r') as f:
      line = f.readline()
      current_transcribe_worker_pid, current_transcribe_worker_parent_pid = line.split(',')

    print('transcribe_worker is running {}! ' .format(current_transcribe_worker_pid))
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  iter_dirs(FLAGS.input_dir)
  mp3_pars = None
  if FLAGS.dataset == 'self':
      mp3_pars = generate_predict_set(all_mp3_dir)
  elif FLAGS.dataset == 'v1':
    mp3_pars = generate_test_set(True)
  elif FLAGS.dataset == 'v2':
    mp3_pars = generate_test_set(False)
  #print(len(mp3_pars))
  for mp3_file, label_midi_file in mp3_pars:
    if FLAGS.mode == 'remote':
      convert_mp3_by_remote(mp3_file, label_midi_file)
    elif FLAGS.mode == 'local':
      convert_eval_record_to_midi(mp3_file, label_midi_file)
    elif FLAGS.mode == 'remote_score_correct':
      convert_usb_record_for_compare(mp3_file, label_midi_file)
    else:
      convert_mp3_to_midi(mp3_file, label_midi_file)
 
  if current_transcribe_worker_parent_pid is not None:
    print('kill worker {}'.format(current_transcribe_worker_pid))

    # pid for nohup
    kill_trans_worker(current_transcribe_worker_parent_pid)
    while os.path.isfile(transcribe_worker_pid_file):
      print('waiting for transcribe_worker to quit')
      time.sleep(2)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
