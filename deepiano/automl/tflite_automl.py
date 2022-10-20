"""
1. 将训练参数形成一张表，可根据该表自动进行模型训练(按各种参数组合，可以先后启动多个模型训练)
2. 将训练后的所有模型（按参数）转成tflite模型，并放入指定目录下
3. 搜索测试集test目录的所有文件，形成(wav, midi)列表
4. 将测试集(wav, midi)列表根据客户端预测参数，分别灌入所有tflite模型，保存为midi文件后，对比该预测midi文件与label midi文件的差别。
5. 将4中所有差异进行比较，排序输出最优解
6. 可以指定输出某个tflite模型的识别状态
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging

import os

import subprocess

import tensorflow as tf

SETTINGS = 'settings.json'
TYPE_TRAIN = 'train'
TYPE_EXPORT = 'export'
TYPE_PREDICT = 'predict'
TYPE_INFER = 'infer'
TYPE_PPEA = 'ppea'
TYPE_ANALYSE = 'analyse'

AUTO_ML = 'auto_ml'


class Base:
  def __init__(self, *args, **kwargs):
    super(Base, self).__init__()
    self.obj_args = args
    self.obj_kwargs = kwargs

    self.enabled = True
    self.for_server = False

    self.reset_command_args()

  def reset_command_args(self):
    base_args = self.get_base_command_args()
    for (k, v) in self.obj_kwargs.items():
      if k == 'hparams' and isinstance(v, dict):
        base_args.append('--hparams')
        hparams = ''
        for (h_k, h_v) in v.items():
          hparams = '{hparams}{h_k}={h_v},'.format(hparams=hparams, h_k=h_k, h_v=h_v)
        base_args.append(hparams[0:-1])
      else:
        if isinstance(v, bool):  # for bool args
          if k == 'enabled':
            self.enabled = v
          elif k == 'for_server':
            self.for_server = v
          elif not v:
            base_args.append('--no{k}'.format(k=k))
          else:
            base_args.append('--{k}'.format(k=k))
        elif isinstance(v, str):  # for string args
          base_args.append('--{k} "{v}"'.format(k=k, v=v))
        else:
          base_args.append('--{k} {v}'.format(k=k, v=v))
    self.command = ' '.join(base_args)

  def start(self, is_async=False):
    if not self.is_enabled():
      print('{} disabled! {}'.format(self.name, self.obj_kwargs))
      return True

    self.reset_command_args()
    print('{} start! {}'.format(self.name, self.obj_kwargs))

    if is_async:
      process_handle = subprocess.Popen(
        self.command, stdout=None, stderr=None, shell=True, preexec_fn=os.setpgrp)
      return process_handle
    process_handle = subprocess.Popen(
        self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process_handle.communicate()

    errors = stderr.decode().split('\n')
    for line in errors:
      logging.info(line)
    if process_handle.returncode == 0:
      logging.info('{} OK!'.format(self.name()))
    else:
      logging.info('{} ERROR!'.format(self.name()))

    return process_handle.returncode == 0

  def stop(self):
    pass

  def get_base_command_args(self):
    return []

  def is_enabled(self):
    return self.enabled

  def is_for_server(self):
    return self.for_server

  def name(self):
    return 'Base'


class Train(Base):
  def __init__(self, *args, **kwargs):
    super(Train, self).__init__(*args, **kwargs)

  def get_base_command_args(self):
    if self.is_for_server():
      return ['python', '../test/train.py']
    else:
      return ['python', 'tflite_automl_train.py']

  def name(self):
    return 'Train'


class ExportTflite(Base):
  def __init__(self, *args, **kwargs):
    super(ExportTflite, self).__init__(*args, **kwargs)

  def get_base_command_args(self):
    return ['python', 'tflite_automl_export_tflite.py']

  def name(self):
    return 'ExportTflite'


class Predict(Base):
  def __init__(self, *args, **kwargs):
    super(Predict, self).__init__(*args, **kwargs)

  def get_base_command_args(self):
    return ['python', 'tflite_automl_predict.py']

  def name(self):
    return 'Predict'


class Infer(Base):
  def __init__(self, *args, **kwargs):
    super(Infer, self).__init__(*args, **kwargs)

  def get_base_command_args(self):
    return ['python', 'tflite_automl_infer.py']

  def name(self):
    return 'Infer'


class Ppea(Base):
  def __init__(self, *args, **kwargs):
    super(Ppea, self).__init__(*args, **kwargs)

  def get_base_command_args(self):
    return ['python', 'tflite_automl_ppea.py']

  def name(self):
    return 'Ppea'


class Analyser(Base):
  def __init__(self, *args, **kwargs):
    super(Analyser, self).__init__(*args, **kwargs)

  def get_base_command_args(self):
    return ['python', 'tflite_automl_analyser.py']

  def name(self):
    return 'Analyser'


def parseJson():
  with open(SETTINGS) as f:
    settings = json.loads(f.read())

  return settings


def run_auto_ml_plan(v):
  num_steps_plan = []
  train_task = None
  export_task = None
  predict_tasks = []
  infer_tasks = []
  analyse_task = None

  ppea_tasks = []

  for item in v:
    if item.get('type') == TYPE_TRAIN:
      steps_list = item.get('num_steps_plan')
      num_steps_begin = item.get('num_steps_begin')
      num_steps_end = item.get('num_steps_end')
      num_steps_gap = item.get('num_steps_gap')
      if steps_list is not None and len(steps_list) > 0:
        num_steps_plan.extend(item.get('num_steps_plan'))
      elif num_steps_begin is not None and num_steps_end is not None and num_steps_gap is not None:
        num_steps_plan.extend([step for step in range(num_steps_begin, num_steps_end + 1, num_steps_gap)])

      train_task = Train(**item)

    if item.get('type') == TYPE_EXPORT:
      export_task = ExportTflite(**item)

    if item.get('type') == TYPE_PREDICT:
      predict_task = Predict(**item)
      predict_tasks.append(predict_task)

    if item.get('type') == TYPE_INFER:
      infer_task = Infer(**item)
      infer_tasks.append(infer_task)

    if item.get('type') == TYPE_PPEA:
      dataset = item.get('dataset')
      for v in dataset:
        info = copy.deepcopy(item)
        info.pop('dataset')
        info.update({'dataset': v.get('dataset_type')})
        info.update({'name': v.get('name')})
        info.update({'input_dir': v.get('input_dir')})
        info.update({'output_dir': v.get('output_dir')})

        ppea_task = Ppea(**info)
        ppea_tasks.append(ppea_task)

    if item.get('type') == TYPE_ANALYSE:
      analyse_task = Analyser(**item)

  if not train_task or not export_task or not predict_tasks or not infer_tasks or not analyse_task or not ppea_tasks:
    print('auto ml config error!')
    return

  origin_export_tflite_output_model_path = export_task.obj_kwargs.get('output_model_path')

  origin_analyse_output_dir = analyse_task.obj_kwargs.get('output_dir')
  if not os.path.isdir(origin_analyse_output_dir):
    os.makedirs(origin_analyse_output_dir)

  print('BEGIN auto ml!!!')
  for num_steps in num_steps_plan:
    print('BEGIN auto ml loop: num_steps: {}'.format(num_steps))

    train_task.obj_kwargs['num_steps'] = num_steps
    ok = train_task.start()
    if not ok:
      print('train error! num_steps: {}'.format(num_steps))
      return

    train_model_dir = train_task.obj_kwargs.get('model_dir')
    fixed_export_tflite_output_model_path = origin_export_tflite_output_model_path + '-num_steps-{}'.format(num_steps)
    if not os.path.isfile(fixed_export_tflite_output_model_path):
      export_task.obj_kwargs['model_dir'] = train_model_dir
      export_task.obj_kwargs['output_model_path'] = fixed_export_tflite_output_model_path
      ok = export_task.start()
      if not ok:
        print('export tflite error! num_steps: {}'.format(num_steps))
        return

    # for tflite predict tasks
    for predict_task in predict_tasks:
      if not predict_task.is_enabled():
        continue
      origin_predict_output_dir = predict_task.obj_kwargs.get('output_dir')
      if not os.path.isdir(origin_predict_output_dir):
        os.makedirs(origin_predict_output_dir)

      predict_result_path = predict_task.obj_kwargs['name'] + '-num_steps-{}'.format(num_steps)
      fixed_predict_output_dir = os.path.join(origin_predict_output_dir, predict_result_path)
      predict_task.obj_kwargs['output_dir'] = fixed_predict_output_dir
      predict_task.obj_kwargs['model_path'] = fixed_export_tflite_output_model_path
      ok = predict_task.start()
      if not ok:
        print('predict error! num_steps: {}'.format(num_steps))
        return
      analyse_task.obj_kwargs['input_dir'] = fixed_predict_output_dir
      fixed_analyse_output_path = os.path.join(origin_analyse_output_dir, 'predict', predict_result_path)
      analyse_task.obj_kwargs['output_dir'] = fixed_analyse_output_path
      analyse_task.obj_kwargs['name'] = predict_result_path
      if not os.path.isdir(fixed_analyse_output_path):
        os.makedirs(fixed_analyse_output_path)
      ok = analyse_task.start()
      if not ok:
        print('analyse for predict error!num_steps: {}'.format(num_steps))
        return
      # recover output_dir
      predict_task.obj_kwargs['output_dir'] = origin_predict_output_dir

    # for tflite infer tasks
    fixed_infer_checkpoint_path = os.path.join(train_model_dir, 'model.ckpt-{}.meta'.format(num_steps))
    if not os.path.isfile(fixed_infer_checkpoint_path):
      print('infer error for no checkpoint_path!num_steps: {}'.format(num_steps))
      return

    for infer_task in infer_tasks:
      if not infer_task.is_enabled():
        continue
      origin_infer_output_dir = infer_task.obj_kwargs.get('output_dir')
      if not os.path.isdir(origin_infer_output_dir):
        os.makedirs(origin_infer_output_dir)

      infer_result_path = infer_task.obj_kwargs['name'] + '-num_steps-{}'.format(num_steps)
      fixed_infer_output_dir = os.path.join(origin_infer_output_dir, infer_result_path)
      infer_task.obj_kwargs['output_dir'] = fixed_infer_output_dir
      infer_task.obj_kwargs['checkpoint_path'] = os.path.join(train_model_dir, 'model.ckpt-{}'.format(num_steps))
      ok = infer_task.start()
      if not ok:
        print('infer error! num_steps: {}'.format(num_steps))
        return

      analyse_task.obj_kwargs['input_dir'] = fixed_infer_output_dir
      fixed_analyse_output_path = os.path.join(origin_analyse_output_dir, 'infer', infer_result_path)
      analyse_task.obj_kwargs['output_dir'] = fixed_analyse_output_path
      analyse_task.obj_kwargs['name'] = infer_result_path
      if not os.path.isdir(fixed_analyse_output_path):
        os.makedirs(fixed_analyse_output_path)

      ok = analyse_task.start()
      if not ok:
        print('analyse for infer error!num_steps: {}'.format(num_steps))
        return
      infer_task.obj_kwargs['output_dir'] = origin_infer_output_dir

    # for local ppea server tasks
    fixed_ppea_checkpoint_path = os.path.join(train_model_dir, 'model.ckpt-{}.meta'.format(num_steps))
    if not os.path.isfile(fixed_ppea_checkpoint_path):
      print('ppea error for no checkpoint_path!num_steps: {}'.format(num_steps))
      return

    for ppea_task in ppea_tasks:
      if not ppea_task.is_enabled():
        continue
      origin_ppea_output_dir = ppea_task.obj_kwargs.get('output_dir')
      if not os.path.isdir(origin_ppea_output_dir):
        os.makedirs(origin_ppea_output_dir)

      ppea_result_path = ppea_task.obj_kwargs['name'] + '-num_steps-{}'.format(num_steps)
      fixed_ppea_output_dir = os.path.join(origin_ppea_output_dir, ppea_result_path)
      ppea_task.obj_kwargs['output_dir'] = fixed_ppea_output_dir
      ppea_task.obj_kwargs['model_dir'] = train_model_dir
      ppea_task.obj_kwargs['checkpoint_path'] = os.path.join(train_model_dir, 'model.ckpt-{}'.format(num_steps))
      print('ppea start num_steps: {}'.format(num_steps))
      ok = ppea_task.start()
      print('ppea end num_steps: {}'.format(num_steps))
      if not ok:
        print('infer error! num_steps: {}'.format(num_steps))
        return

      analyse_task.obj_kwargs['input_dir'] = fixed_ppea_output_dir
      fixed_analyse_output_path = os.path.join(origin_analyse_output_dir, 'ppea', ppea_result_path)
      analyse_task.obj_kwargs['output_dir'] = fixed_analyse_output_path
      analyse_task.obj_kwargs['name'] = ppea_result_path
      if not os.path.isdir(fixed_analyse_output_path):
        os.makedirs(fixed_analyse_output_path)

      ok = analyse_task.start()
      if not ok:
        print('analyse for ppea error!num_steps: {}'.format(num_steps))
        return
      ppea_task.obj_kwargs['output_dir'] = origin_ppea_output_dir

    print('END auto ml loop: num_steps: {}'.format(num_steps))

  print('END auto ml!!!')


def run():
  settings = parseJson()

  for s in settings:
    v = settings[s]
    if isinstance(v, list) and s == AUTO_ML:
      run_auto_ml_plan(v)

  # train
  for s in settings:
    v = settings[s]
    if isinstance(v, list) and s == TYPE_TRAIN:
      for item in v:
        if item.get('type') == TYPE_TRAIN:
          print('begin train')
          train = Train(**item)
          ok = train.start()
          print('end train: {}'.format(ok))
          if not ok:
            print('stop train for error! {}'.format(ok))
            return

  # export tflite
  for s in settings:
    v = settings[s]
    if isinstance(v, list) and s == TYPE_EXPORT:
      for item in v:
        if item.get('type') == TYPE_EXPORT:
          print('begin export tflite')
          export_tflite = ExportTflite(**item)
          ok = export_tflite.start()
          print('end export tflite: {}'.format(ok))

  # predict
  for s in settings:
    v = settings[s]
    if isinstance(v, list) and s == TYPE_PREDICT:
      for item in v:
        if item.get('type') == TYPE_PREDICT:
          print('begin predict')
          predict = Predict(**item)
          ok = predict.start()
          print('end predict: {}'.format(ok))

  # infer: for tfrecord
  for s in settings:
    v = settings[s]
    if isinstance(v, list) and s == TYPE_INFER:
      for item in v:
        if item.get('type') == TYPE_INFER:
          print('begin infer')
          infer = Infer(**item)
          ok = infer.start()
          print('end infer: {}'.format(ok))

  # ppea: infer for server
  for s in settings:
    v = settings[s]
    if isinstance(v, list) and s == TYPE_PPEA:
      for item in v:
        if item.get('type') == TYPE_PPEA:
          print('begin ppea')
          infer = Ppea(**item)
          ok = infer.start()
          print('end ppea: {}'.format(ok))

  # analyse
  for s in settings:
    v = settings[s]
    if isinstance(v, list) and s == TYPE_ANALYSE:
      for item in v:
        if item.get('type') == TYPE_ANALYSE:
          print('begin analyse')
          analyser = Analyser(**item)
          ok = analyser.start()
          print('end analyse: {}'.format(ok))


def main(argv):
  del argv
  run()


if __name__ == '__main__':
  tf.app.run(main)
