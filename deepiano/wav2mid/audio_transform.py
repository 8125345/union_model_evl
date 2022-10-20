"""SoX-based audio transform functions for the purpose of data augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import subprocess
import tempfile
from glob import glob
from functools import lru_cache

import sox
import tensorflow.compat.v1 as tf
from tensorflow.contrib import training as contrib_training

# The base pipeline is a list of stages, each of which consists of a name
# (corresponding to a SoX function) and a dictionary of parameters with name
# (corresponding to an argument to the SoX function) and min and max values
# along with the scale (linear or log) used to sample values. The min and max
# values can be overridden with hparams.
AUDIO_TRANSFORM_PIPELINE = [
    # Pitch shift.
    ('pitch', {
        'n_semitones': (-0.1, 0.1, 'linear'),
    }),

    # Contrast (simple form of compression).
    ('contrast', {
        'amount': (0.0, 100.0, 'linear'),
    }),

    # Two independent EQ modifications.
    ('equalizer', {
        'frequency': (32.0, 4096.0, 'log'),
        'width_q': (2.0, 2.0, 'linear'),
        'gain_db': (-10.0, 5.0, 'linear'),
    }),
    ('equalizer', {
        'frequency': (32.0, 4096.0, 'log'),
        'width_q': (2.0, 2.0, 'linear'),
        'gain_db': (-10.0, 5.0, 'linear'),
    }),

    # Reverb (for now just single-parameter).
    ('reverb', {
        'reverberance': (0.0, 70.0, 'linear'),
    }),
]

# Default hyperparameter values from the above pipeline. Note the additional
# `transform_audio` hparam that defaults to False, i.e. by default no audio
# transformation will be performed.
DEFAULT_AUDIO_TRANSFORM_HPARAMS = contrib_training.HParams(
    transform_audio=False,
    audio_transform_noise_enable=False,
    audio_transform_noise_type='recorded',
    audio_transform_noise_dir='../../data/noise/vocal/',
    audio_transform_min_noise_vol=0.0,
    audio_transform_max_noise_vol=0.04,
    **dict(('audio_transform_%s_%s_%s' % (m, stage_name, param_name), value)
           for stage_name, params_dict in AUDIO_TRANSFORM_PIPELINE
           for param_name, (min_value, max_value, _) in params_dict.items()
           for m, value in [('min', min_value), ('max', max_value)]))


class AudioTransformParameter(object):
  """An audio transform parameter with min and max value."""

  def __init__(self, name, min_value, max_value, scale):
    """Initialize an AudioTransformParameter.

    Args:
      name: The name of the parameter. Should be the same as the name of the
          parameter passed to sox.
      min_value: The minimum value of the parameter, a float.
      max_value: The maximum value of the parameter, a float.
      scale: 'linear' or 'log', the scale with which to sample the parameter
          value.

    Raises:
      ValueError: If `scale` is not 'linear' or 'log'.
    """
    if scale not in ('linear', 'log'):
      raise ValueError('invalid parameter scale: %s' % scale)

    self.name = name
    self.min_value = min_value
    self.max_value = max_value
    self.scale = scale

  def sample(self):
    """Sample the parameter, returning a random value in its range.

    Returns:
      A value drawn uniformly at random between `min_value` and `max_value`.
    """
    if self.scale == 'linear':
      return random.uniform(self.min_value, self.max_value)
    else:
      log_min_value = math.log(self.min_value)
      log_max_value = math.log(self.max_value)
      return math.exp(random.uniform(log_min_value, log_max_value))


class AudioTransformStage(object):
  """A stage in an audio transform pipeline."""

  def __init__(self, name, params):
    """Initialize an AudioTransformStage.

    Args:
      name: The name of the stage. Should be the same as the name of the method
          called on a sox.Transformer object.
      params: A list of AudioTransformParameter objects.
    """
    self.name = name
    self.params = params

  def apply(self, transformer):
    """Apply this stage to a sox.Transformer object.

    Args:
      transformer: The sox.Transformer object to which this pipeline stage
          should be applied. No audio will actually be transformed until the
          `build` method is called on `transformer`.
    """
    args = dict((param.name, param.sample()) for param in self.params)
    getattr(transformer, self.name)(**args)


def construct_pipeline(hparams, pipeline):
  """Construct an audio transform pipeline from hyperparameters.

  Args:
    hparams: A tf.contrib.training.HParams object specifying hyperparameters to
        use for audio transformation. These hyperparameters affect the min and
        max values for audio transform parameters.
    pipeline: A list of pipeline stages, each specified as a tuple of stage
        name (SoX method) and a dictionary of parameters.

  Returns:
    The resulting pipeline, a list of AudioTransformStage objects.
  """
  return [
      AudioTransformStage(
          name=stage_name,
          params=[
              AudioTransformParameter(
                  param_name,
                  getattr(hparams, 'audio_transform_min_%s_%s' % (stage_name,
                                                                  param_name)),
                  getattr(hparams, 'audio_transform_max_%s_%s' % (stage_name,
                                                                  param_name)),
                  scale)
              for param_name, (_, _, scale) in params_dict.items()
          ]) for stage_name, params_dict in pipeline
  ]


def run_pipeline(pipeline, input_filename, output_filename):
  """Run an audio transform pipeline.

  This will run the pipeline on an input audio file, producing an output audio
  file. Transform parameters will be sampled at each stage.

  Args:
    pipeline: The pipeline to run, a list of AudioTransformStage objects.
    input_filename: Path to the audio file to be transformed.
    output_filename: Path to the resulting output audio file.
  """
  transformer = sox.Transformer()
  transformer.set_globals(guard=True)
  for stage in pipeline:
    stage.apply(transformer)
  transformer.build(input_filename, output_filename)


def add_noise(input_filename, output_filename, noise_vol, noise_type, noise_dir):
  """Add noise to a wav file using sox.

  Args:
    input_filename: Path to the original wav file.
    output_filename: Path to the output wav file that will consist of the input
        file plus noise.
    noise_vol: The volume of the noise to add.
    noise_type: One of "whitenoise", "pinknoise", "brownnoise".

  Raises:
    ValueError: If `noise_type` is not one of "whitenoise", "pinknoise", or
        "brownnoise".
  """
  if noise_type not in ('whitenoise', 'pinknoise', 'brownnoise', 'recorded'):
    raise ValueError('invalid noise type: %s' % noise_type)

  import time
  import shutil
  if noise_type == 'recorded':
      add_pre_recorded_noise(input_filename, output_filename, noise_dir)
  else:
      args = ['sox', input_filename, '-p', 'synth', noise_type, 'vol',
              str(noise_vol), '|', 'sox', '-m', input_filename, '-',
              output_filename]
      command = ' '.join(args)
      tf.logging.info('Executing: %s', command)

      process_handle = subprocess.Popen(
          command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
      process_handle.communicate()


def get_audio_duration(fn):
    duration = subprocess.run(['soxi', '-D', fn], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    return float(duration)


@lru_cache(32)
def read_noise_files(noise_dir):
    result = {}
    for fn in glob(os.path.join(noise_dir, '*.wav')):
        duration = get_audio_duration(fn)
        result[fn] = duration

    return list(result.items())


def add_pre_recorded_noise(input_filename, output_filename, noise_dir, noise_vol_range=(-40, -5)):
    noise_file_info = read_noise_files(noise_dir)
    if not noise_file_info:
        print('no noise files, skip')
        return
    print(noise_file_info)
    noise_file, noise_duration = random.choice(noise_file_info)
    noise_vol = random.uniform(*noise_vol_range)
    input_duration = get_audio_duration(input_filename)
    start = random.uniform(0, noise_duration - input_duration)

    command = 'sox {noise_file} -p trim {start} {input_duration} fade q 0.05 {input_duration} 0.05 gain {noise_vol} | sox -m {input_filename} - {output_filename}'.format(**{
        'input_filename': input_filename,
        'output_filename': output_filename,
        'noise_file': noise_file,
        'noise_vol': noise_vol,
        'input_duration': input_duration,
        'start': start,
    })

    print(command)

    process_handle = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    process_handle.communicate()


def transform_wav_audio(wav_audio, hparams, pipeline=None):
  """Transform the contents of a wav file based on hyperparameters.

  Args:
    wav_audio: The contents of a wav file; this will be written to a temporary
        file and transformed via SoX.
    hparams: The tf.contrib.training.HParams object to use to construct the
        audio transform pipeline.
    pipeline: A list of pipeline stages, each specified as a tuple of stage
        name (SoX method) and a dictionary of parameters. If None, uses
        `AUDIO_TRANSFORM_PIPELINE`.

  Returns:
    The contents of the wav file that results from applying the audio transform
    pipeline to the input audio.
  """
  if not hparams.transform_audio:
    return wav_audio

  pipeline = construct_pipeline(
      hparams, pipeline if pipeline is not None else AUDIO_TRANSFORM_PIPELINE)

  with tempfile.NamedTemporaryFile(suffix='.wav') as temp_input_with_noise:
    if not hparams.audio_transform_noise_enable:
      temp_input_with_noise.write(wav_audio)
      temp_input_with_noise.flush()
      print("no add noise:", hparams.audio_transform_noise_enable, temp_input_with_noise.name)
    else:
      with tempfile.NamedTemporaryFile(suffix='.wav') as temp_input:
        temp_input.write(wav_audio)
        temp_input.flush()

        # Add noise before all other pipeline steps.
        noise_vol = random.uniform(hparams.audio_transform_min_noise_vol,
                                   hparams.audio_transform_max_noise_vol)
        add_noise(temp_input.name, temp_input_with_noise.name, noise_vol,
                  hparams.audio_transform_noise_type, hparams.audio_transform_noise_dir)

    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_output:
      run_pipeline(pipeline, temp_input_with_noise.name, temp_output.name)
      return temp_output.read()
