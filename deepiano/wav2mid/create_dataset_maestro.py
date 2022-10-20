r"""Beam job for creating transcription dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepiano.wav2mid import configs
from deepiano.wav2mid import create_dataset
from deepiano.wav2mid import data
import tensorflow.compat.v1 as tf


def main(argv):
  del argv


  create_dataset.pipeline(
      configs.CONFIG_MAP, configs.DATASET_CONFIG_MAP, data.preprocess_example,
      data.input_tensors_to_example)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
