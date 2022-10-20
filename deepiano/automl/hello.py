from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def run():
  print('hello world!!!!')


def main(argv):
  del argv
  run()


if __name__ == '__main__':
  tf.app.run(main)
