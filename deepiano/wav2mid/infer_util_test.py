"""Tests for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepiano.wav2mid import infer_util

import numpy as np
import tensorflow.compat.v1 as tf


class InferUtilTest(tf.test.TestCase):

  def testProbsToPianorollViterbi(self):
    frame_probs = np.array([[0.2, 0.1], [0.5, 0.1], [0.5, 0.1], [0.8, 0.1]])
    onset_probs = np.array([[0.1, 0.1], [0.1, 0.1], [0.9, 0.1], [0.1, 0.1]])
    pianoroll = infer_util.probs_to_pianoroll_viterbi(frame_probs, onset_probs)
    np.testing.assert_array_equal(
        [[False, False], [False, False], [True, False], [True, False]],
        pianoroll)


if __name__ == '__main__':
  tf.test.main()
