"""Tests for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deepiano.wav2mid import metrics
from deepiano.protobuf import music_pb2

import numpy as np
import tensorflow.compat.v1 as tf


class MetricsTest(tf.test.TestCase):

  def testSequenceToValuedIntervals(self):
    sequence = music_pb2.NoteSequence()
    sequence.notes.add(pitch=60, start_time=1.0, end_time=2.0, velocity=80)
    # Should be dropped because it is 0 duration.
    sequence.notes.add(pitch=60, start_time=3.0, end_time=3.0, velocity=90)

    intervals, pitches, velocities = metrics.sequence_to_valued_intervals(
        sequence)
    np.testing.assert_array_equal([[1., 2.]], intervals)
    np.testing.assert_array_equal([60], pitches)
    np.testing.assert_array_equal([80], velocities)


if __name__ == '__main__':
  tf.test.main()
