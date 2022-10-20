"""Defines shared constants used in transcription models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import librosa


MIN_MIDI_PITCH = librosa.note_to_midi('A0')
MAX_MIDI_PITCH = librosa.note_to_midi('C8')
MIDI_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1
