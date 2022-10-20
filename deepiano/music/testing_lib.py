"""Testing support code."""

from deepiano.protobuf import music_pb2

from google.protobuf import text_format

# Shortcut to text annotation types.
BEAT = music_pb2.NoteSequence.TextAnnotation.BEAT
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


def add_track_to_sequence(note_sequence, instrument, notes,
                          is_drum=False, program=0):
  """Adds instrument track to NoteSequence."""
  for pitch, velocity, start_time, end_time in notes:
    note = note_sequence.notes.add()
    note.pitch = pitch
    note.velocity = velocity
    note.start_time = start_time
    note.end_time = end_time
    note.instrument = instrument
    note.is_drum = is_drum
    note.program = program
    if end_time > note_sequence.total_time:
      note_sequence.total_time = end_time


def add_chords_to_sequence(note_sequence, chords):
  for figure, time in chords:
    annotation = note_sequence.text_annotations.add()
    annotation.time = time
    annotation.text = figure
    annotation.annotation_type = CHORD_SYMBOL


def add_beats_to_sequence(note_sequence, beats):
  for time in beats:
    annotation = note_sequence.text_annotations.add()
    annotation.time = time
    annotation.annotation_type = BEAT


def add_control_changes_to_sequence(note_sequence, instrument, control_changes):
  for time, control_number, control_value in control_changes:
    control_change = note_sequence.control_changes.add()
    control_change.time = time
    control_change.control_number = control_number
    control_change.control_value = control_value
    control_change.instrument = instrument


def add_pitch_bends_to_sequence(
    note_sequence, instrument, program, pitch_bends):
  for time, bend in pitch_bends:
    pitch_bend = note_sequence.pitch_bends.add()
    pitch_bend.time = time
    pitch_bend.bend = bend
    pitch_bend.program = program
    pitch_bend.instrument = instrument
    pitch_bend.is_drum = False  # Assume false for this test method.


def add_quantized_steps_to_sequence(sequence, quantized_steps):
  assert len(sequence.notes) == len(quantized_steps)

  for note, quantized_step in zip(sequence.notes, quantized_steps):
    note.quantized_start_step = quantized_step[0]
    note.quantized_end_step = quantized_step[1]

    if quantized_step[1] > sequence.total_quantized_steps:
      sequence.total_quantized_steps = quantized_step[1]


def add_quantized_chord_steps_to_sequence(sequence, quantized_steps):
  chord_annotations = [a for a in sequence.text_annotations
                       if a.annotation_type == CHORD_SYMBOL]
  assert len(chord_annotations) == len(quantized_steps)
  for chord, quantized_step in zip(chord_annotations, quantized_steps):
    chord.quantized_step = quantized_step


def add_quantized_control_steps_to_sequence(sequence, quantized_steps):
  assert len(sequence.control_changes) == len(quantized_steps)

  for cc, quantized_step in zip(sequence.control_changes, quantized_steps):
    cc.quantized_step = quantized_step


def parse_test_proto(proto_type, proto_string):
  instance = proto_type()
  text_format.Merge(proto_string, instance)
  return instance
