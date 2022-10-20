"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import glob
import logging
import traceback
from shutil import copyfile

import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy
import csv

# import constants as music_constants
# import midi_io
# from protobuf import music_pb2
# import wav2spec
from deepiano.music import constants as music_constants
from deepiano.music import midi_io
from deepiano.protobuf import music_pb2
from deepiano.server import wav2spec


class PitchInfo:
    pitch = 0
    count = 0
    min_prob = 0.0
    max_prob = 0.0
    timestamp = 0

    last_min_prob = 0.0
    last_max_prob = 0.0
    last_timestamp = 0
    new_start = 0

    prev_send_timestamp = 0

    def __init__(self, pitch, count, min_prob, max_prob, timestamp, last_min_prob, last_max_prob, last_timestamp,
                 new_start, prev_send_timestamp):
        self.pitch = pitch
        self.count = count
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.timestamp = timestamp
        self.last_min_prob = last_min_prob
        self.last_max_prob = last_max_prob
        self.last_timestamp = last_timestamp
        self.new_start = new_start,
        self.prev_send_timestamp = prev_send_timestamp


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_path', '../converted_model.tflite',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'hparams',
    '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'client_threshold', 0.01,
    'client_threshold.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_integer(
    'chunk_padding', 3,
    'chunk_padding')
tf.app.flags.DEFINE_string(
    'data_type', 'test',
    'data_type')
tf.app.flags.DEFINE_string('input_dirs', './data/mix',
                           'Directory where the mixed wavs are')
tf.app.flags.DEFINE_string('input_mid_dirs', './data/maestro-v3.0.0',
                            'Directort where the midi labels are')
tf.app.flags.DEFINE_string('output_dir', './data/test/predict',
                           'Directory where the predicted midi & midi labels will be placed.')
tf.app.flags.DEFINE_string('output_client_dir', './data/test/predict_client',
                           'Directory where the predicted midi & midi labels will be placed.')
tf.app.flags.DEFINE_string('output_json_dir', './data/test/predict_json',
                           'output midi json data to file')


ChunkPrediction = collections.namedtuple(
    'ChunkPrediction',
    ('onset_predictions', 'velocity_values'))


def hparams_frames_per_second():
    """Compute frames per second"""
    return 16000 / 512


onset_threshold_list = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def pianoroll_to_note_sequence(chunk_func_c,
                               frames_per_second,
                               velocity=70,
                               instrument=0,
                               program=0,
                               qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                               min_midi_pitch=music_constants.MIN_MIDI_PITCH,
                               wav_file=None):
    frame_length_seconds = 1 / frames_per_second

    sequence_dic = {}
    for th in onset_threshold_list:
        sequence_dic[th] = music_pb2.NoteSequence()
        sequence_dic[th].tempos.add().qpm = qpm
        sequence_dic[th].ticks_per_quarter = music_constants.STANDARD_PPQ

    sequence_client = music_pb2.NoteSequence()
    sequence_client.tempos.add().qpm = qpm
    sequence_client.ticks_per_quarter = music_constants.STANDARD_PPQ

    note_duration = frame_length_seconds * 3  # to remove redundant same midi

    total_frames = FLAGS.chunk_padding  # left padding

    tiny_dict = {}  # {pitch: {PitchInfo}}}
    note_list = []  # result note

    def unscale_velocity(vel):
        unscaled = max(min(vel, 1.), 0) * 80. + 10.
        if math.isnan(unscaled):
            return 0
        return int(unscaled)

    def process_chunk(chunk_prediction):
        nonlocal total_frames

        onset_predictions = chunk_prediction.onset_predictions
        velocity_values = chunk_prediction.velocity_values
        prev_midi_sent_timestamp = 0

        for i, onset in enumerate(onset_predictions):
            for pitch, prob in enumerate(onset):
                if prob <= 0:
                    continue

                pitch = pitch + min_midi_pitch
                pitch_info = tiny_dict.get(pitch)
                if not pitch_info:
                    pitch_info = PitchInfo(pitch, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    tiny_dict[pitch] = pitch_info

                timestamp = (total_frames + i) * frame_length_seconds
                # if timestamp - last_note.get(pitch, -1) <= note_duration:
                #     continue

                min_prob = 0
                max_prob = 0
                is_last = False
                ts = 0

                last_min_prob = 0
                last_max_prob = 0
                last_timestamp = 0

                count = pitch_info.count
                if count == 0:
                    count = 1
                    last_min_prob = pitch_info.last_min_prob
                    last_max_prob = pitch_info.last_max_prob
                    last_timestamp = pitch_info.last_timestamp

                    if last_min_prob > 0 and last_max_prob > 0:
                        min_prob = prob if prob < last_min_prob else last_min_prob
                        if prob >= last_max_prob:
                            max_prob = prob
                            ts = timestamp
                            is_last = True
                        else:
                            max_prob = last_max_prob
                            ts = last_timestamp

                        pitch_info.last_min_prob = 0
                        pitch_info.last_max_prob = 0
                        pitch_info.last_timestamp = 0
                    else:
                        min_prob = max_prob = prob
                        ts = timestamp
                        is_last = True
                else:
                    current_max_prob = pitch_info.max_prob

                    new_start = pitch_info.new_start
                    if prob < current_max_prob and new_start == 1:
                        # if prob > 0.01:
                        pitch_info.count = 1
                        pitch_info.min_prob = prob
                        pitch_info.max_prob = prob
                        pitch_info.timestamp = timestamp
                        continue

                    count += 1
                    pitch_info.new_start = 0

                    current_min_prob = pitch_info.min_prob
                    current_timestamp = pitch_info.timestamp

                    min_prob = prob if prob < current_min_prob else current_min_prob

                    if prob > current_max_prob:
                        max_prob = prob
                        ts = timestamp
                        is_last = True
                    else:
                        max_prob = current_max_prob
                        ts = current_timestamp

                    if count > 5:
                        current_last_min_prob = pitch_info.last_min_prob
                        current_last_max_prob = pitch_info.last_max_prob
                        current_last_timestamp = pitch_info.last_timestamp

                        if current_last_min_prob > 0 and current_last_max_prob > 0:
                            last_min_prob = prob if prob < current_last_min_prob else current_last_min_prob

                            if prob >= current_last_max_prob:
                                last_max_prob = prob
                                last_timestamp = timestamp
                            else:
                                last_max_prob = current_last_max_prob
                                last_timestamp = current_last_timestamp
                        else:
                            last_min_prob = last_max_prob = prob
                            last_timestamp = timestamp

                        pitch_info.last_min_prob = last_min_prob
                        pitch_info.last_max_prob = last_max_prob
                        pitch_info.last_timestamp = last_timestamp

                pitch_info.count = count
                pitch_info.min_prob = min_prob
                pitch_info.max_prob = max_prob
                pitch_info.timestamp = ts

                canWrite = False
                # 可以发送midi的条件
                # 1. 有最大值且最大值且不是最后一个
                # 2. 最小值与最大值之前差异很大(如若干倍)
                # 3. 最大值要大于某个阈值(不能发送太小值)
                # 4. 有时间戳
                # 5. 超过最大阈值时，不管是否最后一个直接发送：为了加速识别速度。可能导致发送多个同音
                if (not is_last or max_prob >= 0.9) and min_prob * 50 < max_prob and max_prob >= FLAGS.client_threshold and ts > 0:
                    canWrite = True

                if canWrite:
                    prev_send_timestamp = pitch_info.prev_send_timestamp
                    delta_time = ts - prev_send_timestamp

                    if delta_time < frame_length_seconds * 1.8:
                        print(format('ignore pitch: %d max_prob: %.10f timestamp: %d priv_ts: %d delta: %d')
                              % (pitch, max_prob, last_min_prob, prev_send_timestamp, delta_time))
                    else:
                        pitch_info.prev_send_timestamp = ts
                        if prev_midi_sent_timestamp > ts:
                            ts = prev_midi_sent_timestamp
                        else:
                            prev_midi_sent_timestamp = ts

                        start_time = (total_frames + i) * frame_length_seconds
                        note = format('{\"pitch\": %d, \"start_time\": %.5f, \"end_time\": %.5f, \"timestamp\":  '
                                      '%.5f, \"prob\": %.10f, \"velocity\": %d}') % (
                                   pitch, start_time, start_time + note_duration, ts,
                                   max_prob,
                                   unscale_velocity(velocity_values[i, pitch] if velocity_values else velocity))

                        note_list.append(note)
                        print('midi event: ', note)

                    pitch_info.count = 1
                    pitch_info.min_prob = prob
                    pitch_info.max_prob = prob
                    pitch_info.timestamp = timestamp

                    pitch_info.last_min_prob = 0
                    pitch_info.last_max_prob = 0
                    pitch_info.last_timestamp = 0
                    pitch_info.new_start = 1
                elif count >= 10:
                    pitch_info.count = 0

        total_frames += len(onset_predictions)

    print('begin process chunk')
    for chunk in chunk_func_c(wav_file):
        process_chunk(chunk)
    print('end process chunk')

    if not os.path.isdir(FLAGS.output_json_dir):
        os.makedirs(FLAGS.output_json_dir)

    #  write note data to file
    if len(note_list) > 0:
        index = wav_file.rindex('/') + 1
        file_name = wav_file[index:len(wav_file)]
        json_path = os.path.join(FLAGS.output_json_dir, file_name[0:-3] + 'json')
        print('json_path: ', json_path)
        with open(json_path, 'w') as json_file:
            json_file.write('[')
            json_file.write(','.join(note_list))
            json_file.write(']')
    
    for i in note_list:
        date = json.loads(i)
        note_client = sequence_client.notes.add()
        note_client.start_time = date["timestamp"]
        note_client.end_time = date["end_time"]
        note_client.pitch = date["pitch"]
        note_client.velocity = date["velocity"]
        note_client.instrument = instrument
        note_client.program = program


        for onset_threshold in onset_threshold_list:
            if date["prob"] > onset_threshold:
                note = sequence_dic[onset_threshold].notes.add()
                note.start_time = date["timestamp"]
                note.end_time = date["end_time"]
                note.pitch = date["pitch"]
                note.velocity = date["velocity"]
                note.instrument = instrument
                note.program = program

    for onset_threshold in onset_threshold_list:
        sequence_dic[onset_threshold].total_time = total_frames * frame_length_seconds
    sequence_client.total_time = total_frames * frame_length_seconds
    return sequence_dic, sequence_client


def generate_predict_set_from_csv(input_dirs):
    predict_file_pairs = []
    logging.info('generate_predict_set_from_csv %s' % input_dirs)
    for input_dir in input_dirs.split(","):
        input_dir = input_dir.strip()
        csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
        for csv_file in csv_files:
            with open(csv_file) as f:
                items = csv.reader(f)
                for item in items:
                    if items.line_num == 1:
                        continue

                    if "maestro-v3.0.0.csv" in csv_file:
                        split = item[2]
                        midi_filename = item[4]
                        audio_filename = item[5]
                    else:
                        split = item[0]
                        midi_filename = item[1]
                        audio_filename = item[2]

                    if split == 'test':
                        wav_file = os.path.join(input_dir, audio_filename)
                        mid_file = os.path.join(FLAGS.input_mid_dirs, midi_filename)
                        logging.info('midi file: %s' % mid_file)
                        if os.path.isfile(mid_file):
                            predict_file_pairs.append((wav_file, mid_file))
    logging.info('generate_predict_set_from_csv %d' % len(predict_file_pairs))
    return predict_file_pairs


def generate_predict_set(input_dirs):
    predict_files = []
    logging.info('generate_predict_set %s' % input_dirs)
    for directory in input_dirs.split(","):
        logging.info('generate_predict_set! path: %s' % directory)
        wav_files = glob.glob(os.path.join(directory, '*.wav'))
        for wav_file in wav_files:
            mid_file = os.path.join(directory, os.path.splitext(os.path.basename(wav_file))[0] + ".mid")
            predict_files.append((wav_file, mid_file))
    logging.info('generate_predict_set! %d' % len(predict_files))
    return predict_files


def transcribe_chunked(argv):
    del argv

    tf.logging.set_verbosity(FLAGS.log)
    tf.logging.info('init...')

    interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    chunk_frames = int(input_details[0]['shape'].tolist()[0] / 229)
    chunk_padding = FLAGS.chunk_padding
    frames_nopadding = chunk_frames - chunk_padding * 2
    assert frames_nopadding > 0

    # print('chunk_frames: %d chunk_padding: %d frames_nopadding: %d' % (chunk_frames, chunk_padding, frames_nopadding))
    print(input_details)
    print(output_details)

    def gen_input(wav_filename):
        spec = wav2spec.wav2spec(wav_filename)
        for i in range(chunk_padding, spec.shape[0], frames_nopadding):
            start = i - chunk_padding
            end = i + frames_nopadding + chunk_padding
            chunk_spec = spec[start:end]
            if chunk_spec.shape[0] == chunk_padding * 2 + frames_nopadding:
                input_item = {
                    'spec': chunk_spec.flatten()
                }
                yield input_item

    def chunk_func(wav_filename):
        # start_time = time.time()
        print(wav_filename)
        for input_item in gen_input(wav_filename):
            interpreter.set_tensor(input_details[0]['index'], input_item['spec'])
            interpreter.invoke()

            onset_probs_flat = interpreter.get_tensor(output_details[0]['index'])
            # velocity_values_flat = interpreter.get_tensor(output_details[1]['index'])

            if chunk_padding > 0:
                onset_probs_flat = onset_probs_flat[chunk_padding:-chunk_padding]
                # velocity_values_flat = velocity_values_flat[chunk_padding:-chunk_padding]

            # onset_predictions = onset_probs_flat > FLAGS.onset_threshold
            # velocity_values = velocity_values_flat

            yield ChunkPrediction(
                onset_predictions=onset_probs_flat,
                velocity_values=None)
        # logging.info('predict time: ', time.time() - start_time)

    if FLAGS.data_type == "all":
        predict_file_pairs = generate_predict_set(FLAGS.input_dirs)
    else:
        predict_file_pairs = generate_predict_set_from_csv(FLAGS.input_dirs)
    logging.info('predict start! %d' % len(predict_file_pairs))

    for th in onset_threshold_list:
        if not os.path.isdir(FLAGS.output_dir+'_%s'%th):
            os.makedirs(FLAGS.output_dir+'_%s'%th)

    if not os.path.isdir(FLAGS.output_client_dir):
        os.makedirs(FLAGS.output_client_dir)

    for wav_file, label_midi_file in predict_file_pairs:
        try:

            sequence_prediction_dic, sequence_prediction_client = pianoroll_to_note_sequence(
                chunk_func,
                frames_per_second=hparams_frames_per_second(),
                min_midi_pitch=21,
                wav_file=wav_file
            )

            _, label_midi_file_name = os.path.split(label_midi_file)
            for th in onset_threshold_list:
                sequence_prediction = sequence_prediction_dic[th]
                copyed_label_midi_file = os.path.join(FLAGS.output_dir+'_%s'%th, label_midi_file_name + '.label.midi')
                copyfile(label_midi_file, copyed_label_midi_file)

                predicted_label_midi_file = os.path.join(FLAGS.output_dir+'_%s'%th, label_midi_file_name + '.predicted.midi')
                midi_io.sequence_proto_to_midi_file(sequence_prediction, predicted_label_midi_file)

            copyed_label_midi_client_file = os.path.join(FLAGS.output_client_dir, label_midi_file_name + '.label.midi')
            copyfile(label_midi_file, copyed_label_midi_client_file)

            predicted_label_midi_client_file = os.path.join(FLAGS.output_client_dir, label_midi_file_name + '.predicted.midi')
            midi_io.sequence_proto_to_midi_file(sequence_prediction_client, predicted_label_midi_client_file)
        except Exception:
            print("Exception wav_file:%s" % wav_file)
            print(traceback.format_exc())

    print('predict end!')


def console_entry_point():
    tf.app.run(transcribe_chunked)


if __name__ == '__main__':
    console_entry_point()
