"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import time
import os
import glob
import logging
import traceback
from shutil import copyfile

#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import csv

import constants as music_constants
import midi_io
import audio_utils
from protobuf import music_pb2
import wav2spec


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
tf.app.flags.DEFINE_string('model_path', './models/forceconcat_202200707_1.tflite',
                           'Path to look for acoustic checkpoints.')
tf.app.flags.DEFINE_string(
    'hparams',
    '',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'client_threshold', 0.01,
    'client_threshold.')
tf.app.flags.DEFINE_string('log', 'INFO', 'The threshold for what messages will be logged: DEBUG, INFO, WARN, ERROR, or FATAL.')

tf.app.flags.DEFINE_integer('front_chunk_padding', 27, 'front_chunk_padding')
tf.app.flags.DEFINE_integer('no_chunk_padding', 2, 'no_chunk_padding')
tf.app.flags.DEFINE_integer('back_chunk_padding', 3, 'back_chunk_padding')
tf.app.flags.DEFINE_integer('delay', 40, 'bgm delay time ms.')

tf.app.flags.DEFINE_string('data_type', 'test', 'data_type')

tf.app.flags.DEFINE_string('input_dirs', './data/bgm',
                           'Directory where the mixed wavs are')
tf.app.flags.DEFINE_string('output_dir', './data/test/predict_noise',
                           'Directory where the predicted midi & midi labels will be placed.')
# tf.app.flags.DEFINE_string('output_json_dir', './data/test/predict_noise_json',
#                            'Directory where the predicted midi & midi labels will be placed.')


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
                               mix_file=None,
                               bgm_file=None):
    frame_length_seconds = 1 / frames_per_second

    sequence_dic = {}
    redundant_dic = {}
    redundant_client_dic = {}
    for th in onset_threshold_list:
        sequence_dic[th] = music_pb2.NoteSequence()
        sequence_dic[th].tempos.add().qpm = qpm
        sequence_dic[th].ticks_per_quarter = music_constants.STANDARD_PPQ
		
        redundant_dic[th] = 0
        redundant_client_dic[th] = 0

    sequence_client = music_pb2.NoteSequence()
    sequence_client.tempos.add().qpm = qpm
    sequence_client.ticks_per_quarter = music_constants.STANDARD_PPQ

    note_duration = frame_length_seconds * 3  # to remove redundant same midi

    total_frames = FLAGS.front_chunk_padding  # left padding

    tiny_dict = {}  # {pitch: {PitchInfo}}}
    note_list = []  # result note

    def unscale_velocity(vel):
        unscaled = max(min(vel, 1.), 0) * 80. + 10.
        if math.isnan(unscaled):
            return 0
        return int(unscaled)

    # redundant_dic[] = 0
    # redundant_client = 0
    def process_chunk(chunk_prediction):
        nonlocal total_frames
        # nonlocal redundant_dic
        # nonlocal redundant_client_dic

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
                # ????????????midi?????????
                # 1. ?????????????????????????????????????????????
                # 2. ???????????????????????????????????????(????????????)
                # 3. ??????????????????????????????(?????????????????????)
                # 4. ????????????
                # 5. ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
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
    for chunk in chunk_func_c(mix_file, bgm_file):
        process_chunk(chunk)
    print('end process chunk')

    # if not os.path.isdir(FLAGS.output_json_dir):
    #     os.makedirs(FLAGS.output_json_dir)

    # #  write note data to file
    # if len(note_list) > 0:
    #     index = mix_file.rindex('/') + 1
    #     file_name = mix_file[index:len(mix_file)]
    #     json_path = os.path.join(FLAGS.output_json_dir, file_name[0:-3] + 'json')
    #     print('json_path: ', json_path)
    #     with open(json_path, 'w') as json_file:
    #         json_file.write('[')
    #         json_file.write(','.join(note_list))
    #         json_file.write(']')

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
                redundant_dic[onset_threshold] += 1

    for onset_threshold in onset_threshold_list:
        sequence_dic[onset_threshold].total_time = total_frames * frame_length_seconds
    sequence_client.total_time = total_frames * frame_length_seconds
    return sequence_dic, sequence_client, redundant_dic


def generate_predict_set(input_dirs):
    predict_files = []
    logging.info('generate_predict_set %s' % input_dirs)
    for directory in input_dirs.split(","):
        logging.info('generate_predict_set! path: %s' % directory)
        bgm_files = glob.glob(os.path.join(directory, '*_bgm.wav'))
        for bgm_file in bgm_files:
            mix_file = os.path.join(directory, os.path.splitext(os.path.basename(bgm_file))[0][:-4] + ".wav")
            # mid_file = os.path.join(FLAGS.input_mid_dirs, os.path.splitext(os.path.basename(mix_file))[0] + ".mid")
            # if os.path.isfile(mid_file):
            predict_files.append((mix_file, bgm_file))
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

    front_chunk_padding = FLAGS.front_chunk_padding
    frames_nopadding = FLAGS.no_chunk_padding
    back_chunk_padding = FLAGS.back_chunk_padding

    print(input_details)
    print(output_details)

    def gen_input(mix_filename, bgm_filename):
        mix_spec = wav2spec.wav2spec(mix_filename)
        # bgm_spec = wav2spec.wav2spec(bgm_filename)
        delay = FLAGS.delay*16
        bgm_wav = audio_utils.file2arr(bgm_filename)
        bgm_wav = np.concatenate((bgm_wav, np.zeros(delay, dtype=bgm_wav.dtype)))
        bgm_wav = bgm_wav[delay:]
        bgm_spec = audio_utils.wav2spec(bgm_wav)

        for i in range(front_chunk_padding, mix_spec.shape[0], frames_nopadding):
            start = i - front_chunk_padding
            end = i + frames_nopadding + back_chunk_padding
            
            mix_chunk_spec = mix_spec[start:end]
            bgm_chunk_spec = bgm_spec[start:end]

            if bgm_chunk_spec.shape[0]==mix_chunk_spec.shape[0] and mix_chunk_spec.shape[0] == front_chunk_padding + frames_nopadding + back_chunk_padding :
                chunk_line = mix_chunk_spec.reshape((1,-1))
                bgm_chunk_line = bgm_chunk_spec.reshape((1,-1))
                
                concat_line = np.concatenate((bgm_chunk_line, chunk_line), axis=1)
                feature = np.squeeze(concat_line)

                yield feature

    def chunk_func(mix_filename, bgm_filename):
        start_time = time.time()
        for input_item in gen_input(mix_filename, bgm_filename):
            interpreter.set_tensor(input_details[0]['index'], input_item)
            interpreter.invoke()

            onset_probs_flat = interpreter.get_tensor(output_details[0]['index'])
            # velocity_values_flat = interpreter.get_tensor(output_details[1]['index'])

            if front_chunk_padding > 0:
                onset_probs_flat = onset_probs_flat[3:-3]
                # velocity_values_flat = velocity_values_flat[chunk_padding:-chunk_padding]

            # onset_predictions = onset_probs_flat > FLAGS.onset_threshold
            # velocity_values = velocity_values_flat

            yield ChunkPrediction(
                onset_predictions=onset_probs_flat,
                velocity_values=None)
        logging.info('File: %s'%mix_filename)
        logging.info('predict time: %f'% (time.time() - start_time))

    predict_file_pairs = generate_predict_set(FLAGS.input_dirs)
    logging.info('predict start! %d' % len(predict_file_pairs))

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    f_out = open(os.path.join(FLAGS.output_dir, "result.txt"), "w")
    redundants_dic = dict(dict(zip(onset_threshold_list, [0,0,0,0,0,0,0,0,0,0,0])))
    for mix_file, bgm_file in predict_file_pairs:
        try:
            sequence_prediction_dic, _, redundant_dic = pianoroll_to_note_sequence(
                chunk_func,
                frames_per_second=hparams_frames_per_second(),
                min_midi_pitch=21,
                mix_file=mix_file,
                bgm_file=bgm_file
            )

            _, label_midi_file_name = os.path.split(bgm_file)
            for th in onset_threshold_list:
                sequence_prediction = sequence_prediction_dic[th]

                predicted_label_midi_file = os.path.join(FLAGS.output_dir, label_midi_file_name + '_%s.predicted.midi'%th)
                midi_io.sequence_proto_to_midi_file(sequence_prediction, predicted_label_midi_file)
                print("redundant", redundant_dic[th])
                redundants_dic[th] += redundant_dic[th]

            f_out.write(str(mix_file) + "\t" + str(redundant_dic) + "\n")
        except Exception:
            print("Exception wav_file:%s" % mix_file)
            print(traceback.format_exc())
    print("redundants", redundants_dic)
    f_out.write("all: \n")
    for th in onset_threshold_list:
        f_out.write( str(th) + "\t" + str(redundants_dic[th]) + "\n")
    f_out.close()

    print('predict end!')


def console_entry_point():
    tf.app.run(transcribe_chunked)


if __name__ == '__main__':
    console_entry_point()
