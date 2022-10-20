"""
分析并对比所有预测结果
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import logging
import mido
import re
import json
import pandas as pd
from shutil import copyfile

import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir',
                           './data/test/predict',
                           'Directory where the predicted midi & labels are')
tf.app.flags.DEFINE_string('output_dir', './data/test/analyse',
                           'Directory where the analyser results will be placed.')
tf.app.flags.DEFINE_enum(
    'mode', 'midi', ['midi', 'json'],
    'which file mode to analyse')
tf.app.flags.DEFINE_string('name', 'predict_2',
                           'name label')


def f1_score(precision, recall):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.0


def match_time_series(notes1, notes2, window=0.1):
    matched = []
    missed = []
    redundant = []
    unknown = []
    missed_note_dict = dict()
    note_dict = dict()
    redundant_note_dict = dict()

    notes1 = list(notes1)
    notes2 = list(notes2)

    # set default status: unknown
    for n in notes1:
        n['match_result'] = 'unknown'
        note_dict[n['pitch']] = note_dict.get(n['pitch'], 0) + 1
        missed_note_dict[n['pitch']] = 0

    for n1 in notes1:
        t1, p1 = n1['time'], n1['pitch']

        for n2 in notes2:
            if n2.get('match_result'):
                continue

            t2, p2 = n2['time'], n2['pitch']

            if t1 - t2 > window:
                redundant.append(n2)
                n2['match_result'] = 'redundant'
                if redundant_note_dict.get(p1) is None:
                    redundant_note_dict[p1] = dict()
                redundant_note_dict[p1][p2] = redundant_note_dict[p1].get(p2, 0) + 1

            elif abs(t1 - t2) <= window:
                if p1 == p2:
                    matched.append((n1, n2))
                    n1['match_result'] = 'matched'
                    n2['match_result'] = 'matched'

                    diff = t1 - t2
                    n1['time_diff'] = diff
                    n2['time_diff'] = -diff
                    break
                else:
                    if redundant_note_dict.get(p1) is None:
                        redundant_note_dict[p1] = dict()
                    redundant_note_dict[p1][p2] = redundant_note_dict[p1].get(p2, 0) + 1

            elif t2 - t1 > window:
                missed.append(n1)
                n1['match_result'] = 'missed'
                missed_note_dict[p1] = missed_note_dict.get(p1, 0) + 1
                break

    # let all unknown status to 'missed'
    for n in notes1:
        if n['match_result'] == 'unknown':
            missed.append(n)
            missed_note_dict[n['pitch']] = missed_note_dict.get(n['pitch'], 0) + 1
            unknown.append(n)

    # let all empty status in label midi to redundant
    for n in notes2:
        if not n.get('match_result'):
            redundant.append(n)

    result = {
        'notes1': notes1,
        'notes2': notes2,
        'matched': matched,
        'missed': missed,
        'redundant': redundant,
        'unknown': unknown,
        'missed_note_dict': missed_note_dict,
        'note_dict': note_dict,
        'redundant_note_dict': redundant_note_dict
    }

    return result


def midi_to_ts(midi):
    total_time = 0
    midi = list(midi)
    for m in midi:
        total_time = m.time = total_time + m.time

    return [{'time': m.time, 'pitch': m.note, 'data': m} for m in midi if m.type == 'note_on' and m.velocity > 0]


def compare_midi(midi1, midi2):
    ts1 = midi_to_ts(midi1)
    print("ts1", ts1)
    ts2 = midi_to_ts(midi2)
    print("ts2", ts2)
    r = match_time_series(ts1, ts2)
    print("r", r)
    return r


def generate_analyse_set(input_dirs):
    analyse_file_pairs = []
    logging.info('generate_analyse_set %s' % input_dirs)
    for directory in input_dirs:
        path = os.path.join(FLAGS.input_dir, directory)
        logging.info('generate_analyse_set! path: %s' % path)
        path = os.path.join(path, '*.predicted.midi')
        predicted_midi_files = glob.glob(path)
        # find matching mid label files
        for midi_file in predicted_midi_files:
            m = re.match(r'(.*).predicted.midi', midi_file)
            if m:
                label_midi_name_root = m.group(1)
                label_mid_file = label_midi_name_root + '.label.midi'
                analyse_file_pairs.append((midi_file, label_mid_file))
    logging.info('generate_analyse_set! %d' % len(analyse_file_pairs))
    return analyse_file_pairs


def generate_analyse_json_set(input_dirs):
    analyse_file_pairs = []
    logging.info('generate_analyse_json_set %s' % input_dirs)
    for directory in input_dirs:
        path = os.path.join(FLAGS.input_dir, directory)
        logging.info('generate_analyse_json_set! path: %s' % path)
        path = os.path.join(path, '*.predicted.json')
        predicted_midi_files = glob.glob(path)
        # find matching json label files
        for midi_file in predicted_midi_files:
            m = re.match(r'(.*).predicted.json', midi_file)
            if m:
                label_midi_name_root = m.group(1)
                label_mid_file = label_midi_name_root + '.label.json'
                analyse_file_pairs.append((midi_file, label_mid_file))
    logging.info('generate_analyse_json_set! %d' % len(analyse_file_pairs))
    return analyse_file_pairs


def get_matched_info(label_results, predicted_results):
    label_scoreNoteIds = []

    fixed_label_results = []
    for n in label_results:
        n_scoreNoteId = n.get('scoreNoteId')
        if not n_scoreNoteId:
            print('n scoreNoteId empty')
        elif n_scoreNoteId in label_scoreNoteIds:
            print('scoreId dup')
        else:
            label_scoreNoteIds.append(n_scoreNoteId)
            fixed_label_results.append(n)

    predict_scoreNoteIds = []
    fixed_predicted_results = []
    for m in predicted_results:
        m_scoreNoteId = m.get('scoreNoteId')
        if not m_scoreNoteId:
            print('m scoreNoteId empty')
        elif m_scoreNoteId in predict_scoreNoteIds:
            print('scoreId dup')
        else:
            predict_scoreNoteIds.append(m_scoreNoteId)
            fixed_predicted_results.append(m)

    matched_num = 0
    for n in fixed_label_results:
        for m in fixed_predicted_results:
            if n.get('scoreNoteId') == m.get('scoreNoteId') and n.get('pitch') == 'right' and m.get('pitch') == 'right':
                matched_num = matched_num + 1

    label_pitch_types_info = {}
    predict_pitch_types_info = {}
    for n in fixed_label_results:
        num = label_pitch_types_info.get('label_' + n.get('pitch'))
        if num:
            label_pitch_types_info['label_' + n.get('pitch')] = num + 1
        else:
            label_pitch_types_info['label_' + n.get('pitch')] = 1

    for m in fixed_predicted_results:
        num = predict_pitch_types_info.get('predict_' + m.get('pitch'))
        if num:
            predict_pitch_types_info['predict_' + m.get('pitch')] = num + 1
        else:
            predict_pitch_types_info['predict_' + m.get('pitch')] = 1

    return matched_num, label_pitch_types_info, predict_pitch_types_info


def get_wrong_info(label_noatations, predicted_notations):
    wrong_num = 0

    label_scoreNoteIds = []
    fixed_label_noatations = []
    for n in label_noatations:
        n_scoreId = n.get('scoreNoteId')
        if not n_scoreId:
            print('n scoreNoteId empty')
        elif n_scoreId in label_scoreNoteIds and n.get('type') == 'wrong':
            print('scoreId dup')
        else:
            label_scoreNoteIds.append(n_scoreId)
            fixed_label_noatations.append(n)

    fixed_predicted_notations = []
    predict_scoreNoteIds = []
    for m in predicted_notations:
        m_scoreId = m.get('scoreNoteId')
        if not m_scoreId:
            print('n scoreNoteId empty')
        elif m_scoreId in predict_scoreNoteIds and m.get('type') == 'wrong':
            print('scoreId dup')
        else:
            predict_scoreNoteIds.append(m_scoreId)
            fixed_predicted_notations.append(m)

    for n in fixed_label_noatations:
        for m in fixed_predicted_notations:
            if n.get('scoreNoteId') == m.get('scoreNoteId') and n.get('type') == 'wrong' and m.get('type') == 'wrong':
                wrong_num = wrong_num + 1

    label_types_info = {}
    predict_types_info = {}

    for n in fixed_label_noatations:
        num = label_types_info.get('label_' + n.get('type'))
        if num:
            label_types_info['label_' + n.get('type')] = num + 1
        else:
            label_types_info['label_' + n.get('type')] = 1

    for m in fixed_predicted_notations:
        num = predict_types_info.get('predict_' + m.get('type'))
        if num:
            predict_types_info['predict_' + m.get('type')] = num + 1
        else:
            predict_types_info['predict_' + m.get('type')] = 1

    return wrong_num, label_types_info, predict_types_info


def parse_json_data(label_json_path, predict_json_path):
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_json_data = json.loads(f.read())

    with open(predict_json_path, 'r', encoding='utf-8') as f:
        predict_json_data = json.loads(f.read())

    if not label_json_data or not predict_json_data:
        print('json data error!')
        return None

    matched_num, label_matched_info, predict_matched_info = get_matched_info(label_json_data.get('results'),
                                                                             predict_json_data.get('results'))
    wrong_num, label_wrong_info, predict_wrong_info = get_wrong_info(label_json_data.get('notations'),
                                                                     predict_json_data.get('notations'))

    missed_matched_num = label_matched_info.get('label_right') - matched_num
    redundant_matched_num = predict_matched_info.get('predict_right') - matched_num

    missed_wrong_num = label_wrong_info.get('label_wrong') - wrong_num
    redundant_wrong_num = predict_wrong_info.get('predict_wrong') - wrong_num

    matched = matched_num + wrong_num
    missed = missed_matched_num + missed_wrong_num
    redundant = redundant_matched_num + redundant_wrong_num

    return {
        'matched': matched,
        'missed': missed,
        'redundant': redundant
    }


def get_midi_info(label_midi, analysed_midi, from_json):
    if from_json:
        r = parse_json_data(label_midi, analysed_midi)
        return r

    midi1 = mido.MidiFile(label_midi)
    midi2 = mido.MidiFile(analysed_midi)
    r = compare_midi(midi1, midi2)

    return {
        'matched': len(r.get('matched')),
        'missed': len(r.get('missed')),
        'redundant': len(r.get('redundant')),
        'label_midi_count': len(r.get('notes1')),
        'predict_midi_count': len(r.get('notes2')),
        "missed_note_dict": r.get("missed_note_dict"),
        'note_dict': r.get('note_dict'),
        'redundant_note_dict': r.get('redundant_note_dict')
    }


def calc_midi_data(f, from_json):
    total_matched_num = 0
    total_missed_num = 0
    total_redundant_num = 0

    file_names = []
    matched_results = []
    missed_results = []
    redundant_results = []

    acc_results = []
    precision_results = []
    recall_results = []
    f1_results = []

    total_note_dict = dict()
    total_missed_note_dict = dict()
    total_missed_rate_dict = dict()
    total_redundant_note_dict = dict()

    total_num = 0
    too_low_infos = []

    total_all_midi_num = 0
    if from_json:
        analyse_file_pairs = generate_analyse_json_set([''])
    else:
        analyse_file_pairs = generate_analyse_set([''])
    for analysed_midi, label_midi in analyse_file_pairs:
        print("label_midi", label_midi)
        r = get_midi_info(label_midi, analysed_midi, from_json)

        matched_num = r.get('matched')
        missed_num = r.get('missed')
        redundant_num = r.get('redundant')

        label_midi_count = r.get('label_midi_count')
        predict_midi_count = r.get('predict_midi_count')

        missed_note_dict = r.get('missed_note_dict')
        for k, v in missed_note_dict.items():
            if k in total_missed_note_dict.keys():
                total_missed_note_dict[k] += v
            else:
                total_missed_note_dict[k] = v

        note_dict = r.get('note_dict')
        for k, v in note_dict.items():
            if k in total_note_dict.keys():
                total_note_dict[k] += v
            else:
                total_note_dict[k] = v

        redundant_note_dict = r.get("redundant_note_dict")
        for k, v in redundant_note_dict.items():
            if k in total_redundant_note_dict.keys():
                for i, j in v.items():
                    if i in total_redundant_note_dict[k].keys():
                        total_redundant_note_dict[k][i] += j
                    else:
                        total_redundant_note_dict[k][i] = j
            else:
                total_redundant_note_dict[k] = v

        all_midi_num = matched_num + missed_num + redundant_num
        print('matched: %d  missed: %d redundant: %d, label_midi_count: %d, predict_midi_count: %d' % (
        matched_num, missed_num, redundant_num, label_midi_count, predict_midi_count))

        if (matched_num + redundant_num) == 0 or (matched_num + missed_num) == 0:
            continue

        acc = matched_num / all_midi_num
        precision = matched_num / (matched_num + redundant_num)
        recall = matched_num / (matched_num + missed_num)
        f1 = f1_score(precision, recall)
        print('Acc: %f Precision: %f Recall: %f  F1:%f ' % (acc,
                                                            precision,
                                                            recall,
                                                            f1))

        f.write('   label midi: %s\n' % label_midi)
        f.write('analysed midi: %s\n' % analysed_midi)
        f.write('  matched num: %d\n' % matched_num)
        f.write('   missed num: %d\n' % missed_num)
        f.write('redundant num: %d\n' % redundant_num)

        f.write('          Acc: %f\n' % acc)
        f.write('    Precision: %f\n' % precision)
        f.write('       Recall: %f\n' % recall)
        f.write('            F1:%f\n' % f1)
        f.write('     note_dict:%s\n' % str(note_dict))
        f.write('missed_note_dict:%s\n' % str(missed_note_dict))
        f.write('redundant_note_dict:%s\n\n' % str(redundant_note_dict))

        total_all_midi_num = total_all_midi_num + all_midi_num
        total_matched_num = total_matched_num + matched_num
        total_missed_num = total_missed_num + missed_num
        total_redundant_num = total_redundant_num + redundant_num

        file_names.append(label_midi)
        matched_results.append(matched_num)
        missed_results.append(missed_num)
        redundant_results.append(redundant_num)
        acc_results.append(acc)
        precision_results.append(precision)
        recall_results.append(recall)
        f1_results.append(f1)
        total_num = total_num + 1

        if recall < 0.9 or precision < 0.9:
            too_low_infos.append(
                (label_midi, analysed_midi, matched_num, missed_num, redundant_num, acc, precision, recall, f1))

    if total_all_midi_num == 0:
        print('Empty!!!!')
        return

    total_acc = total_matched_num / total_all_midi_num
    total_precision = total_matched_num / (total_matched_num + total_redundant_num)
    total_recall = total_matched_num / (total_matched_num + total_missed_num)
    total_f1 = f1_score(total_precision, total_recall)

    print('Total num[%d] too_low_num[%d] acc: %f Precision: %f Recall: %f F1: %f--------------\n' % (total_num,
                                                                                                     len(too_low_infos),
                                                                                                     total_acc,
                                                                                                     total_precision,
                                                                                                     total_recall,
                                                                                                     total_f1))

    f.write('Total num[%d] too_low_num[%d] acc: %f Precision: %f Recall: %f F1: %f--------------\n' % (total_num,
                                                                                                       len(
                                                                                                           too_low_infos),
                                                                                                       total_acc,
                                                                                                       total_precision,
                                                                                                       total_recall,
                                                                                                       total_f1))

    for info in too_low_infos:
        f.write('[LOW]  label midi: %s\n' % info[0])
        f.write('[LOW]  analyse midi: %s\n' % info[1])
        f.write('  matched num: %d\n' % info[2])
        f.write('   missed num: %d\n' % info[3])
        f.write('redundant num: %d\n' % info[4])
        f.write('    Precision: %f\n' % info[6])
        f.write('       Recall: %f\n' % info[7])
        f.write('           F1: %f\n\n' % info[8])

    # to csv
    result = pd.DataFrame()
    result['file'] = file_names
    result['matched'] = matched_results
    result['missed'] = missed_results
    result['redundant'] = redundant_results
    result['acc'] = acc_results
    result['precision'] = precision_results
    result['recall'] = recall_results
    result['f1'] = f1_results
    result.to_csv(FLAGS.output_dir + '/' + FLAGS.name + '_analyse_results.csv', encoding='utf_8_sig')

    save_too_low_info(too_low_infos)

    write_dict_file(os.path.join(FLAGS.output_dir, "total_note_dict"), total_note_dict)
    write_dict_file(os.path.join(FLAGS.output_dir, "total_missed_note_dict"), total_missed_note_dict)
    for k, v in total_missed_note_dict.items():
        total_missed_rate_dict[k] = (total_note_dict[k], total_missed_note_dict[k] / total_note_dict[k])
    write_total_missed_rate_dict(os.path.join(FLAGS.output_dir, "total_missed_rate_dict"), total_missed_rate_dict)
    write_total_redundant_note_dict(os.path.join(FLAGS.output_dir, "total_redundant_note_dict"), total_redundant_note_dict)
    error_type_dict = cal_error_type(total_redundant_note_dict)
    write_dict_file(os.path.join(FLAGS.output_dir, "error_type_dict"), error_type_dict)


def cal_error_type(total_redundant_note_dict):
    errors = 0
    error_type_dict = dict()
    for k, v in total_redundant_note_dict.items():
        for i, j in v.items():
            error_type_dict[i - k] = error_type_dict.get(i - k, 0) + j
            errors += j
    for k, v in error_type_dict.items():
        error_type_dict[k] = v/errors
    return error_type_dict


def write_dict_file(file_name, info_dict):
    sorted_info_dict = sorted(info_dict.items(), key=lambda x: x[1], reverse=True)
    with open(file_name, "w") as f:
        for i in sorted_info_dict:
            f.write(str(i[0]) + "\t" + str(i[1]) + "\n")


def write_total_missed_rate_dict(file_name, info_dict):
    sorted_info_dict = sorted(info_dict.items(), key=lambda x: x[1][1], reverse=True)
    with open(file_name, "w") as f:
        for i in sorted_info_dict:
            f.write(str(i[0]) + "\t" + str(i[1]) + "\n")


def write_total_redundant_note_dict(file_name, info_dict):
    sorted_dict = sorted(info_dict.items(), key=lambda x: len(x[1]), reverse=True)
    with open(file_name, "w") as f:
        for i in sorted_dict:
            sorted_info = sorted(i[1].items(), key=lambda x: x[1], reverse=True)
            f.write(str(i[0]) + "\t" + str(len(sorted_info)) + "\t" + str(sorted_info) + "\n")


def save_too_low_info(too_low_infos):
    label_midi_file_names = []
    predict_midi_file_names = []
    matched_results = []
    missed_results = []
    redundant_results = []

    acc_results = []
    precision_results = []
    recall_results = []
    f1_results = []

    for info in too_low_infos:
        label_midi_file_names.append(os.path.basename(info[0]))
        predict_midi_file_names.append(os.path.basename(info[1]))
        matched_results.append(info[2])
        missed_results.append(info[3])
        redundant_results.append(info[4])
        acc_results.append(info[5])
        precision_results.append(info[6])
        recall_results.append(info[7])
        f1_results.append(info[8])

    result = pd.DataFrame()
    result['file'] = label_midi_file_names
    result['matched'] = matched_results
    result['missed'] = missed_results
    result['redundant'] = redundant_results
    result['acc'] = acc_results
    result['precision'] = precision_results
    result['recall'] = recall_results
    result['f1'] = f1_results
    result.to_csv(FLAGS.output_dir + '/too_low/' + FLAGS.name + '_too_low_analyse_results.csv', encoding='utf_8_sig')


def main(argv):
    del argv
    print(FLAGS.output_dir)

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if not os.path.isdir(FLAGS.output_dir + '/too_low'):
        os.makedirs(FLAGS.output_dir + '/too_low')

    analyse_result_file = os.path.join(FLAGS.output_dir, FLAGS.name + '_analyse_results.txt')
    print(analyse_result_file)
    with open(analyse_result_file, 'w') as f:
        if FLAGS.mode == 'midi':
            calc_midi_data(f, False)
        elif FLAGS.mode == 'json':
            calc_midi_data(f, True)


if __name__ == '__main__':
    tf.app.run(main)
