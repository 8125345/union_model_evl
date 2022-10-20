import mido

def match_time_series(notes1, notes2, window=0.1):
    matched = []
    missed = []
    redundant = []
    unknown = []

    notes1 = list(notes1)
    notes2 = list(notes2)

    # set default status: unknown
    for n in notes1:
      n['match_result'] = 'unknown'

    for n1 in notes1:
        t1, p1 = n1['time'], n1['pitch']

        for n2 in notes2:
            if n2.get('match_result'):
                continue

            t2, p2 = n2['time'], n2['pitch']


            if t1 - t2 > window:
                redundant.append(n2)
                n2['match_result'] = 'redundant'

            elif abs(t1 - t2) <= window and p1 == p2:
                matched.append((n1, n2))
                n1['match_result'] = 'matched'
                n2['match_result'] = 'matched'

                diff = t1 - t2
                n1['time_diff'] = diff
                n2['time_diff'] = -diff

                # print(round(diff, 10))
                break

            elif t2 - t1 > window:
                missed.append(n1)
                n1['match_result'] = 'missed'
                break

    # let all unknown status to 'missed'
    for n in notes1:
      if n['match_result'] == 'unknown':
        missed.append(n)
        unknown.append(n)

    result = {
        'notes1': notes1,
        'notes2': notes2,
        'matched': matched,
        'missed': missed,
        'redundant': redundant,
        'unknown': unknown,
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
    ts2 = midi_to_ts(midi2)
    r = match_time_series(ts1, ts2)
    return r


if __name__ == '__main__':
    import sys
    midi1 = mido.MidiFile(sys.argv[1])
    midi2 = mido.MidiFile(sys.argv[2])
    r = compare_midi(midi1, midi2)
    print('matched: %d  missed: %d redundant: %d' % (len(r.get('matched')), len(r.get('missed')), len(r.get('redundant'))))
