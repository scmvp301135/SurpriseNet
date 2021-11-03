import numpy as np
from tonal import tonal_centroid, note2number
from constants import Constants
from utils import *

chord_num = Constants.ALL_NUM_CHORDS

def my_argmax(a):
    rows = np.where(a == a.max(axis=1)[:, None])[0]
    rows_multiple_max = rows[:-1][rows[:-1] == rows[1:]]
    my_argmax = a.argmax(axis=1)
    my_argmax[rows_multiple_max] = -1
    return my_argmax

# chord histogram entropy
def CHE_and_CC(chord_sequence):
    chord_index = np.argmax(chord_sequence, axis=1)

    chord_statistics = np.asarray([0 for i in range(chord_num)])
    for i in range(chord_index.shape[0]):
        chord_statistics[chord_index[i]] += 1

    CC = 0
    for i in chord_statistics:
        if i != 0:
            CC += 1
    sequence_length = chord_index.shape[0]
    chord_statistics = chord_statistics / sequence_length
    
    # calculate entropy
    H = 0
    H = sum([H + - p_i * np.log(p_i+1e-6) for p_i in chord_statistics])

    return H, CC

# chord tonal distance
def CTD(chord_sequence):
    chord_index = np.argmax(chord_sequence, axis=1)
    chord_note = []
    for i in chord_index:
        chord_note.append(note2number(pianoroll_to_note(INDICES_TO_PIANOROLL[i])))
        
    y = 0
    for n in range(len(chord_note) - 1):
        y += np.sqrt(np.sum((np.asarray(tonal_centroid(chord_note[n+1])) - np.asarray(tonal_centroid(chord_note[n]))) ** 2))
    
    if (len(chord_note) - 1) == 0:
        return 0
    
    else:
        return y / (len(chord_note) - 1)

# Chord tone to non-chord tone ratio
def CTnCTR(melody_sequence, chord_sequence):
    chord_index = np.argmax(chord_sequence, axis=1)
    chord_note = []

    for i in chord_index:
        chord_note.append(note2number(pianoroll_to_note(INDICES_TO_PIANOROLL[i])))

    melody_sequence = melody_sequence.reshape((melody_sequence.shape[0]*48, 12))
    melody_index = my_argmax(melody_sequence)
    melody_index = melody_index.reshape((-1, 48))

    c = 0
    p = 0
    n = 0
    for melody_m, chord_m in zip(melody_index, chord_note):
        for i in range(len(melody_m)):
            m = melody_m[i]
            if m != -1:
                if m in chord_m:
                    c += 1
                else:
                    n += 1
                    for j in range(i, len(melody_m)):
                        if melody_m[j] != -1:
                            if melody_m[j] != melody_m[i]:
                                if melody_m[j] in chord_m and abs(melody_m[i]-melody_m[j]) <= 2:
                                    p += 1
                                break
    if (c+n) == 0:
        return 0
    return (c+p)/(c+n)


# Pitch consonance score
def PCS(melody_sequence,chord_sequence):
    chord_index = np.argmax(chord_sequence, axis=1)
    chord_note = []
    
    for i in chord_index:
        chord_note.append(note2number(pianoroll_to_note(INDICES_TO_PIANOROLL[i])))

    melody_sequence = melody_sequence.reshape((melody_sequence.shape[0] * 48, 12))
    melody_index = my_argmax(melody_sequence)
    melody_index = melody_index.reshape((-1, 48))

    score = 0
    count = 0
    for melody_m, chord_m in zip(melody_index, chord_note):
        for m in melody_m:
            if m != -1:
                for c in chord_m:
                    # unison, maj, minor 3rd, perfect 5th, maj, minor 6,
                    if abs(m - c) == 0 or abs(m - c) == 3 or abs(m - c) == 4 or abs(m - c) == 7 or abs(m - c) == 8 or abs(m - c) == 9 or abs(m - c) == 5:
                        if abs(m - c) == 5:
                            count += 1
                        else:
                            count += 1
                            score += 1
                    else:
                        count += 1
                        score += -1
    if count == 0:
        return 0
    return score/count

# melody_sequence-chord tonal distance

def MCTD(melody_sequence, chord_sequence):
    chord_index = np.argmax(chord_sequence, axis=1)
    chord_note = []
    for i in chord_index:
        chord_note.append(note2number(pianoroll_to_note(INDICES_TO_PIANOROLL[i])))

    melody_sequence = melody_sequence.reshape((melody_sequence.shape[0] * 48, 12))
    melody_index = my_argmax(melody_sequence)
    melody_index = melody_index.reshape((-1, 48))

    y = 0
    count = 0
    for melody_m, chord_m in zip(melody_index, chord_note):
        for m in melody_m:
            if m != -1:
                y += np.sqrt(np.sum((np.asarray(tonal_centroid([m])) - np.asarray(tonal_centroid(chord_m)))) ** 2)
                count += 1
    if count == 0:
        return 0
    return y/count




