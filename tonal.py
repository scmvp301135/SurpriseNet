"""
Author
    * Chung En Sun 2020
    * Yi Wei Chen 2021
"""

import math
import numpy as np

def pianoroll2number(pianoroll):
    midi_number = []
    for i in range(128):
        if pianoroll[i] != 0:
            midi_number.append(i)
    
    for i in range(len(midi_number)):
        midi_number[i] = midi_number[i]%12
    
    return midi_number

def number2onehot(number):
    onehot = [0 for i in range(12)]
    for i in number:
        onehot[i] = 1

    return onehot

def pianoroll2base(pianoroll):
    midi_number = []
    for i in range(128):
        if pianoroll[i] != 0:
            midi_number.append(i)
            break
    
    for i in range(len(midi_number)):
        midi_number[i] = midi_number[i]%12
    
    return midi_number

def note2number(notes):
    note2num = {'C':0,
                'C#':1, 'Db':1,
                'D':2,
                'D#': 3, 'Eb': 3,
                'E':4,
                'F':5,
                'F#': 6, 'Gb': 6,
                'G':7,
                'G#': 8, 'Ab': 8,
                'A':9,
                'A#': 10, 'Bb': 10,
                'B':11}
    num = []
    for note in notes:
        num.append(note2num[note])
    return num
def chord962note(chord):
    chord_list = [
                ['C','E','G'], ['C','Eb','G'], ['C','E','G#'], ['C','Eb','Gb'], ['C','F','G'], ['C','E','G','B'], ['C','Eb','G','Bb'], ['C','E','G','Bb'],
                ['Db','F','Ab'], ['C#','E','G#'], ['Db','F','A'], ['C#','E','G'], ['Db','Gb','Ab'], ['Db','F','Ab','C'], ['C#','E','G#','B'], ['Db','F','Ab','B'],
                ['D','F#','A'], ['D','F','A'], ['D','F#','A#'], ['D','F','Ab'], ['D','G','A'], ['D','F#','A','C#'], ['D','F','A','C'], ['D','F#','A','C'],
                ['Eb','G','Bb'], ['Eb','Gb','Bb'], ['Eb','G','B'], ['D#','F#','A'], ['Eb','Ab','Bb'], ['Eb','G','Bb','D'], ['Eb','Gb','Bb','Db'], ['Eb','G','Bb','Db'],
                ['E','G#','B'], ['E','G','B'], ['E','G#','C'], ['E','G','Bb'], ['E','A','B'], ['E','G#','B','D#'], ['E','G','B','D'], ['E','G#','B','D'],
                ['F','A','C'], ['F','Ab','C'], ['F','A','C#'], ['F','Ab','B'], ['F','Bb','C'], ['F','A','C','E'], ['F','Ab','C','Eb'], ['F','A','C','Eb'],
                ['Gb','Bb','Db'], ['F#','A','C#'], ['Gb','Bb','D'], ['F#','A','C'], ['Gb','B','Db'], ['Gb','Bb','Db','F'], ['F#','A','C#','E'], ['Gb','Bb','Db','E'],
                ['G','B','D'], ['G','Bb','D'], ['G','B','D#'], ['G','Bb','Db'], ['G','C','D'], ['G','B','D','F#'], ['G','Bb','D','F'], ['G','B','D','F'],
                ['Ab','C','Eb'], ['G#','B','D#'], ['Ab','C','E'], ['G#','B','D'], ['Ab','Db','Eb'], ['Ab','C','Eb','G'], ['G#','B','D#','F#'], ['Ab','C','Eb','Gb'],
                ['A','C#','E'], ['A','C','E'], ['A','C#','F'], ['A','C','Eb'], ['A','D','E'], ['A','C#','E','G#'], ['A','C','E','G'], ['A','C#','E','G'],
                ['Bb','D','F'], ['Bb','Db','F'], ['Bb','D','F#'], ['Bb','Db','E'], ['Bb','Eb','F'], ['Bb','D','F','A'], ['Bb','Db','F','Ab'], ['Bb','D','F','Ab'],
                ['B','D#','F#'], ['B','D','F#'], ['B','D#','G'], ['B','D','F'], ['B','E','F#'], ['B','D#','F#','A#'], ['B','D','F#','A'], ['B','D#','F#','A'],
    ]
    return chord_list[chord]

def chord482note(chord):
    chord_list = [
                ['C','E','G'], ['C','Eb','G'], ['C','E','G#'], ['C','Eb','Gb'],
                ['Db','F','Ab'], ['C#','E','G#'], ['Db','F','A'], ['C#','E','G'],
                ['D','F#','A'], ['D','F','A'], ['D','F#','A#'], ['D','F','Ab'],
                ['Eb','G','Bb'], ['Eb','Gb','Bb'], ['Eb','G','B'], ['D#','F#','A'],
                ['E','G#','B'], ['E','G','B'], ['E','G#','C'], ['E','G','Bb'],
                ['F','A','C'], ['F','Ab','C'], ['F','A','C#'], ['F','Ab','B'],
                ['Gb','Bb','Db'], ['F#','A','C#'], ['Gb','Bb','D'], ['F#','A','C'],
                ['G','B','D'], ['G','Bb','D'], ['G','B','D#'], ['G','Bb','Db'],
                ['Ab','C','Eb'], ['G#','B','D#'], ['Ab','C','E'], ['G#','B','D'],
                ['A','C#','E'], ['A','C','E'], ['A','C#','F'], ['A','C','Eb'],
                ['Bb','D','F'], ['Bb','Db','F'], ['Bb','D','F#'], ['Bb','Db','E'],
                ['B','D#','F#'], ['B','D','F#'], ['B','D#','G'], ['B','D','F'],
    ]
    return chord_list[chord]

def tonal_centroid(notes):
    fifths_lookup = {9:[1.0, 0.0], 2:[math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)], 7:[math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0:[0.0, 1.0], 5:[math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)], 10:[math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3:[-1.0, 0.0], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 1:[math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6:[0.0, -1.0], 11:[math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    minor_thirds_lookup = {3:[1.0, 0.0], 7:[1.0, 0.0], 11:[1.0, 0.0],
                           0:[0.0, 1.0], 4:[0.0, 1.0], 8:[0.0, 1.0],
                           1:[-1.0, 0.0], 5:[-1.0, 0.0], 9:[-1.0, 0.0],
                           2:[0.0, -1.0], 6:[0.0, -1.0], 10:[0.0, -1.0]}
    major_thirds_lookup = {0:[0.0, 1.0], 3:[0.0, 1.0], 6:[0.0, 1.0], 9:[0.0, 1.0],
                           2:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 5:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 11:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           1:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 7:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 10:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    r1 =1
    r2 =1
    r3 = 0.5
    if notes:
        for note in notes:
            for i in range(2):
                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]
        for i in range(2):
            fifths[i] /= len(notes)
            minor[i] /= len(notes)
            major[i] /= len(notes)

    return fifths + minor + major

def tonal_centroid_base(base):
    fifths_lookup = {9:[1.0, 0.0], 2:[math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)], 7:[math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0:[0.0, 1.0], 5:[math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)], 10:[math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3:[-1.0, 0.0], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 1:[math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6:[0.0, -1.0], 11:[math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    diatonic_lookup = {3:[1.0, 0.0], 2:[math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)], 1:[math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0:[0.0, 1.0], 11:[math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)], 10:[math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     9:[-1.0, 0.0], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 7:[math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6:[0.0, -1.0], 5:[math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    diatonic = [0.0, 0.0]
    r1 = 1
    r2 = 1
    if base:
        for i in range(2):
            fifths[i] = r1 * fifths_lookup[base[0]][i]
            diatonic[i] = r2 * diatonic_lookup[base[0]][i]

    return fifths + diatonic


def chord_function(base):
    chord_function_lookup = {0:[0], 1:[2], 2:[1], 3:[1], 4:[2], 5:[1], 6:[1], 7:[2], 8:[2], 9:[0], 10:[1], 11:[2]}

    function = [-1]
    if base:
        function = chord_function_lookup[base[0]]

    return function

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def neg_dist(x, y):
    dis = 0
    for i, j in zip(x, y):
        dis += math.pow(i-j, 2)
    return -math.sqrt(dis)

def tonal2chord_prob(embedding):
    #order: major, minor. aug, dim, sus, major7, minor7, dominant7
    chord_list = [
                ['C','E','G'], ['C','Eb','G'], ['C','E','G#'], ['C','Eb','Gb'], ['C','F','G'], ['C','E','G','B'], ['C','Eb','G','Bb'], ['C','E','G','Bb'],
                ['Db','F','Ab'], ['C#','E','G#'], ['Db','F','A'], ['C#','E','G'], ['Db','Gb','Ab'], ['Db','F','Ab','C'], ['C#','E','G#','B'], ['Db','F','Ab','B'],
                ['D','F#','A'], ['D','F','A'], ['D','F#','A#'], ['D','F','Ab'], ['D','G','A'], ['D','F#','A','C#'], ['D','F','A','C'], ['D','F#','A','C'],
                ['Eb','G','Bb'], ['Eb','Gb','Bb'], ['Eb','G','B'], ['D#','F#','A'], ['Eb','Ab','Bb'], ['Eb','G','Bb','D'], ['Eb','Gb','Bb','Db'], ['Eb','G','Bb','Db'],
                ['E','G#','B'], ['E','G','B'], ['E','G#','C'], ['E','G','Bb'], ['E','A','B'], ['E','G#','B','D#'], ['E','G','B','D'], ['E','G#','B','D'],
                ['F','A','C'], ['F','Ab','C'], ['F','A','C#'], ['F','Ab','B'], ['F','Bb','C'], ['F','A','C','E'], ['F','Ab','C','Eb'], ['F','A','C','Eb'],
                ['Gb','Bb','Db'], ['F#','A','C#'], ['Gb','Bb','D'], ['F#','A','C'], ['Gb','B','Db'], ['Gb','Bb','Db','F'], ['F#','A','C#','E'], ['Gb','Bb','Db','E'],
                ['G','B','D'], ['G','Bb','D'], ['G','B','D#'], ['G','Bb','Db'], ['G','C','D'], ['G','B','D','F#'], ['G','Bb','D','F'], ['G','B','D','F'],
                ['Ab','C','Eb'], ['G#','B','D#'], ['Ab','C','E'], ['G#','B','D'], ['Ab','Db','Eb'], ['Ab','C','Eb','G'], ['G#','B','D#','F#'], ['Ab','C','Eb','Gb'],
                ['A','C#','E'], ['A','C','E'], ['A','C#','F'], ['A','C','Eb'], ['A','D','E'], ['A','C#','E','G#'], ['A','C','E','G'], ['A','C#','E','G'],
                ['Bb','D','F'], ['Bb','Db','F'], ['Bb','D','F#'], ['Bb','Db','E'], ['Bb','Eb','F'], ['Bb','D','F','A'], ['Bb','Db','F','Ab'], ['Bb','D','F','Ab'],
                ['B','D#','F#'], ['B','D','F#'], ['B','D#','G'], ['B','D','F'], ['B','E','F#'], ['B','D#','F#','A#'], ['B','D','F#','A'], ['B','D#','F#','A'],
    ]

    tonal_list = []
    for chord in chord_list:
        tonal_list.append(tonal_centroid(note2number(chord)))

    dis_list = []
    for i in range(len(tonal_list)):
        dis = neg_dist(embedding, tonal_list[i])
        dis_list.append(dis)

    dis_list = np.asarray(dis_list)
    prob = softmax(dis_list)

    return prob

def tonal2chord_prob48(embedding):
    #order: major, minor. aug, dim, sus, major7, minor7, dominant7
    chord_list = [
                ['C','E','G'], ['C','Eb','G'], ['C','E','G#'], ['C','Eb','Gb'],
                ['Db','F','Ab'], ['C#','E','G#'], ['Db','F','A'], ['C#','E','G'],
                ['D','F#','A'], ['D','F','A'], ['D','F#','A#'], ['D','F','Ab'],
                ['Eb','G','Bb'], ['Eb','Gb','Bb'], ['Eb','G','B'], ['D#','F#','A'],
                ['E','G#','B'], ['E','G','B'], ['E','G#','C'], ['E','G','Bb'],
                ['F','A','C'], ['F','Ab','C'], ['F','A','C#'], ['F','Ab','B'],
                ['Gb','Bb','Db'], ['F#','A','C#'], ['Gb','Bb','D'], ['F#','A','C'],
                ['G','B','D'], ['G','Bb','D'], ['G','B','D#'], ['G','Bb','Db'],
                ['Ab','C','Eb'], ['G#','B','D#'], ['Ab','C','E'], ['G#','B','D'],
                ['A','C#','E'], ['A','C','E'], ['A','C#','F'], ['A','C','Eb'],
                ['Bb','D','F'], ['Bb','Db','F'], ['Bb','D','F#'], ['Bb','Db','E'],
                ['B','D#','F#'], ['B','D','F#'], ['B','D#','G'], ['B','D','F'],
    ]

    tonal_list = []
    for chord in chord_list:
        tonal_list.append(tonal_centroid(note2number(chord)))

    dis_list = []
    for i in range(len(tonal_list)):
        dis = neg_dist(embedding, tonal_list[i])
        dis_list.append(dis)

    dis_list = np.asarray(dis_list)
    prob = softmax(dis_list)

    return prob

def function2chord_prob(function):
    function_list = [0, 0, 0, 0, 0, 0, 0, 0,
                    2, 2, 2, 2, 2, 2, 2, 2,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 2,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 2,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 2]
    chord_prob = []
    alpha0 = 1
    alpha1 = 1.8
    alpha2 = 1
    for f in function_list:
        if f == 0:
            chord_prob.append(function[0] * alpha0)
        if f == 1:
            chord_prob.append(function[1] * alpha1)
        if f ==2:
            chord_prob.append(function[2] * alpha2)

    return np.asarray(chord_prob)

def function2chord_prob_baseline(function):
    function_list = [0, 0, 1, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     2, 1, 1, 1,
                     1, 1, 1, 1,
                     0, 0, 1, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 2]
    chord_prob = []
    alpha0 = 1
    alpha1 = 1.8
    alpha2 = 1
    for f in function_list:
        if f == 0:
            chord_prob.append(function[0] * alpha0)
        if f == 1:
            chord_prob.append(function[1] * alpha1)
        if f == 2:
            chord_prob.append(function[2] * alpha2)

    return np.asarray(chord_prob)

def alter2chord_prob(alter):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    alter_list = [0, 1, 1, 1, 0, 0, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 0, 1, 1, 0, 1, 0, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 0, 1, 1, 0, 1, 0, 1,
                    0, 1, 1, 1, 1, 0, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    0, 1, 1, 1, 0, 0, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 0, 1, 1, 0, 1, 0, 1,
                    1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 0, 1, 1, 1, 1]
    chord_prob = []
    alpha0 = 1
    alpha1 = 1
    for f in alter_list:
        if f == 0:
            chord_prob.append(alter[0] * alpha0)
        if f == 1:
            chord_prob.append(alter[1] * alpha1)

    return np.asarray(chord_prob)

def joint_prob2chord(probs):
    #order: major, minor. aug, dim, sus, major7, minor7, dominant7
    chord_list = [
                ['C','E','G'], ['C','Eb','G'], ['C','E','G#'], ['C','Eb','Gb'], ['C','F','G'], ['C','E','G','B'], ['C','Eb','G','Bb'], ['C','E','G','Bb'],
                ['Db','F','Ab'], ['C#','E','G#'], ['Db','F','A'], ['C#','E','G'], ['Db','Gb','Ab'], ['Db','F','Ab','C'], ['C#','E','G#','B'], ['Db','F','Ab','B'],
                ['D','F#','A'], ['D','F','A'], ['D','F#','A#'], ['D','F','Ab'], ['D','G','A'], ['D','F#','A','C#'], ['D','F','A','C'], ['D','F#','A','C'],
                ['Eb','G','Bb'], ['Eb','Gb','Bb'], ['Eb','G','B'], ['D#','F#','A'], ['Eb','Ab','Bb'], ['Eb','G','Bb','D'], ['Eb','Gb','Bb','Db'], ['Eb','G','Bb','Db'],
                ['E','G#','B'], ['E','G','B'], ['E','G#','C'], ['E','G','Bb'], ['E','A','B'], ['E','G#','B','D#'], ['E','G','B','D'], ['E','G#','B','D'],
                ['F','A','C'], ['F','Ab','C'], ['F','A','C#'], ['F','Ab','B'], ['F','Bb','C'], ['F','A','C','E'], ['F','Ab','C','Eb'], ['F','A','C','Eb'],
                ['Gb','Bb','Db'], ['F#','A','C#'], ['Gb','Bb','D'], ['F#','A','C'], ['Gb','B','Db'], ['Gb','Bb','Db','F'], ['F#','A','C#','E'], ['Gb','Bb','Db','E'],
                ['G','B','D'], ['G','Bb','D'], ['G','B','D#'], ['G','Bb','Db'], ['G','C','D'], ['G','B','D','F#'], ['G','Bb','D','F'], ['G','B','D','F'],
                ['Ab','C','Eb'], ['G#','B','D#'], ['Ab','C','E'], ['G#','B','D'], ['Ab','Db','Eb'], ['Ab','C','Eb','G'], ['G#','B','D#','F#'], ['Ab','C','Eb','Gb'],
                ['A','C#','E'], ['A','C','E'], ['A','C#','F'], ['A','C','Eb'], ['A','D','E'], ['A','C#','E','G#'], ['A','C','E','G'], ['A','C#','E','G'],
                ['Bb','D','F'], ['Bb','Db','F'], ['Bb','D','F#'], ['Bb','Db','E'], ['Bb','Eb','F'], ['Bb','D','F','A'], ['Bb','Db','F','Ab'], ['Bb','D','F','Ab'],
                ['B','D#','F#'], ['B','D','F#'], ['B','D#','G'], ['B','D','F'], ['B','E','F#'], ['B','D#','F#','A#'], ['B','D','F#','A'], ['B','D#','F#','A'],
    ]

    index = np.argmax(probs, axis=0)
    return chord_list[index]

def joint_prob2pianoroll96(probs):
    #order: major, minor. aug, dim, sus, major7, minor7, dominant7
    chord_list = [
                [48,55,64], [48,55,63], [48,56,64], [48,54,63], [48,55,65], [48,55,59,64], [48,55,58,63], [48,55,58,64],
                [49,56,65], [49,56,64], [49,57,65], [49,55,64], [49,56,66], [49,56,60,65], [49,56,59,64], [49,56,59,65],
                [50,57,66], [50,57,65], [50,58,66], [50,56,65], [50,57,67], [50,57,61,66], [50,57,60,65], [50,57,60,66],
                [51,58,67], [51,58,66], [51,59,67], [51,57,66], [51,58,68], [51,58,62,67], [51,58,61,66], [51,58,61,67],
                [52,59,68], [52,59,67], [52,60,68], [52,58,67], [52,59,69], [52,59,63,68], [52,59,62,67], [52,59,62,68],
                [41,48,57], [41,48,56], [41,49,57], [41,47,56], [41,48,58], [41,48,52,57], [41,48,51,56], [41,48,51,57],
                [42,49,58], [42,49,57], [42,50,58], [42,48,57], [42,49,59], [42,49,53,58], [42,49,52,57], [42,49,52,58],
                [43,50,59], [43,50,58], [43,51,59], [43,49,58], [43,50,60], [43,50,54,59], [43,50,53,58], [43,50,53,59],
                [44,51,60], [44,51,59], [44,52,60], [44,50,59], [44,51,61], [44,51,55,60], [44,51,54,59], [44,51,54,60],
                [45,52,61], [45,52,60], [45,53,61], [45,51,60], [45,52,62], [45,52,56,61], [45,52,55,60], [45,52,55,61],
                [46,53,62], [46,53,61], [46,54,62], [46,52,61], [46,53,63], [46,53,57,62], [46,53,56,61], [46,53,56,62],
                [47,54,63], [47,54,62], [47,55,63], [47,53,62], [47,54,64], [47,54,58,63], [47,54,57,62], [47,54,57,63],
    ]
    index = np.argmax(probs, axis=0)
    pianoroll = [0 for i in range(128)]
    for note in chord_list[index]:
        pianoroll[note] = 1
    pianoroll = np.asarray(pianoroll)
    return pianoroll

def joint_prob2pianoroll48(probs):
    #order: major, minor. aug, dim, sus, major7, minor7, dominant7
    chord_list = [
                [48,55,64], [48,55,63], [48,56,64], [48,54,63],
                [49,56,65], [49,56,64], [49,57,65], [49,55,64],
                [50,57,66], [50,57,65], [50,58,66], [50,56,65],
                [51,58,67], [51,58,66], [51,59,67], [51,57,66],
                [52,59,68], [52,59,67], [52,60,68], [52,58,67],
                [41,48,57], [41,48,56], [41,49,57], [41,47,56],
                [42,49,58], [42,49,57], [42,50,58], [42,48,57],
                [43,50,59], [43,50,58], [43,51,59], [43,49,58],
                [44,51,60], [44,51,59], [44,52,60], [44,50,59],
                [45,52,61], [45,52,60], [45,53,61], [45,51,60],
                [46,53,62], [46,53,61], [46,54,62], [46,52,61],
                [47,54,63], [47,54,62], [47,55,63], [47,53,62],
    ]
    index = np.argmax(probs, axis=0)
    pianoroll = [0 for i in range(128)]
    for note in chord_list[index]:
        pianoroll[note] = 1
    pianoroll = np.asarray(pianoroll)
    return pianoroll

def tonal_base2prob(embedding):
    base_list = [[i] for i in range(12)]

    tonal_base_list = []
    for base in base_list:
        tonal_base_list.append(tonal_centroid_base(base))

    min_dis = float("inf")
    index = None
    dis_list = []
    for i in range(len(tonal_base_list)):
        dis = neg_dist(embedding, tonal_base_list[i])
        dis_list.append(dis)

    dis_list = np.asarray(dis_list)
    prob = softmax(dis_list)

    return prob

def function2base_prob(function):
    function_list = [0, 2, 1, 1, 2, 1, 1, 2, 1, 0, 1, 2]

    base_prob = []
    alpha0 = 1
    alpha1 = 1.8
    alpha2 = 1
    for f in function_list:
        if f == 0:
            base_prob.append(function[0] * alpha0)
        if f == 1:
            base_prob.append(function[1] * alpha1)
        if f == 2:
            base_prob.append(function[2] * alpha2)

    return np.asarray(base_prob)

def alter2base_prob(alter):
    alter_list = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

    base_prob = []
    alpha0 = 1
    alpha1 = 1
    for f in alter_list:
        if f == 0:
            base_prob.append(alter[0] * alpha0)
        if f == 1:
            base_prob.append(alter[1] * alpha1)

    return np.asarray(base_prob)

def base_prob2pianoroll(probs):
    base_list = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    index = np.argmax(probs, axis=0)
    pianoroll = [0 for i in range(128)]
    pianoroll[base_list[index]] = 1
    pianoroll = np.asarray(pianoroll)
    return pianoroll

def symbol2number96(symbol):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    order = 0
    char_list = list(symbol)

    if 'o' in char_list or 'ø' in char_list:
        # dim
        order = 4
    else:
        if '#' in char_list and '5' in char_list:
            # aug
            order = 3
        else:
            if 's' in char_list and 'u' in char_list:
                # sus
                order = 5
            else:
                if '7' in char_list or '9' in char_list or '11' in char_list:
                    #7th chord
                    if 'm' in char_list:
                        #major7, minor7
                        if 'j' in char_list:
                            #maj7
                            order = 6
                        else:
                            #minor7
                            order = 7
                    else:
                        #dominant7
                        order = 8
                else:
                    #major. minor
                    if 'm' in char_list and 'j' not in char_list:
                        #minor
                        order = 2
                    else:
                        #major
                        order = 1
    index = 0
    if len(char_list) == 1:
        if char_list[0] == 'A' or char_list[0] == 'a':
            index = 9
        if char_list[0] == 'B' or char_list[0] == 'b':
            index = 11
        if char_list[0] == 'C' or char_list[0] == 'c':
            index = 0
        if char_list[0] == 'D' or char_list[0] == 'd':
            index = 2
        if char_list[0] == 'E' or char_list[0] == 'e':
            index = 4
        if char_list[0] == 'F' or char_list[0] == 'f':
            index = 5
        if char_list[0] == 'G' or char_list[0] == 'g':
            index = 7
    else:
        if char_list[1] == 'b':
            #flat
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 8
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 10
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 1
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 3
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 6
        else:
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 9
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 11
            if char_list[0] == 'C' or char_list[0] == 'c':
                index = 0
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 2
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 4
            if char_list[0] == 'F' or char_list[0] == 'f':
                index = 5
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 7

    return np.asarray([index * 8 + (order - 1)])

def symbol2onehot96(symbol):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    order = 0
    char_list = list(symbol)

    if 'o' in char_list or 'ø' in char_list:
        # dim
        order = 4
    else:
        if '#' in char_list and '5' in char_list:
            # aug
            order = 3
        else:
            if 's' in char_list and 'u' in char_list:
                # sus
                order = 5
            else:
                if '7' in char_list or '9' in char_list or '11' in char_list:
                    #7th chord
                    if 'm' in char_list:
                        #major7, minor7
                        if 'j' in char_list:
                            #maj7
                            order = 6
                        else:
                            #minor7
                            order = 7
                    else:
                        #dominant7
                        order = 8
                else:
                    #major. minor
                    if 'm' in char_list and 'j' not in char_list:
                        #minor
                        order = 2
                    else:
                        #major
                        order = 1
    index = 0
    if len(char_list) == 1:
        if char_list[0] == 'A' or char_list[0] == 'a':
            index = 9
        if char_list[0] == 'B' or char_list[0] == 'b':
            index = 11
        if char_list[0] == 'C' or char_list[0] == 'c':
            index = 0
        if char_list[0] == 'D' or char_list[0] == 'd':
            index = 2
        if char_list[0] == 'E' or char_list[0] == 'e':
            index = 4
        if char_list[0] == 'F' or char_list[0] == 'f':
            index = 5
        if char_list[0] == 'G' or char_list[0] == 'g':
            index = 7
    else:
        if char_list[1] == 'b':
            #flat
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 8
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 10
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 1
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 3
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 6
        else:
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 9
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 11
            if char_list[0] == 'C' or char_list[0] == 'c':
                index = 0
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 2
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 4
            if char_list[0] == 'F' or char_list[0] == 'f':
                index = 5
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 7

    s = [0 for i in range(96)]
    s[index * 8 + (order - 1)] = 1
    return s

def symbol2number48(symbol):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    order = 0
    char_list = list(symbol)

    if 'o' in char_list or 'ø' in char_list:
        # dim
        order = 4
    else:
        if '#' in char_list and '5' in char_list:
            # aug
            order = 3
        else:
            #major. minor
            if 'm' in char_list and 'j' not in char_list:
                #minor
                order = 2
            else:
                #major
                order = 1

    index = 0
    if len(char_list) == 1:
        if char_list[0] == 'A' or char_list[0] == 'a':
            index = 9
        if char_list[0] == 'B' or char_list[0] == 'b':
            index = 11
        if char_list[0] == 'C' or char_list[0] == 'c':
            index = 0
        if char_list[0] == 'D' or char_list[0] == 'd':
            index = 2
        if char_list[0] == 'E' or char_list[0] == 'e':
            index = 4
        if char_list[0] == 'F' or char_list[0] == 'f':
            index = 5
        if char_list[0] == 'G' or char_list[0] == 'g':
            index = 7
    else:
        if char_list[1] == 'b':
            #flat
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 8
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 10
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 1
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 3
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 6
        else:
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 9
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 11
            if char_list[0] == 'C' or char_list[0] == 'c':
                index = 0
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 2
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 4
            if char_list[0] == 'F' or char_list[0] == 'f':
                index = 5
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 7

    return np.asarray([index*4 + (order-1)])

def symbol2onehot48(symbol):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    order = 0
    char_list = list(symbol)

    if 'o' in char_list or 'ø' in char_list:
        # dim
        order = 4
    else:
        if '#' in char_list and '5' in char_list:
            # aug
            order = 3
        else:
            #major. minor
            if 'm' in char_list and 'j' not in char_list:
                #minor
                order = 2
            else:
                #major
                order = 1

    index = 0
    if len(char_list) == 1:
        if char_list[0] == 'A' or char_list[0] == 'a':
            index = 9
        if char_list[0] == 'B' or char_list[0] == 'b':
            index = 11
        if char_list[0] == 'C' or char_list[0] == 'c':
            index = 0
        if char_list[0] == 'D' or char_list[0] == 'd':
            index = 2
        if char_list[0] == 'E' or char_list[0] == 'e':
            index = 4
        if char_list[0] == 'F' or char_list[0] == 'f':
            index = 5
        if char_list[0] == 'G' or char_list[0] == 'g':
            index = 7
    else:
        if char_list[1] == 'b':
            #flat
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 8
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 10
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 1
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 3
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 6
        else:
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 9
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 11
            if char_list[0] == 'C' or char_list[0] == 'c':
                index = 0
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 2
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 4
            if char_list[0] == 'F' or char_list[0] == 'f':
                index = 5
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 7

    s = [0 for i in range(48)]
    s[index * 4 + (order - 1)] = 1
    return s

def chord_function_symbol(index):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    #TODO!!!
    chord_function_lookup = [0, 0, 1, 1, 1, 0, 0, 2,
                             1, 1, 1, 1, 1, 1, 1, 2,
                             1, 1, 1, 1, 1, 1, 1, 2,
                             ]
    return chord_function_lookup[index]

def chord_function_symbol_baseline(number):
    # order: major, minor. aug, dim
    chord_function_lookup = [0, 0, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             2, 1, 1, 1,
                             1, 1, 1, 1,
                             0, 0, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 2]

    return [chord_function_lookup[number[0]]]

def chord_function_symbol_baseline_onehot(number):
    # order: major, minor. aug, dim
    chord_function_lookup = [0, 0, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             2, 1, 1, 1,
                             1, 1, 1, 1,
                             0, 0, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 2]
    function = [0 for i in range(3)]
    function[chord_function_lookup[number[0]]] = 1
    return function

def alter_or_not(number):
    alter = [0]
    if number:
        if 1 in number or 3 in number or 6 in number or 8 in number or 10 in number:
            alter = [1]

    return alter

def roman2romantrain(roman):
    #0 for degree 1 ... 6 for degree 7
    if roman == 'rest':
        return [0], 1
    return [int(roman)-1], 0

def roman2romantrainonehot(roman):
    r = [0 for i in range(7)]
    # 0 for degree 1 ... 6 for degree 7
    if roman == 'rest':
        r[0] = 1
        return r

    r[int(roman)-1] = 1
    return r

def sec2sectrain(sec):
    #0 for no sec
    #1 for offset 1 (sec: 2) ... 6 for offset 6 (sec: 7)
    if sec:
        return [int(sec)-1]
    else:
        return [0]

def sec2sectrainonehot(sec):
    s = [0 for i in range(7)]
    # 0 for no sec
    # 1 for offset 1 (sec: 2) ... 6 for offset 6 (sec: 7)
    if sec:
        s[int(sec) - 1] =1
    else:
        s[0] = 1

    return s
def borrowed2borrowedtrain(borrowed):
    # 6 - S(6) Supermode     (F#) - ######
    #  5 - S(5) Supermode      (B) - #####
    #  4 - S(4) Supermode      (E) - ####
    #  3 - S(3) Supermode      (A) - ###
    #  2 - S(2) Supermode      (D) - ##
    #  1 - Lydian              (G) - #
    #  G - Ionian/Major        (C)
    # B1 - Mixolydian          (F) - b
    # B2 - Dorian             (Bb) - bb
    # B3 - Aeolian/Minor      (Eb) - bbb
    # B4 - Phrygian           (Ab) - bbbb
    # B5 - Locrian            (Db) - bbbbb
    # B6 - ???? Supermode     (Gb) - bbbbbb
    if not borrowed:
        borrowed = 'None'
    else:
        if borrowed == 'b':
            borrowed = '-3'
        else:
            if int(borrowed) < -6:
                borrowed = '-6'
            if int(borrowed) > 6:
                borrowed = '6'
    borrowed_table = {'0':[0], '-1':[1], '-2':[2], '-3':[3], '-4':[4], '-5':[5], '-6':[6],
                      '1': [7], '2':[8], '3':[9], '4':[10], '5':[11], '6':[12], 'None':[13]}
    return borrowed_table[borrowed]

def borrowed2borrowedtrainonehot(borrowed):
    # 6 - S(6) Supermode     (F#) - ######
    #  5 - S(5) Supermode      (B) - #####
    #  4 - S(4) Supermode      (E) - ####
    #  3 - S(3) Supermode      (A) - ###
    #  2 - S(2) Supermode      (D) - ##
    #  1 - Lydian              (G) - #
    #  G - Ionian/Major        (C)
    # B1 - Mixolydian          (F) - b
    # B2 - Dorian             (Bb) - bb
    # B3 - Aeolian/Minor      (Eb) - bbb
    # B4 - Phrygian           (Ab) - bbbb
    # B5 - Locrian            (Db) - bbbbb
    # B6 - ???? Supermode     (Gb) - bbbbbb
    if not borrowed:
        borrowed = 'None'
    else:
        if borrowed == 'b':
            borrowed = '-3'
        else:
            if int(borrowed) < -6:
                borrowed = '-6'
            if int(borrowed) > 6:
                borrowed = '6'
    borrowed_table = {'0':[0], '-1':[1], '-2':[2], '-3':[3], '-4':[4], '-5':[5], '-6':[6],
                      '1': [7], '2':[8], '3':[9], '4':[10], '5':[11], '6':[12], 'None':[13]}
    b = [0 for i in range(14)]
    b[borrowed_table[borrowed][0]] = 1

    return b

def roman2chord_prob(mode, roman, sec, borrowed):
    # alpha = [100, 100, 100]
    # for i in range(len(roman)):
    #     if i == 1 or i == 2 or i == 5 or i == 6:
    #         roman[i] *= alpha[0]
    # for i in range(len(sec)):
    #     if i != 0:
    #         sec[i] *= alpha[1]
    # for i in range(len(borrowed)):
    #     if i != 13:
    #         borrowed[i] *= alpha[2]
    roman_index = np.argmax(roman, axis=0)
    sec_index = np.argmax(sec, axis=0)
    borrowed_index = np.argmax(borrowed, axis=0)
    if not mode:
        mode = '1'

    MODE_TO_KEY = {
        1: 0,
        2: -2,
        3: -4,
        4: 1,
        5: -1,
        6: -3,
        7: -5
    }

    KEY_TO_SCALE = {
        7: [1, 3, 5, 6, 8, 10, 12],  # F#
        6: [1, 3, 5, 6, 8, 10, 11],  # F#
        5: [1, 3, 4, 6, 8, 10, 11],  # B
        4: [1, 3, 4, 6, 8, 9, 11],  # E
        3: [1, 2, 4, 6, 8, 9, 11],  # A
        2: [1, 2, 4, 6, 7, 9, 11],  # D
        1: [0, 2, 4, 6, 7, 9, 11],  # G  (Lydian)
        0: [0, 2, 4, 5, 7, 9, 11],  # C  (Ionian/Major)
        -1: [0, 2, 4, 5, 7, 9, 10],  # F  (Mixolydian)
        -2: [0, 2, 3, 5, 7, 9, 10],  # Bb (Dorian)
        -3: [0, 2, 3, 5, 7, 8, 10],  # Eb (Aeolian/Minor)
        -4: [0, 1, 3, 5, 7, 8, 10],  # Ab (Phrygian)
        -5: [0, 1, 3, 5, 6, 8, 10],  # Db (Locrian)
        -6: [-1, 1, 3, 5, 6, 8, 10],  # Gb
        -7: [-1, 0, 3, 4, 6, 8, 10],  # Gb

        # from michael-jackson/you-are-not-alone/bridge
        'b': [0, 2, 3, 5, 7, 8, 10],  # Eb (Aeolian/Minor)

    }
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    #              0         1       2      3      4       5          6             7
    mode_chord_lookup = {1:[[0, 2, 4, 5], [2*8+1, 2*8+4, 2*8+6], [4*8+1, 4*8+4, 4*8+6], [5*8, 5*8+2, 5*8+4, 5*8+5], [7*8+2, 7*8+4, 7*8+7], [9*8+1, 9*8+4, 9*8+6], [11*8+3]],
                         2:[[1, 4, 6], [2*8+1, 2*8+4, 2*8+6], [3*8, 3*8+2, 3*8+4, 3*8+5], [5*8+2, 5*8+4, 5*8+7], [7*8+1, 7*8+4, 7*8+6], [9*8+3], [10*8, 10*8+2, 10*8+4, 10*8+5]],
                         3:[[1, 4, 6], [1*8, 1*8+2, 1*8+4, 1*8+5], [3*8+2, 3*8+4, 3*8+7], [5*8+1, 5*8+4, 5*8+6], [7*8+3], [8*8, 8*8+2, 8*8+4, 8*8+5], [10*8+1, 10*8+4, 10*8+6]],
                         4:[[0, 2, 4, 5], [2*8+2, 2*8+4, 2*8+7], [4*8+1, 4*8+4, 4*8+6], [6*8+3], [7*8, 7*8+2, 7*8+4, 7*8+5], [9*8+1, 9*8+4, 9*8+6], [11*8+1, 11*8+4, 11*8+6]],
                         5:[[2, 4, 7], [2*8+1, 2*8+4, 2*8+6], [4*8+3], [5*8, 5*8+2, 5*8+4, 5*8+5], [7*8+1, 7*8+4, 7*8+6], [9*8+1, 9*8+4, 9*8+6], [10*8, 10*8+2, 10*8+4, 10*8+5]],
                         6:[[1, 4, 6], [2*8+3], [3*8, 3*8+2, 3*8+4, 3*8+5], [5*8+1, 5*8+4, 5*8+6], [7*8+1, 7*8+4, 7*8+6], [8*8, 8*8+2, 8*8+4, 8*8+5], [10*8+2, 10*8+4, 10*8+7]],
                         7:[[3], [1*8, 1*8+2, 1*8+4, 1*8+5], [3*8+1, 3*8+4, 3*8+6], [5*8+1, 5*8+4, 5*8+6], [6*8, 6*8+2, 6*8+4, 6*8+5], [8*8+2, 8*8+4, 8*8+7], [10*8+1, 10*8+4, 10*8+6]]}
    # borrowed_table = {'0':[0], '-1':[1], '-2':[2], '-3':[3], '-4':[4], '-5':[5], '-6':[6],
    #                   '1': [7], '2':[8], '3':[9], '4':[10], '5':[11], '6':[12], 'None':[13]}
    borrowed_chord_lookup = {0:[[0, 2, 4, 5], [2*8+1, 2*8+4, 2*8+6], [4*8+1, 4*8+4, 4*8+6], [5*8, 5*8+2, 5*8+4, 5*8+5], [7*8+2, 7*8+4, 7*8+7], [9*8+1, 9*8+4, 9*8+6], [11*8+3]],
                             1:[[2, 4, 7], [2*8+1, 2*8+4, 2*8+6], [4*8+3], [5*8, 5*8+2, 5*8+4, 5*8+5], [7*8+1, 7*8+4, 7*8+6], [9*8+1, 9*8+4, 9*8+6], [10*8, 10*8+2, 10*8+4, 10*8+5]],
                             2:[[1, 4, 6], [2*8+1, 2*8+4, 2*8+6], [3*8, 3*8+2, 3*8+4, 3*8+5], [5*8+2, 5*8+4, 5*8+7], [7*8+1, 7*8+4, 7*8+6], [9*8+3], [10*8, 10*8+2, 10*8+4, 10*8+5]],
                             3:[[1, 4, 6], [2*8+3], [3*8, 3*8+2, 3*8+4, 3*8+5], [5*8+1, 5*8+4, 5*8+6], [7*8+1, 7*8+4, 7*8+6], [8*8, 8*8+2, 8*8+4, 8*8+5], [10*8+2, 10*8+4, 10*8+7]],
                             4:[[1, 4, 6], [1*8, 1*8+2, 1*8+4, 1*8+5], [3*8+2, 3*8+4, 3*8+7], [5*8+1, 5*8+4, 5*8+6], [7*8+3], [8*8, 8*8+2, 8*8+4, 8*8+5], [10*8+1, 10*8+4, 10*8+6]],
                             5:[[3], [1*8, 1*8+2, 1*8+4, 1*8+5], [3*8+1, 3*8+4, 3*8+6], [5*8+1, 5*8+4, 5*8+6], [6*8, 6*8+2, 6*8+4, 6*8+5], [8*8+2, 8*8+4, 8*8+7], [10*8+1, 10*8+4, 10*8+6]],
                             6:[[11*8, 11*8+2, 11*8+4, 11*8+5], [1*8+2, 1*8+4, 1*8+7], [3*8+1, 3*8+4, 3*8+6], [5*8+3], [6*8, 6*8+2, 6*8+4, 6*8+5], [8*8+1, 8*8+4, 8*8+6], [10*8+1, 10*8+4, 10*8+6]],
                             7:[[0, 2, 4, 5], [2*8+2, 2*8+4, 2*8+7], [4*8+1, 4*8+4, 4*8+6], [6*8+3], [7*8, 7*8+2, 7*8+4, 7*8+5], [9*8+1, 9*8+4, 9*8+6], [11*8+1, 11*8+4, 11*8+6]],
                             8:[[1*8+3], [2*8, 2*8+2, 2*8+4, 2*8+5], [4*8+1, 4*8+4, 4*8+6], [6*8+1, 6*8+4, 6*8+6], [7*8, 7*8+2, 7*8+4, 7*8+5], [9*8+2, 9*8+4, 9*8+7], [11*8+1, 11*8+4, 11*8+6]],
                             9:[[1*8+1, 1*8+4, 1*8+6], [2*8, 2*8+2, 2*8+4, 2*8+5], [4*8+2, 4*8+4, 4*8+7], [6*8+1, 6*8+4, 6*8+6], [8*8+3], [9*8, 9*8+2, 9*8+4, 9*8+5], [11*8+1, 11*8+4, 11*8+6]],
                             10:[[1*8+1, 1*8+4, 1*8+6], [3*8+3], [4*8, 4*8+2, 4*8+4, 4*8+5], [6*8+1, 6*8+4, 6*8+6], [8*8+1, 8*8+4, 8*8+6], [9*8, 9*8+2, 9*8+4, 9*8+5], [11*8+2, 11*8+4, 11*8+7]],
                             11:[[1*8+1, 1*8+4, 1*8+6], [3*8+1, 3*8+4, 3*8+6], [4*8, 4*8+2, 4*8+4, 4*8+5], [6*8+2, 6*8+4, 6*8+7], [8*8+1, 8*8+4, 8*8+6], [10*8+3], [11*8, 11*8+2, 11*8+4, 11*8+5]],
                             12:[[1*8+2, 1*8+4, 1*8+7], [3*8+1, 3*8+4, 3*8+6], [5*8+3], [6*8, 6*8+2, 6*8+4, 6*8+5], [8*8+1, 8*8+4, 8*8+6], [10*8+1, 10*8+4, 10*8+6], [11*8, 11*8+2, 11*8+4, 11*8+5]]}
    sec_n = 0
    borrowed_n = 0
    normal_n = 0
    result = mode_chord_lookup[int(mode)][roman_index]
    if sec_index != 0:
        scale = KEY_TO_SCALE[MODE_TO_KEY[int(mode)]]
        offset = scale[sec_index]
        for i in range(len(result)):
            result[i] = (result[i] + offset * 8) % 96
        sec_n = 1
    else:
        if borrowed_index != 13:
            result = borrowed_chord_lookup[borrowed_index][roman_index]
            borrowed_n = 1
        else:
            result = mode_chord_lookup[int(mode)][roman_index]
            normal_n = 1

    prob = [0 for i in range(96)]
    for i in result:
        prob[i] = 1

    return np.asarray(prob), normal_n, sec_n, borrowed_n