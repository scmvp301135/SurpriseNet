import numpy as np
import pickle
from constants import Constants

def run():
    # Load data
    BEAT_PER_CHORD = 2
    max_chord_seq = 272 * 2 
    melody_data = np.load('./data/melody_data.npy')

    error = 0

    # Pad 0 to the positions if the length of sequence is smaller than max length   
    for i in range(len(number_96)):
        number_96[i] = np.pad(number_96[i], ((0, max_chord_seq-number_96[i].shape[0]), (0, 0)), constant_values = (0, 0))
    number_96 = np.asarray(number_96)
    for i in range(len(onehot_96)):
        onehot_96[i] = np.pad(onehot_96[i], ((0, max_chord_seq-onehot_96[i].shape[0]), (0, 0)), constant_values = (0, 0))
    onehot_96 = np.asarray(onehot_96)
    for i in range(len(roman)):
        roman[i] = np.pad(roman[i], ((0, max_chord_seq-roman[i].shape[0]), (0, 0)), constant_values = (0, 0))
    roman = np.asarray(roman)
    for i in range(len(roman_onehot)):
        roman_onehot[i] = np.pad(roman_onehot[i], ((0, max_chord_seq-roman_onehot[i].shape[0]), (0, 0)), constant_values = (0, 0))
    roman_onehot = np.asarray(roman_onehot)
    for i in range(len(sec)):
        sec[i] = np.pad(sec[i], ((0, max_chord_seq-sec[i].shape[0]), (0, 0)), constant_values = (0, 0))
    sec = np.asarray(sec)
    for i in range(len(sec_onehot)):
        sec_onehot[i] = np.pad(sec_onehot[i], ((0, max_chord_seq-sec_onehot[i].shape[0]), (0, 0)), constant_values = (0, 0))
    sec_onehot = np.asarray(sec_onehot)
    for i in range(len(borrowed)):
        borrowed[i] = np.pad(borrowed[i], ((0, max_chord_seq-borrowed[i].shape[0]), (0, 0)), constant_values = (0, 0))
    borrowed = np.asarray(borrowed)
    for i in range(len(borrowed_onehot)):
        borrowed_onehot[i] = np.pad(borrowed_onehot[i], ((0, max_chord_seq-borrowed_onehot[i].shape[0]), (0, 0)), constant_values = (0, 0))
    borrowed_onehot = np.asarray(borrowed_onehot)

    # Print shape
    print('shape of number 96 symbol:', number_96.shape)
    print('shape of roman:', roman.shape)
    print('shape of sec:', sec.shape)
    print('shape of borrowed:', borrowed.shape)
    print('shape of onehot 96 symbol:', onehot_96.shape)
    print('shape of roman onehot:', roman_onehot.shape)
    print('shape of sec onehot:', sec_onehot.shape)
    print('shape of borrowed onehot:', borrowed_onehot.shape)

    # Save sequence and one hot data by song
    np.save('number_96_' + str(BEAT_PER_CHORD) + '_beat', number_96)
    np.save('roman_' + str(BEAT_PER_CHORD) + '_beat', roman)
    np.save('sec_' + str(BEAT_PER_CHORD) + '_beat', sec)
    np.save('borrowed_' + str(BEAT_PER_CHORD) + '_beat', borrowed)
    np.save('onehot_96_' + str(BEAT_PER_CHORD) + '_beat', onehot_96)
    np.save('roman_onehot_' + str(BEAT_PER_CHORD) + '_beat', roman_onehot)
    np.save('sec_onehot_' + str(BEAT_PER_CHORD) + '_beat', sec_onehot)
    np.save('borrowed_onehot_' + str(BEAT_PER_CHORD) + '_beat', borrowed_onehot)

    print('chord and roman mismatch:', error)

    # Calculate balancing weight array
    def cal_weight(weight):
        total = 0
        for i in range(len(weight)):
            weight[i] = 1 / weight[i]
        for i in range(len(weight)):
            total += weight[i]
        for i in range(len(weight)):
            weight[i] = weight[i] * len(weight) / total

        return np.asarray(weight)

    weight_chord = cal_weight(weight_chord)
    weight_roman = cal_weight(weight_roman)
    weight_sec = cal_weight(weight_sec)
    weight_borrowed = cal_weight(weight_borrowed)

    print('weight_chord: ', weight_chord)
    # print('weight_roman: ', weight_roman)
    # print('weight_sec: ', weight_sec)
    # print('weight_borrowed: ', weight_borrowed)

    np.save('weight_chord_' + str(BEAT_PER_CHORD) + '_beat', weight_chord)
    np.save('weight_roman_' + str(BEAT_PER_CHORD) + '_beat', weight_roman)
    np.save('weight_sec_' + str(BEAT_PER_CHORD) + '_beat', weight_sec)
    np.save('weight_borrowed_' + str(BEAT_PER_CHORD) + '_beat', weight_borrowed)

    # dim 128 Melody data to 12 one hot
    melody = []
    print('change melody to one hot 12...')
    for song in melody_data:
        number_song = []
        for frame in song:
            number = pianoroll2number(frame)
            embedding = number2onehot(number)
            number_song.append(embedding)
        melody.append(number_song)

    melody = np.asarray(melody)
    print('shape of melody:', melody.shape)
    # melody = melody.reshape((18005, 272, 12*24*2))
    melody = melody.reshape((-1, max_chord_seq, 12 * 24 * BEAT_PER_CHORD))
    print('reshape to beat unit:', melody.shape)

    np.save('melody_baseline_' + str(BEAT_PER_CHORD) + '_beat', melody)

## Main
def main():
    run()
    
if __name__ == '__main__':
    main()