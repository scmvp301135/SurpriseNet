from tonal import symbol2number96, roman2romantrain, sec2sectrain, borrowed2borrowedtrain, pianoroll2number, number2onehot, \
symbol2onehot96, roman2romantrainonehot, sec2sectrainonehot, borrowed2borrowedtrainonehot
import numpy as np
import pickle

# Load data
BEAT_PER_CHORD = 1
max_chord_seq = 272 * 2 
melody_data = np.load('./melody_data_' + str(BEAT_PER_CHORD) + '_beat' + '.npy')

# Chord symbol data per 2 beats
f = open('symbol_data_' + str(BEAT_PER_CHORD) + '_beat', 'rb')
symbol_data = pickle.load(f)
f.close()

# Scale degree data per 2 beats
f = open('roman_data_' + str(BEAT_PER_CHORD) + '_beat', 'rb')
roman_data = pickle.load(f)
f.close()

# Secondary data per 2 beats
f = open('sec_data_' + str(BEAT_PER_CHORD) + '_beat', 'rb')
sec_data = pickle.load(f)
f.close()

# Borrowed data per 2 beats
f = open('borrowed_data_' + str(BEAT_PER_CHORD) + '_beat', 'rb')
borrowed_data = pickle.load(f)
f.close()

number_96 = []
onehot_96 = []
roman = []
roman_onehot = []
sec = []
sec_onehot = []
borrowed = []
borrowed_onehot = []
error = 0

# Loss weighting array
weight_chord = [1000 for i in range(96)]
# weight_chord = [10000 for i in range(96)]
weight_roman = [0 for i in range(7)]
weight_sec = [50000 for i in range(7)]
weight_borrowed = [200000 for i in range(14)]

print('change symbol to number 96 and prepare roman data...')
for song, song_r, song_s, song_b in zip(symbol_data, roman_data, sec_data, borrowed_data):
    
    # Initial lists to append
    number_song = []
    onehot_song = []
    roman_song = []
    onehot_roman = []
    sec_song = []
    onehot_sec = []
    borrowed_song = []
    onehot_borrowed = []
    
    # Blank Sequence unit
    pre = np.asarray([0])
    roman_pre = np.asarray([0])
    sec_pre = np.asarray([0])
    borrowed_pre = np.asarray([13])
    
    # One hot encoding unit (Why is borrowed 13?)
    temp = [0 for i in range(96)]
    temp[0] = 1
    onehot_pre = np.asarray(temp)
    temp = [0 for i in range(7)]
    temp[0] = 1
    onehot_roman_pre = np.asarray(temp)
    temp = [0 for i in range(7)]
    temp[0] = 1
    onehot_sec_pre = np.asarray(temp)
    temp = [0 for i in range(14)]
    temp[13] = 1
    onehot_borrowed_pre = np.asarray(temp)
    
    for i in range(len(song)):
        
        # if some beat is none, pad 0 to sequence array and one hot array
        if not song[i]:
            
            number_song.append(pre)
            onehot_song.append(onehot_pre)
            weight_chord[pre[0]] += 1
            roman_song.append(roman_pre)
            onehot_roman.append(onehot_roman_pre)
            weight_roman[roman_pre[0]] += 1
            sec_song.append(sec_pre)
            onehot_sec.append(onehot_sec_pre)
            weight_sec[sec_pre[0]] += 1
            borrowed_song.append(borrowed_pre)
            onehot_borrowed.append(onehot_borrowed_pre)
            weight_borrowed[borrowed_pre[0]] += 1
        
        # else simplifiy data
        else:
            # Simplify chord
            number96 = symbol2number96(song[i])
            onehot96 = symbol2onehot96(song[i])
            
            # Convert to one hot
            r, e = roman2romantrain(song_r[i])
            onehot_r = roman2romantrainonehot(song_r[i])
            
            s = sec2sectrain(song_s[i])
            onehot_s = sec2sectrainonehot(song_s[i])
            
            b = borrowed2borrowedtrain(song_b[i])
            onehot_b = borrowed2borrowedtrainonehot(song_b[i])
            error += e
            
            # Append converted data
            number_song.append(number96)
            onehot_song.append(onehot96)
            weight_chord[number96[0]] += 1
            roman_song.append(r)
            onehot_roman.append(onehot_r)
            weight_roman[r[0]] += 1
            sec_song.append(s)
            onehot_sec.append(onehot_s)
            weight_sec[s[0]] += 1
            borrowed_song.append(b)
            onehot_borrowed.append(onehot_b)
            weight_borrowed[b[0]] += 1
            
            # Update previous one hot
            pre = number96
            onehot_pre = onehot96
            roman_pre = r
            onehot_roman_pre = onehot_r
            sec_pre = s
            onehot_sec_pre = onehot_s
            borrowed_pre = b
            onehot_borrowed_pre = onehot_b
    
    # Rearrange all data
    number_96.append(np.asarray(number_song))
    onehot_96.append(np.asarray(onehot_song))
    roman.append(np.asarray(roman_song))
    roman_onehot.append(np.asarray(onehot_roman))
    sec.append(np.asarray(sec_song))
    sec_onehot.append(np.asarray(onehot_sec))
    borrowed.append(np.asarray(borrowed_song))
    borrowed_onehot.append(np.asarray(onehot_borrowed))

    
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