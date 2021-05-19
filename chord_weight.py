from tonal import symbol2number96, roman2romantrain, sec2sectrain, borrowed2borrowedtrain, pianoroll2number, number2onehot, \
symbol2onehot96, roman2romantrainonehot, sec2sectrainonehot, borrowed2borrowedtrainonehot
import numpy as np
import pickle

# Load data
melody_data = np.load('./melody_data.npy')

# Chord symbol data per 2 beats
f = open('symbol_data', 'rb')
symbol_data = pickle.load(f)
f.close()

# Scale degree data per 2 beats
f = open('roman_data', 'rb')
roman_data = pickle.load(f)
f.close()

# Secondary data per 2 beats
f = open('sec_data', 'rb')
sec_data = pickle.load(f)
f.close()

# Borrowed data per 2 beats
f = open('borrowed_data', 'rb')
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
# weight_chord = [1000 for i in range(96)]
weight_chord = [10000 for i in range(96)]
# weight_roman = [0 for i in range(7)]
# weight_sec = [50000 for i in range(7)]
# weight_borrowed = [200000 for i in range(14)]

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
        
        # else simplifiy data
        else:
            # Simplify chord
            number96 = symbol2number96(song[i])
            onehot96 = symbol2onehot96(song[i])
            
            # Append converted data
            number_song.append(number96)
            onehot_song.append(onehot96)
            weight_chord[number96[0]] += 1
            
            # Update previous one hot
            pre = number96
            onehot_pre = onehot96
            
    
    # Rearrange all data
    number_96.append(np.asarray(number_song))
    onehot_96.append(np.asarray(onehot_song))


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

print('weight_chord: ', weight_chord)
np.save('weight_chord_10000', weight_chord)
