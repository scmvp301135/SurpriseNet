from tonal import symbol2number96, roman2romantrain, sec2sectrain, borrowed2borrowedtrain, pianoroll2number, number2onehot, \
symbol2onehot96, roman2romantrainonehot, sec2sectrainonehot, borrowed2borrowedtrainonehot
import numpy as np
import pickle
from tqdm import tqdm

# Load data
# melody_data = np.load('./melody_data.npy')

# Chord symbol data per 2 beats
f = open('./data/symbol_data_2_beat', 'rb')
symbol_data = pickle.load(f)
f.close()

# # Scale degree data per 2 beats
# f = open('roman_data', 'rb')
# roman_data = pickle.load(f)
# f.close()

# # Secondary data per 2 beats
# f = open('sec_data', 'rb')
# sec_data = pickle.load(f)
# f.close()

# # Borrowed data per 2 beats
# f = open('borrowed_data', 'rb')
# borrowed_data = pickle.load(f)
# f.close()

number_96 = []
onehot_96 = []

# Loss weighting array
# weight_chord = [1000 for i in range(96)]
# weight_roman = [0 for i in range(7)]
# weight_sec = [50000 for i in range(7)]
# weight_borrowed = [200000 for i in range(14)]

print('change symbol to number 96 and prepare roman data...')
for song in tqdm(symbol_data):
    
    # Initial lists to append
    number_song = []
    onehot_song = []
    
    # Blank Sequence unit
    pre = np.asarray([0])
    
    # One hot encoding unit (Why is borrowed 13?)
    temp = [0 for i in range(96)]
    temp[0] = 1
    onehot_pre = np.asarray(temp)
    
    for i in range(len(song)):
        
        # if some beat is none, pad 0 to sequence array and one hot array
        if not song[i]:
            number_song.append(pre)
            onehot_song.append(onehot_pre)
        
        # else simplifiy data
        else:
            # Simplify chord
            number96 = symbol2number96(song[i])
            onehot96 = symbol2onehot96(song[i])
            
            # Append converted data
            number_song.append(number96)
            onehot_song.append(onehot96)
    
            # Update previous one hot
            pre = number96
            onehot_pre = onehot96
            
    
    # Rearrange all data
    number_96.append(np.asarray(number_song))
    onehot_96.append(np.asarray(onehot_song))

    
max_chord_seq = 272
BEAT_PER_CHORD = 2

# Pad 0 to the positions if the length of sequence is smaller than max length   
for i in range(len(number_96)):
    number_96[i] = np.pad(number_96[i], ((0, max_chord_seq-number_96[i].shape[0]), (0, 0)), constant_values = (0, 0))
number_96 = np.asarray(number_96)

for i in range(len(onehot_96)):
    onehot_96[i] = np.pad(onehot_96[i], ((0, max_chord_seq-onehot_96[i].shape[0]), (0, 0)), constant_values = (0, 0))
onehot_96 = np.asarray(onehot_96)

# Print shape
print('shape of number 96 symbol:', number_96.shape)
print('shape of onehot 96 symbol:', onehot_96.shape)



# Save sequence and one hot data by song
np.save('number_96_' + str(BEAT_PER_CHORD) + '_beat', number_96)
np.save('onehot_96_' + str(BEAT_PER_CHORD) + '_beat', onehot_96)



