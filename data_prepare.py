import os
import numpy as np
import pypianoroll as pr
import pickle
import json
import math
from tqdm import tqdm
from constants import Constants

class prepare_data():

    def __init__(self, data_path = "datasets"):
        
        # Raw data
        self.melody_pianoroll = []
        self.chord_pianoroll = []
        self.symbols = []
        self.seq_length = []
        self.tempos = []
        self.downbeats = []
        self.max_melody_len = 0
        self.max_chord_len = 0
        self.all_num_chords = 0

        # Converted data
        self.chord_indices = []
        self.chord_onehots = []
        self.chord_weights = []
        self.number96 = []
        self.onehot_96 = []
        self.weight_96 = []

        # Recursive search files
        for root, dirs, files in tqdm(list(os.walk(os.path.join(data_path, "pianoroll")))):
            for file in files:
                if file.endswith(".npz"):
                    # Arrange symbol and roman data paths from pianoroll data 
                    path_to_symbol = os.path.join(root, file).replace("/pianoroll/","/event/")[:-4] + "_symbol_nokey.json"

                    # Read .npz(midi) file 
                    midi = pr.Multitrack(os.path.join(root, file))
                    if len(midi.tracks) == 2:

                        # Extract melody
                        melody = midi.tracks[0]

                        # Get the max length of the melody sequence
                        self.max_melody_len = max(self.max_melody_len, melody.pianoroll.shape[0])

                        # Extract chord
                        chord = midi.tracks[1]
                        chord_list = []
                        for i in range(chord.pianoroll.shape[0]):
                            # Get chord per 2 beats 
                            if i % (Constants.BEAT_RESOLUTION * Constants.BEAT_PER_CHORD) == 0:
                                chord_list.append(chord.pianoroll[i])

                        # Chord to numpy
                        chord_np = np.asarray(chord_list)

                        # Get the max length of the chord sequence
                        self.max_chord_len = max(self.max_chord_len, chord_np.shape[0])

                        # Gather all data to a big list
                        self.melody_pianoroll.append(melody.pianoroll)
                        self.chord_pianoroll.append(chord_np)
                        self.seq_length.append(chord_np.shape[0])
                        self.tempos.append(midi.tempo)
                        self.downbeats.append(midi.downbeat)

                        # Create symbol data if pianoroll data exists
                        # Read nokey_symbol json files 
                        f = open(path_to_symbol)
                        event = json.load(f)
                        event_on = []
                        event_off = []
                        symbol = []
                        
                        # Warping factor to normalize sequences to tempo 4/4 for different time signatures, e.g tempo 6/8, 3/4, ...
                        if int(midi.tempo[0]) != 0 and int(event['metadata']['BPM']) != 0:
                            warping_factor = int(event['metadata']['BPM']) // int(midi.tempo[0])
                        else:
                            warping_factor = 1
                        
                        # Extract chord per 2 beat
                        for chord in event['tracks']['chord']:
                            if chord != None:
                                event_on.append(chord['event_on'] / warping_factor)
                                event_off.append(chord['event_off'] / warping_factor)
                                symbol.append(chord['symbol'])
            
                        symbol_len = event_on[-1]
                        symbol_list = []
                        q_index = [2 * i for i in range(len(chord_list))]  

                        for i in range(len(q_index)):
                            if q_index[i] in event_on:
                                symbol_list.append(symbol[event_on.index(q_index[i])])
                            else:
                                if i == q_index[-1]:
                                    symbol_list.append(symbol[-1])

                                else:
                                    count = 0
                                    for k in range(len(symbol)):
                                        if q_index[i] > event_on[k] and q_index[i] < event_off[k]:
                                            symbol_list.append(symbol[k])
                                            count += 1
                                            break
                                    if count == 0:
                                        symbol_list.append('')

                        self.symbols.append(symbol_list)
                        f.close()

                        # Check if there is mismatch between melody and chord data
                        if len(symbol_list) != chord_np.shape[0]:
                            print('mismatch!')
                            print(os.path.join(root, file))
                            print('count',count)
                            continue
                        
                        count += 1 

        # Pad 0 to the positions if the length of melody sequence is smaller than max length                    
        for i in tqdm(range(len(self.melody_pianoroll))):
            self.melody_pianoroll[i] = np.pad(self.melody_pianoroll[i], ((0, self.max_melody_len - self.melody_pianoroll[i].shape[0]), (0, 0)), constant_values = (0, 0))

        # Pad 0 to the positions if the length of chord sequence is smaller than max length               
        for i in tqdm(range(len(self.chord_pianoroll))):
            self.chord_pianoroll[i] = np.pad(self.chord_pianoroll[i], ((0, self.max_chord_len - self.chord_pianoroll[i].shape[0]), (0, 0)), constant_values = (0, 0))

        # Convert all lists to np arrays
        print("Converting data to np array...\n")
        self.melody_pianoroll = np.asarray(self.melody_pianoroll)
        self.chord_pianoroll = np.asarray(self.chord_pianoroll)
        self.seq_length = np.asarray(self.seq_length)

        print("Data dimension:")
        print("melody_pianoroll:", self.melody_pianoroll.shape)
        print("chord_pianoroll:", self.chord_pianoroll.shape)
        print("seq_length:", len(self.seq_length))
        print("tempos:", len(self.tempos))
        print("downbeats:", len(self.downbeats))
        print("symbols:", len(self.symbols), "\n")

        print("Converting symbol data...")
        # Symbol chord data to onehots for all chords
        self.chord_indices, self.chord_onehots, self.chord_weights = self.symbol_to_all_onehots()

        # Symbol chord data to onehots for 96 chords
        self.number96, self.onehot_96, self.weight_96 = self.symbol_to_96_onehots()

    # Calculate balancing weight array
    def cal_weight(self, weight):
        total = 0
        for i in range(len(weight)):
            weight[i] = 1 / weight[i]
        for i in range(len(weight)):
            total += weight[i]
        for i in range(len(weight)):
            weight[i] = weight[i] * len(weight) / total

        return np.asarray(weight)

    def chord_weight(self, chord_indices, chord_num):
        # Loss weighting array
        chord_weights = [1000 for i in range(chord_num)]

        for chord_idx_seq in tqdm(chord_indices):

            # Blank Sequence unit
            pre = np.array([0])

            for i in range(len(chord_idx_seq)):
                
                # if some beat is none, pad 0 to sequence array and one hot array
                if chord_idx_seq[i][0] == 0 :
                    chord_weights[pre[0]] += 1
                    
                else:
                    # Simplify chord
                    chord_idx = chord_idx_seq[i]
                    chord_weights[chord_idx[0]] += 1
                    
                    # Update previous one hot
                    pre = chord_idx

        chord_weights = self.cal_weight(chord_weights)
        
        return chord_weights

    def symbol_to_96_onehots(self):

        from utils.utils import symbol_to_number96, symbol_to_onehot96

        number_96 = []
        onehot_96 = []

        print("Converting symbol to indices and onehots for 96 chords...")
        for song in tqdm(self.symbols):
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
                    number96 = symbol_to_number96(song[i])
                    onehot96 = symbol_to_onehot96(song[i])
                    
                    # Append converted data
                    number_song.append(number96)
                    onehot_song.append(onehot96)
            
                    # Update previous one hot
                    pre = number96
                    onehot_pre = onehot96
                    
            # Rearrange all data
            number_96.append(np.asarray(number_song))
            onehot_96.append(np.asarray(onehot_song))

        # Pad 0 to the positions if the length of sequence is smaller than max length   
        for i in range(len(number_96)):
            number_96[i] = np.pad(number_96[i], ((0, Constants.MAX_SEQUENCE_LENGTH - number_96[i].shape[0]), (0, 0)), constant_values = (0, 0))
        number_96 = np.asarray(number_96)

        for i in range(len(onehot_96)):
            onehot_96[i] = np.pad(onehot_96[i], ((0, Constants.MAX_SEQUENCE_LENGTH - onehot_96[i].shape[0]), (0, 0)), constant_values = (0, 0))
        onehot_96 = np.asarray(onehot_96)

        # Chord weight
        print("Calculate weight for 96 chords...")
        weight_96 = self.chord_weight(number_96, Constants.NUM_CHORDS)

        # Print shape
        print("shape of number 96 symbol:", number_96.shape)
        print("shape of onehot 96 symbol:", onehot_96.shape)
        print("shape of weight 96 symbol:", weight_96.shape)

        return number96, onehot_96, weight_96

    def symbol_to_all_onehots(self):
        # List all chords 
        all_chords = []
        for chord in self.symbols:
            all_chords += chord

        all_chords = sorted(list(set(all_chords)))
        self.all_num_chords = len(all_chords)
        print("Total chord numbers:", self.all_num_chords)

        # Chord symbol to indices
        # Pad 0 to the positions if the length of sequence is smaller than max length 
        symbol_to_indices = dict(zip(all_chords,[i for i in range(self.all_num_chords)]))
        indices_to_symbol = dict(zip([i for i in range(self.all_num_chords)], all_chords))
        # Save indices to symbol pairs
        print("Saving indices_to_symbol.pkl...")
        f = open('./utils/indices_to_symbol.pkl' , 'wb')
        pickle.dump(indices_to_symbol, f)
        f.close()

        # Converting symbol
        print("Converting symbol to indices and onehots for all chord types...")
        chord_indices = []
        for symbol in self.symbols:
            chord_index = [symbol_to_indices[s] if s is not '' else 0 for s in symbol]
            chord_indices.append(np.asarray(chord_index))

        for i in range(len(chord_indices)):
            chord_indices[i] = np.pad(chord_indices[i], (0, Constants.MAX_SEQUENCE_LENGTH - chord_indices[i].shape[0]), constant_values = 0)
            
        chord_indices = np.expand_dims(np.asarray(chord_indices), -1)
        chord_indices = chord_indices.astype(int)

        # Chord indices to onehot
        chord_onehots = np.zeros((chord_indices.shape[0], Constants.MAX_SEQUENCE_LENGTH, self.all_num_chords))

        for i in range(chord_indices.shape[0]):
            for t in range(self.seq_length[i]):
                if chord_indices[i][t][0] != -1 and chord_indices[i][t][0] != 0:
                    chord_onehots[i][t][chord_indices[i][t][0]] = 1
        
        # Collect and find the most common ('chord symbol',fingerring) pair data
        chordsymbol2pianoroll = []
        for song_index in tqdm(range(len(self.symbols))):
            symbol = self.symbols[song_index]
            length = len(symbol)
            pianoroll = self.chord_pianoroll[song_index][:length]
            data = [[s,p] for s, p in zip(symbol,pianoroll)]
            unique_index = sorted(np.unique(pianoroll, return_index = True, axis=0)[-1])
            for ui in unique_index:
                if unique_index[-1] < len(symbol) and (len(symbol) <= self.seq_length[song_index] + 1 or len(symbol) >= self.seq_length[song_index] - 1): 
                    chordsymbol2pianoroll.append(data[ui]) 
                else:
                    print('mismatch', song_index)
                    break
                    
        # Find most frequently chords fingerring
        uniq_fingerring = []
        for target_chord_symbol in all_chords:
            duplicate = []
            for i in chordsymbol2pianoroll:
                if i[0] == target_chord_symbol:
                    duplicate.append(i[1])

            duplicate = np.asarray(duplicate)    
            uniq, indices, counts = np.unique(duplicate, return_index =True, return_counts = True,axis=0)
            uniq_fingerring.append(uniq[counts.argmax(axis=-1)])

        symbol_and_fingerring = dict(zip(all_chords, uniq_fingerring))
        ## Save symbol and fingerring pairs
        print("Saving symbol_and_fingerring.pkl...")
        f = open('./utils/symbol_and_fingerring.pkl' , 'wb')
        pickle.dump(symbol_and_fingerring, f)
        f.close()

        # Chord weight
        print("Calculate weight for all chords...")
        chord_weights = self.chord_weight(chord_indices, self.all_num_chords)

        # Print shape
        print("shape of number for all chords", chord_indices.shape)
        print("shape of onehot for all chords:", chord_onehots.shape)
        print("shape of weight for all chords:", chord_weights.shape, "\n")

        return chord_indices, chord_onehots, chord_weights

