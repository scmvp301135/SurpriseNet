import glob
import os
import numpy as np
import pypianoroll as pr
import pickle
import json
import math
import time 
from tqdm import tqdm

def run():
    # Beat unit frame size
    BEAT_RESOLUTION = 24
    BEAT_PER_CHORD = 2

    melody_data = []
    chord_groundtruth = []
    symbol_data = []
    length = []
    tempos = []

    # First beat?
    downbeats = []
    roman_data = []
    sec_data = []
    borrowed_data = []
    mode_data = []
    max_melody_len = 0
    max_chord_len = 0
    max_event_off = 0
    error = 0
    # os.chdir("./lead-sheet-dataset/datasets/pianoroll")
#     count = 0 
    # Recursive search files
    for root, dirs, files in tqdm(list(os.walk("../lead-sheet-dataset/datasets/pianoroll"))):
        for file in files:
            if file.endswith(".npz"):
                ########################### Arrange pianoroll data ################################
#                 print(os.path.join(root, file))
                path_to_symbol = "../lead-sheet-dataset/datasets/event" + os.path.join(root, file)[40:-4] + "_symbol_nokey.json"
                path_to_roman = "../lead-sheet-dataset/datasets/event" + os.path.join(root, file)[40:-4] + "_roman.json"
#                 print(path_to_symbol)
#                 print(path_to_roman)

                ## Read .npz(midi) file 
                midi = pr.Multitrack(os.path.join(root, file))
                if len(midi.tracks) == 2:

                    # Extract melody
                    melody = midi.tracks[0]
#                     print(melody.pianoroll)
#                     print('melody pianoroll shape',melody.pianoroll.shape)

                    # Get the max length of the melody sequence
                    if max_melody_len < melody.pianoroll.shape[0]:
                        max_melody_len = melody.pianoroll.shape[0]

                    # Extract chord
                    chord = midi.tracks[1]
                    chord_list = []
                    for i in range(chord.pianoroll.shape[0]):

                        # Get the chord per 2 beats 
                        if i%(BEAT_RESOLUTION*BEAT_PER_CHORD) == 0:
                            chord_list.append(chord.pianoroll[i])

                    # Chord to numpy
                    chord_np = np.asarray(chord_list)
                    # print(chord_np)
#                     print('chord pianorolll length',chord_np.shape[0])

                    # Get the max length of the chord sequence
                    if max_chord_len < chord_np.shape[0]:
                        max_chord_len = chord_np.shape[0]

                    # Gather all data to a big list
                    melody_data.append(melody.pianoroll)
                    chord_groundtruth.append(chord_np)
                    length.append(chord_np.shape[0])
#                     print('tempo shape',temp.tempo.shape)
                    tempos.append(midi.tempo)
#                     print('downbeat shape',temp.downbeat.shape)
                    downbeats.append(midi.downbeat)

                    ############# Create symbol data if pianoroll data exists ##############
                    ## Read nokey_symbol json files 
                    f = open(path_to_symbol)
                    event = json.load(f)
                    event_on = []
                    event_off = []
                    symbol = []
                    
                    # Warping factor
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

                    symbol_data.append(symbol_list)
                    f.close()

                    ## Read roman json files and do similar operation
                    f = open(path_to_roman)
                    event = json.load(f)
                    mode_data.append(event['metadata']['mode'])
                    event_on = []
                    event_off = []
                    roman = []
                    sec = []
                    borrowed = []

                    for chord in event['tracks']['chord']:
                        if chord != None:
                            event_on.append(math.ceil(chord['event_on'] // warping_factor))
                            event_off.append(math.ceil(chord['event_off'] // warping_factor))
                            roman.append(chord['sd'])
                            sec.append(chord['sec'])
                            borrowed.append(chord['borrowed'])
                    
                    roman_len = event_off[-1]
                    roman_list = [None for i in range(roman_len)]
                    sec_list = [None for i in range(roman_len)]
                    borrowed_list = [None for i in range(roman_len)]
                    

                    if (event_off[-1] // BEAT_PER_CHORD) > max_event_off:
                        max_event_off = event_off[-1] // BEAT_PER_CHORD

                    for i in range(len(roman)):
                        for j in range(event_on[i], event_off[i]):
                            roman_list[j] = roman[i]
                            sec_list[j] = sec[i]
                            borrowed_list[j] = borrowed[i]

                    roman_list = roman_list[::BEAT_PER_CHORD]
                    sec_list = sec_list[::BEAT_PER_CHORD]
                    borrowed_list = borrowed_list[::BEAT_PER_CHORD]
                    roman_data.append(roman_list)
                    sec_data.append(sec_list)
                    borrowed_data.append(borrowed_list)
                    f.close()
                    
                    if symbol_len != roman_len:
                        error += 1
                        
#                     if len(symbol_list) != chord_np.shape[0]:
#                         print('mismatch!')
#                         print(os.path.join(root, file))
#                         print('count',count)
#                         break
                    
#                     count += 1 


#     Pad 0 to the positions if the length of melody sequence is smaller than max length                    
    for i in tqdm(range(len(melody_data))):
        melody_data[i] = np.pad(melody_data[i], ((0, max_melody_len-melody_data[i].shape[0]), (0, 0)), constant_values = (0, 0))

    # Pad 0 to the positions if the length of chord sequence is smaller than max length               
    for i in tqdm(range(len(chord_groundtruth))):
        chord_groundtruth[i] = np.pad(chord_groundtruth[i], ((0, max_chord_len-chord_groundtruth[i].shape[0]), (0, 0)), constant_values = (0, 0))

    # Convert all lists to np arrays
    melody_data = np.asarray(melody_data)
    chord_groundtruth = np.asarray(chord_groundtruth)
    length = np.asarray(length)

    print(melody_data.shape)
    print(chord_groundtruth.shape)
    print(length.shape)

    # Save np arrays 
#     np.save('./data/melody_data_' + str(BEAT_PER_CHORD) + '_beat', melody_data)
#     np.save('./data/chord_groundtruth_' + str(BEAT_PER_CHORD) + '_beat' , chord_groundtruth)
#     np.save('./data/length_' + str(BEAT_PER_CHORD) + '_beat', length)

    # Save as pickle files
#     f = open('./data/tempos_' + str(BEAT_PER_CHORD) + '_beat', 'wb')
#     pickle.dump(tempos, f)
#     f.close()
#     f = open('./data/downbeats_' + str(BEAT_PER_CHORD) + '_beat', 'wb')
#     pickle.dump(downbeats, f)
#     f.close()

    print('max event off:', max_event_off)
    print('len of symbol data:', len(symbol_data))
    f = open('./data/symbol_data_' + str(BEAT_PER_CHORD) + '_beat' , 'wb')
    pickle.dump(symbol_data, f)
    f.close()

    print('len of roman data:' , len(roman_data))
    f = open('./data/roman_data_' + str(BEAT_PER_CHORD) + '_beat' , 'wb')
    pickle.dump(roman_data, f)
    f.close()

    print('len of sec data:', len(sec_data))
    f = open('./data/sec_data_' + str(BEAT_PER_CHORD) + '_beat', 'wb')
    pickle.dump(sec_data, f)
    f.close()

    print('len of borrowed data:', len(borrowed_data))
    f = open('./data/borrowed_data_'+ str(BEAT_PER_CHORD) + '_beat', 'wb')
    pickle.dump(borrowed_data, f)
    f.close()

    print('len of mode data:', len(mode_data))
    f = open('./data/mode_data_'+ str(BEAT_PER_CHORD) + '_beat', 'wb')
    pickle.dump(mode_data, f)
    f.close()

    print('number of len mismatch:', error)
    

## Main
def main():
    ''' 
    Usage:
    python create_dataseet.py -save_model trained 
    '''
    run()
    
if __name__ == '__main__':
    main()
