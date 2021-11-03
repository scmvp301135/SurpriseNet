from tqdm import tqdm
from tonal import pianoroll2number, joint_prob2pianoroll96
import numpy as np
from pypianoroll import Multitrack, Track
import pypianoroll as pr
from matplotlib import pyplot as plt
import os 
from constants import Constants
from utils import INDICES_TO_PIANOROLL

def argmax2pianoroll_all(joint_prob):
    generate_pianoroll = []
    for song in joint_prob:
        pianoroll = []
        for beat in song:
            max_index = np.argmax(beat, axis=0)
            pianoroll.append(INDICES_TO_PIANOROLL[max_index])
        generate_pianoroll.append(pianoroll)

    generate_pianoroll = np.asarray(generate_pianoroll)
    print('generate_pianoroll shape', generate_pianoroll.shape)
    return generate_pianoroll

# augment chord into frame base
# def sequence2frame_all(generate_pianoroll, groundtruth_pianoroll):
#     print('augment chord into frame base...')
#     generate_pianoroll_frame = []
#     groundtruth_pianoroll_frame = []
#     for gen_song, truth_song in zip(generate_pianoroll, groundtruth_pianoroll):
#         gen_pianoroll = []
#         truth_pianoroll = []
#         for gen_beat, truth_beat in zip(gen_song, truth_song):
#             for i in range(Constants.BEAT_RESOLUTION * Constants.BEAT_PER_CHORD):
#                 gen_pianoroll.append(gen_beat)
#                 truth_pianoroll.append(truth_beat)
#         generate_pianoroll_frame.append(gen_pianoroll)
#         groundtruth_pianoroll_frame.append(truth_pianoroll)

#     generate_pianoroll_frame = np.asarray(generate_pianoroll_frame).astype(int)
#     groundtruth_pianoroll_frame = np.asarray(groundtruth_pianoroll_frame)
#     print('accompany_pianoroll frame shape:', generate_pianoroll_frame.shape)
#     print('groundtruth_pianoroll frame shape:', groundtruth_pianoroll_frame.shape)
#     return generate_pianoroll_frame, groundtruth_pianoroll_frame

# Append argmax index to get pianoroll array
#[batch, beats = 272, chordtypes = 96]
def argmax2pianoroll(joint_prob):
    chord_pianoroll = []
    for song in joint_prob:
        pianoroll = []
        for beat in song:
            pianoroll.append(joint_prob2pianoroll96(beat))
        chord_pianoroll.append(pianoroll)

    chord_pianoroll = np.asarray(chord_pianoroll)

    accompany_pianoroll = chord_pianoroll * 100
    print('accompany_pianoroll shape',chord_pianoroll.shape)
    return accompany_pianoroll

# augment chord into frame base
def sequence2frame(accompany_pianoroll, chord_groundtruth):
    print('augment chord into frame base...')
    accompany_pianoroll_frame = []
    chord_groundtruth_frame = []
    for acc_song, truth_song in zip(accompany_pianoroll, chord_groundtruth):
        acc_pianoroll = []
        truth_pianoroll = []
        for acc_beat, truth_beat in zip(acc_song, truth_song):
            for i in range(Constants.BEAT_RESOLUTION * Constants.BEAT_PER_CHORD):
                acc_pianoroll.append(acc_beat)
                truth_pianoroll.append(truth_beat)
        accompany_pianoroll_frame.append(acc_pianoroll)
        chord_groundtruth_frame.append(truth_pianoroll)

    accompany_pianoroll_frame = np.asarray(accompany_pianoroll_frame).astype(int)
    chord_groundtruth_frame = np.asarray(chord_groundtruth_frame)
    print('accompany_pianoroll frame shape:', accompany_pianoroll_frame.shape)
    print('groundtruth_pianoroll frame shape:', chord_groundtruth_frame.shape)
    return accompany_pianoroll_frame, chord_groundtruth_frame

# write pianoroll
def write_pianoroll(result_dir, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats):

    print('write pianoroll...')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    counter = 0
    for melody_roll, chord_roll, truth_roll, l, tempo, downbeat in tqdm(zip(melody_data, accompany_pianoroll_frame,
                                                                            chord_groundtruth_frame, length, tempos,
                                                                            downbeats), total = len(melody_data)):
        
        melody_roll, chord_roll, truth_roll = melody_roll[:l], chord_roll[:l], truth_roll[:l]

        track1 = Track(pianoroll=melody_roll)
        track2 = Track(pianoroll=chord_roll)
        track3 = Track(pianoroll=truth_roll)

        generate = Multitrack(tracks=[track1, track2], tempo=tempo, downbeat=downbeat, beat_resolution=Constants.BEAT_RESOLUTION)
        truth = Multitrack(tracks=[track1, track3], tempo=tempo, downbeat=downbeat, beat_resolution=Constants.BEAT_RESOLUTION)

        pr.write(generate, result_dir + '/generate_' + str(counter) + '.mid')
        pr.write(truth, result_dir + '/groundtruth_' + str(counter) + '.mid')

        fig, axs = generate.plot()
        plt.savefig(result_dir + '/generate_' + str(counter) + '.png')
        plt.close()
        fig, axs = truth.plot()
        plt.savefig(result_dir + '/groundtruth_' + str(counter) + '.png')
        plt.close()

        counter += 1
    
    print('Finished!')
    

# write one pianoroll at once
def write_one_pianoroll(result_dir, filename, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempo ,downbeat):

    print('write pianoroll...')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    l = length

    melody_roll, chord_roll, truth_roll = melody_data[0][:l], accompany_pianoroll_frame[0][:l], chord_groundtruth_frame[0][:l]

    track1 = Track(pianoroll=melody_roll)
    track2 = Track(pianoroll=chord_roll)
    track3 = Track(pianoroll=truth_roll)

    generate = Multitrack(tracks=[track1, track2], tempo=tempo[0], downbeat=downbeat[0], beat_resolution=Constants.BEAT_RESOLUTION)
    truth = Multitrack(tracks=[track1, track3], tempo=tempo[0], downbeat=downbeat[0], beat_resolution=Constants.BEAT_RESOLUTION)

    pr.write(generate, result_dir + '/generate-' + filename + '.mid')
    pr.write(truth, result_dir + '/groundtruth-' + filename + '.mid')

    fig, axs = generate.plot()
    plt.savefig(result_dir + '/generate-' + filename + '.png')
    plt.close()
    fig, axs = truth.plot()
    plt.savefig(result_dir + '/groundtruth-' + filename + '.png')
    plt.close()

    print('Finished!')