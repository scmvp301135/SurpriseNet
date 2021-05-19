import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from abc import abstractmethod

# Dataset
class ChordGenerDataset(Dataset):
    def __init__(self, data_type, all_chords, val_size = 500):
        
        self.data_type = data_type
        self.all_chords = all_chords
        self.val_size = val_size
        self.melody = np.array([])
        self.chord_index = np.array([])
        self.chord_onehot = np.array([])
        self.length = np.array([])
        self.surprise = np.array([])
        self.load()
        
    def __getitem__(self, index):
        
        chord_onehot = torch.from_numpy(self.chord_onehot[index]).float()
        length = torch.from_numpy(self.length[index])
        melody = torch.from_numpy(self.melody[index]).float()
        surprise = torch.from_numpy(self.surprise[index]).float()
        chord_index = torch.from_numpy(self.chord_index[index]).float()
        
        return chord_onehot, length, melody, surprise, chord_index

    def __len__(self):
        return (self.melody.shape[0])
    
    def load(self):
        
        # Load numpy data
        melody = np.load('./data/melody_aligned.npy')
        length = np.expand_dims(np.load('./data/length.npy'), axis=1)
        
        # Load 96 or 633
        if self.all_chords:
            chord_index = np.load('./data/chord_indices.npy')
            chord_onehot = np.load('./data/chord_onehot.npy')
            surprise = np.load('./data/surprise_all_chords.npy')
            
        else:
            chord_index = np.load('./data/chord_indices_96.npy')
            chord_onehot = np.load('./data/chord_onehot_96.npy')
            surprise = np.load('./data/surprise_96.npy')
        
        # Load training or validation
        if self.data_type == 'train':
#             batch_size = 512
#             melody = np.random.randint(128, size = (batch_size, 272, 2 * 12 * 24))
#             chord_onehot = np.random.randint(96, size = (batch_size, 272, 96))
#             length = np.expand_dims(np.random.randint(1,272, size = (batch_size,)), axis=1)
#             surprise = np.random.randint(0,9, size = (batch_size,272,1))
#             chord_idx = np.random.randint(96 ,size = (batch_size, 272, 1))
            
#             print(length)
#             print(length)
            
            #Splitting data
            print('splitting validation set...')
            train_melody = melody[self.val_size:]
            train_chord_onehot = chord_onehot[self.val_size:]
            train_length = length[self.val_size:]
            train_surprise = surprise[self.val_size:]
            train_chord_index = chord_index[self.val_size:]
            
            self.chord_onehot = train_chord_onehot
            self.length = train_length
            self.melody = train_melody
            self.surprise = train_surprise
            self.chord_index = train_chord_index
        
        elif self.data_type == 'validation':
            val_melody = torch.from_numpy(melody[:self.val_size]).float()
            val_chord_onehot = torch.from_numpy(chord_onehot[:self.val_size]).float()
            val_length = torch.from_numpy(length[:self.val_size])
            val_surprise = torch.from_numpy(surprise[:self.val_size]).float()
            val_chord_index = torch.from_numpy(chord_index[:self.val_size]).float()
            
            self.chord_onehot = val_chord_onehot
            self.length = val_length
            self.melody = val_melody
            self.surprise = val_surprise      
            self.chord_index = val_chord_index
    
# Chord Surprise Dataset
# class SurpriseDataset(Dataset):
#     def __init__(self, melody, chord, length, chord_onehot, surprise):
#         self.melody = melody
#         self.chord = chord
#         # (batch,1) -> (batch,1,1)
#         self.length = np.expand_dims(length, axis=1)
#         self.chord_onehot = chord_onehot
#         # (batch,1) -> (batch,1,1)
#         self.surprise = surprise

#     def __getitem__(self, index):
#         x = torch.from_numpy(self.melody[index]).float()
#         y = torch.from_numpy(self.chord[index]).float()
#         l = torch.from_numpy(self.length[index])
#         x2 = torch.from_numpy(self.chord_onehot[index]).float()
#         surprise = torch.from_numpy(self.surprisingness[index]).float()

#         return x, y, l, x2, surprise

#     def __len__(self):
#         return (self.melody.shape[0])