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
        length = np.load('./data/length.npy')
        
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

            #Splitting data
            print('splitting validation set...')
            train_melody = melody[self.val_size:]
            train_chord_onehot = chord_onehot[self.val_size:]
            train_length = length[self.val_size:]
            train_surprise = surprise[self.val_size:]
            train_chord_index = chord_index[self.val_size:]
            
            self.chord_onehot = train_chord_onehot
            self.length = np.expand_dims(train_length, axis=1)
            self.melody = train_melody
            self.surprise = train_surprise
            self.chord_index = train_chord_index
        
        elif self.data_type == 'validation':
            val_melody = melody[:self.val_size]
            val_chord_onehot = chord_onehot[:self.val_size]
            val_length = length[:self.val_size]
            val_surprise = surprise[:self.val_size]
            val_chord_index = chord_index[:self.val_size]
            
            self.chord_onehot = val_chord_onehot
            self.length = np.expand_dims(val_length, axis=1)
            self.melody = val_melody
            self.surprise = val_surprise      
            self.chord_index = val_chord_index
