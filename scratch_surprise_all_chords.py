from tonal import pianoroll2number, joint_prob2pianoroll96
from tonal import tonal_centroid, chord482note, chord962note, note2number
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from decode import *
import math
from create_surprisingness import markov_chain
from tqdm import tqdm

# Data loading
device = 'cpu'
melody_framewise = np.load('./data/melody_data.npy')
chord_groundtruth_idx = np.load('./data/chord_groundtruth.npy')

melody = np.load('./data/melody_baseline.npy')
chord = np.load('./data/chord_indices.npy')
chord_onehot = np.load('./data/chord_onehot.npy')
length = np.load('./data/length.npy')

f = open('./data/tempos', 'rb')
tempos = pickle.load(f)
f.close()
f = open('./data/downbeats', 'rb')
downbeats = pickle.load(f)
f.close()

val_size = 500
print('splitting testing set...')
val_melody_framewise = melody_framewise[:val_size]
val_chord_groundtruth_idx = chord_groundtruth_idx[:val_size]

val_chord = torch.from_numpy(chord_onehot[:val_size]).float()
val_melody = torch.from_numpy(melody[:val_size]).float()
val_length = torch.from_numpy(length[:val_size])

val_length, val_melody = val_length.to(device), val_melody.to(device)

## Profile function
class profile_type():
    def __init__(self,type_num,length):
        
        self.norm = -np.log(1e-4)
        self.length = length
        self.profile = None
        
        if type_num == 1:
            self.profile = self.type1()
            
        if type_num == 2:
            self.profile = self.type2()
            
        if type_num == 3:
            self.profile = self.type3()
            
        if type_num == 4:
            self.profile = self.type4()
            
        if type_num == 5:
            self.profile = self.type5()
            
        if type_num == 6:
            self.profile = self.type6()
        
    def type1(self):
        x = torch.arange(0,self.length,1)
        y = self.norm / (1 + torch.exp(-( x - self.length / 2 )))
        y[0] = 0
        return y

    def type2(self):   
        x = torch.arange(0,self.length,1)
        y = - self.norm / (1 + torch.exp(-( x - self.length / 2 ))) + self.norm
        y[0] = 0
        return y

    def type3(self):   
        x = torch.Tensor([0] * self.length)
        x[0] = 0
        return x

    def type4(self):   
        x = torch.Tensor([self.norm] * self.length)
        x[0] = 0
        return x

    def type5(self):   
        mu, sigma = self.length / 2, self.length / 8 # mean and standard deviation
        x = torch.arange(0,self.length + 1,1)
        y = 1 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        max_value = max(y)
        ratio = self.norm / max_value
        y *= ratio
        y[0] = 0
        return y

    def type6(self):   
        mu, sigma = self.length / 2, self.length / 8 # mean and standard deviation
        x = torch.arange(0,self.length + 1,1)
        y = 1 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        max_value = max(y)
        ratio = self.norm / max_value
        y = -y * ratio + self.norm
        return y
    
#     def type7(self):   
#         x = torch.Tensor([0.8] * self.length)
#         return x


# Load all chords model
from model.surprise_CVAE_all_chords import CVAE
device = 'cpu'
print('building model...')
model_surp = CVAE(device = device,
                  hidden_size_factor = 2,
                  latent_size_factor = 4,
                  prenet_size_factor = 1).to(device)

model_surp.load_state_dict(torch.load('output_models/model_surprise_cvae_all_chords_latent64_hidden512_lstmprenet256.pth'))
model_surp.eval()

# Sampling
for type_num in range(6,7):
    true_surprise = np.array([])
    pred_surprise = np.array([])

    for song_index in tqdm(range(len(val_length))):

        ## Surprising profile
            s = profile_type(type_num,val_length[song_index].item()).profile[:val_length[song_index]]

            true_surprise = np.concatenate((true_surprise,s.numpy()))

            pad = nn.ConstantPad2d((0, 272 - s.shape[0]), 0)
            surprise = pad(s).unsqueeze(0).unsqueeze(2)
            surprise = model_surp.surprise_embedding(surprise,val_length[song_index].unsqueeze(0))
            latent_size = 64

            z = torch.randn(1,272,latent_size)
            output, chord_pred = model_surp.decode(z,val_melody[song_index].unsqueeze(0), surprise)
            gen_chord_index = torch.max(chord_pred[0],-1).indices[:val_length[song_index]]
    #         print(gen_chord_index)
            #     print(gen_chord_index.shape)
            gen_chord_seq = torch.max(chord_pred[0],-1).indices.unsqueeze(0).unsqueeze(-1)

            ########## Surprise contours ###########
            all_chords = True
            surprisingness_seq, TM = markov_chain(gen_chord_seq,all_chords).create_surprisingness_seqs()
            pred_surprise = np.concatenate((pred_surprise,surprisingness_seq[0][:val_length[song_index]].squeeze()))
    
    np.save('./true_surprise_type' + str(type_num),true_surprise)
    np.save('./pred_surprise_type' + str(type_num),pred_surprise)

    
# # Load weight all chords model
# device = 'cpu'
# print('building model...')
# model_surp = CVAE(device = device,
#                   hidden_size_factor = 2,
#                   latent_size_factor = 4,
#                   prenet_size_factor = 1).to(device)

# model_weight_surp.load_state_dict(torch.load('output_models/model_weight_surprise_cvae_all_chords_latent64_hidden512_lstmprenet256.pth'))
# model_weight_surp.eval()