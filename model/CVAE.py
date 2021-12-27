"""
Author
    * Yi Wei Chen 2021
"""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, num_chords, melody_dim, surprise_dim, latent_size, hidden_size, num_layers, max_seq_length, dropout, bidirectional):
        super(Encoder, self).__init__()

        self.input_size = num_chords + melody_dim + surprise_dim
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.bidirectional_factor = 2 if bidirectional else 1

        # NN modules
        self.rnn = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.encoder_output2mean = nn.Linear(hidden_size * self.bidirectional_factor, latent_size)
        self.encoder_output2logv = nn.Linear(hidden_size * self.bidirectional_factor, latent_size)
        
    def forward(self, input_chord, length, **conditions):
        
        inputs = [input_chord] + [condition for condition in conditions.values()]
        input_seq = torch.cat(inputs, dim=-1)

        # Pack data to encoder
        packed_x = pack_padded_sequence(input_seq, length, batch_first=True, enforce_sorted=False)
        encoder_output ,_ = self.rnn(packed_x)
        
        # Pad back
        encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True, total_length=self.max_seq_length)

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.encoder_output2mean(encoder_output)
        log_var = self.encoder_output2logv(encoder_output)

        return mu, log_var


class Latent(nn.Module):
    def __init__(self, device):
        super(Latent, self).__init__()

        self.device = device
        
    def forward(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # If cuda
        if torch.cuda.is_available():
            eps = eps.to(self.device)
            
        z = eps * std + mu
        
        return z


class Decoder(nn.Module):
    def __init__(self, num_chords, melody_dim, surprise_dim, latent_size, hidden_size, num_layers, bidirectional, dropout):
        super(Decoder, self).__init__()
        
        self.bidirectional_factor = 2 if bidirectional else 1
        self.input_size = latent_size + melody_dim + surprise_dim
        self.latent_size = latent_size

        # Latent to decoder
        self.latent2decoder_input = nn.Linear(self.input_size, hidden_size // 2)

        self.decoder = nn.LSTM(input_size=hidden_size // 2, 
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               batch_first=True, 
                               dropout=dropout, 
                               bidirectional=bidirectional,
                               )

        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(hidden_size * self.bidirectional_factor, num_chords)

    def forward(self, z, **conditions):

        inputs = [z] + [condition for condition in conditions.values()]
        z = torch.cat(inputs, dim=-1)

        # Latent to hidden 
        decoder_input = self.latent2decoder_input(z)
        decoder_output, _ = self.decoder(decoder_input)
        
        # Reconstruct to one-hot chord
        result = self.outputs2chord(decoder_output)
        
        # Softmax
        softmax = F.softmax(result, dim=-1)
        
        return result, softmax

class AR_Decoder(nn.Module):
    def __init__(self,
                 teacher_forcing, 
                 eps_i,
                 encoder_hidden_size, 
                 conductor_hidden_size,
                 decoder_hidden_size,
                 latent_size, 
                 encoder_num_layer,
                 conductor_num_layer,
                 decoder_num_layer, 
                 prenet_input,
                 prenet_hidden_size,
                 prenet_num_layer,
                 batch_size, 
                 device,
                 hidden_size_facto,
                 num_layers_factor,
                 latent_size_factor,
                 ):
        
        super(AR_Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.device = device

        # Define the conductor and chord decoder
        self.conductor = nn.GRU(input_size=Constants.NUM_CHORDS + prenet_hidden_size * prenet_num_layer * 2, 
                                hidden_size=self.conductor_hidden_size, 
                                num_layers=self.conductor_num_layer, 
                                batch_first=True, 
                                bidirectional=False)
        
        self.conductor_embedding = nn.Sequential(
            nn.Linear(in_features=self.conductor_hidden_size, out_features=self.latent_size),
            nn.Tanh()
        )
        
        self.decoder = nn.GRU(input_size=self.latent_size + Constants.NUM_CHORDS + 2 * 12 * Constants.BEAT_RESOLUTION,
                              hidden_size=self.decoder_hidden_size, 
                              num_layers=self.decoder_num_layer, 
                              batch_first=True, 
                              bidirectional=False)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(self.decoder_hidden_size, Constants.NUM_CHORDS)

    def forward(self, z, melody_token, input_chord_seqs, input_melody_seqs):

        conductor_input = self.latent2conductor_input(z).unsqueeze(1)
        melody_token = melody_token.unsqueeze(1)
        conductor_input = torch.cat([conductor_input,melody_token],dim=-1)
        counter = 0
    
        if self.conductor_num_layer > 1:
            conductor_hidden = self.latent2conductor_hidden(z).unsqueeze(0).repeat(self.conductor_num_layer, 1, 1)
        else:
            conductor_hidden = self.latent2conductor_hidden(z).unsqueeze(0)
        
        # Initial zeros
        chord_token = torch.zeros(self.batch_size, 1, Constants.NUM_CHORDS, device=self.device)
        output_chord_seqs = torch.zeros(self.batch_size, Constants.MAX_SEQUENCE_LENGTH, Constants.NUM_CHORDS, dtype=torch.float, device=self.device)
    
        for i in range(Constants.MAX_SEQUENCE_LENGTH // Constants.CHORDS_PER_BAR):
            embedding, conductor_hidden = self.conductor(conductor_input, conductor_hidden)
            embedding = self.conductor_embedding(embedding)
#             decoder_hidden = conductor_hidden 
            decoder_hidden = torch.randn(self.decoder_num_layer, self.batch_size, self.decoder_hidden_size, device=self.device)
        
            if self.use_teacher_forcing():
                # Concat embedding with the previous chord
                embedding = embedding.expand(self.batch_size, Constants.CHORDS_PER_BAR, embedding.size(2))
                idx = range(i * Constants.CHORDS_PER_BAR, (i + 1) * Constants.CHORDS_PER_BAR)
                decoder_input = torch.cat([embedding, input_chord_seqs[:,idx,:], input_melody_seqs[:,idx,:]], dim=-1)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                chord_token = self.outputs2chord(decoder_output)
                output_chord_seqs[:, idx, :] = chord_token
                chord_token = chord_token[:,-1,:].unsqueeze(1)
                
                counter += Constants.CHORDS_PER_BAR
            else:
                for _ in range(Constants.CHORDS_PER_BAR):
                    # Concat embedding with previous chord        
                    decoder_input = torch.cat([embedding, chord_token, input_melody_seqs[:,counter,:].unsqueeze(1)], dim=-1)
                    # Generate a single note (for each batch)
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    chord_token = self.outputs2chord(decoder_output)
                    output_chord_seqs[:,counter,:] = chord_token.squeeze()
                    
                    counter += 1
        # Softmax
        softmax = F.softmax(output_chord_seqs, dim=-1)
        
        return output_chord_seqs, softmax

class Surprise_Prenet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, dropout, bidirectional):
        
        self.surprise_prenet = nn.LSTM(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional)

    def forward(self, surprise_condition, length):

        # Pack data to encoder
        packed_x = pack_padded_sequence(surprise_condition, length, batch_first=True, enforce_sorted=False)
        prenet_output , (hidden, _) = self.surprise_prenet(packed_x)
        
        # Pad back
        prenet_output, _ = pad_packed_sequence(prenet_output, batch_first=True, total_length=Constants.MAX_SEQUENCE_LENGTH)

        return prenet_output    

 
class CVAE(nn.Module):
    def __init__(self,
                 device,
                 num_chords,
                 surprise_dim=0,
                 max_seq_length=272,
                 melody_dim=576,
                 encoder_hidden_size=256, 
                 decoder_hidden_size=256,
                 latent_size=16, 
                 encoder_num_layers=2,
                 decoder_num_layers=2,
                 bidirectional=True,
                 dropout=0.2,
                 ):
        
        super(CVAE, self).__init__()
        self.device = device
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.latent_size = latent_size
        
        # Encoder
        self.encoder = Encoder(num_chords, 
                               melody_dim, 
                               surprise_dim, 
                               latent_size=latent_size,
                               hidden_size=encoder_hidden_size, 
                               num_layers=encoder_num_layers, 
                               max_seq_length=max_seq_length, 
                               dropout=dropout,
                               bidirectional=bidirectional
                               )

        # Reparameterize to latent code z
        self.latent = Latent(device)

        # Decoder
        self.decoder = Decoder(num_chords, 
                               melody_dim, 
                               surprise_dim, 
                               latent_size, 
                               hidden_size=decoder_hidden_size, 
                               num_layers=decoder_num_layers, 
                               dropout=dropout,
                               bidirectional=bidirectional
                              )
    
    def forward(self, input_chord, length, melody_condition):
        
        # Encode
        mu, log_var = self.encoder(input_chord, length, melody_condition=melody_condition)
        
        # Reparameterize
        z = self.latent(mu, log_var)
        
        # Decode
        output, softmax = self.decoder(z, melody_condition=melody_condition)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax, logp, mu, log_var, input_chord


class MusicCVAE(nn.Module):
    def __init__(self,
                 encoder_hidden_size,
                 decoder_hidden_size,
                 latent_size,
                 encoder_num_layers,
                 decoder_num_layers,
                 device,
                 hidden_size_factor,
                 num_layers_factor,
                 latent_size_factor,):
        
        super(CVAE, self).__init__()
        
        self.device = device
        self.encoder_hidden_size = encoder_hidden_size * hidden_size_factor
        self.decoder_hidden_size = decoder_hidden_size * hidden_size_factor
        self.encoder_num_layers = encoder_num_layers * num_layers_factor
        self.decoder_num_layers = decoder_num_layers * num_layers_factor
        self.latent_size = latent_size * latent_size_factor
        
        # Encoder
        self.encoder = Encoder()
        self.latent = Latent()
        self.decoder = Decoder()
    
    def forward(self, input_chord, melody_condition, length):
        
        # Encode
        mu, log_var = self.encode(input_chord,melody_condition,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        output, softmax = self.decode(z, melody_condition)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax,logp, mu, log_var, input_chord

class SurpriseNet(nn.Module):
    def __init__(self,
                 encoder_hidden_size, 
                 decoder_hidden_size,
                 latent_size, 
                 encoder_num_layers,
                 decoder_num_layers, 
                 batch_size, 
                 device,
                 hidden_size_factor,
                 num_layers_factor,
                 latent_size_factor):
        
        super(CVAE, self).__init__()
        
        self.device = device
        self.encoder_hidden_size = encoder_hidden_size * hidden_size_factor
        self.decoder_hidden_size = decoder_hidden_size * hidden_size_factor
        self.encoder_num_layers = encoder_num_layers * num_layers_factor
        self.decoder_num_layers = decoder_num_layers * num_layers_factor
        self.latent_size = latent_size * latent_size_factor
        
        # Encoder
        self.encoder = Encoder()
        self.latent = Latent()
        self.decoder = Decoder()
    
    def forward(self, input_chord, melody_condition, length):
        
        # Encode
        mu, log_var = self.encode(input_chord,melody_condition,length)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        output, softmax = self.decode(z, melody_condition)
        
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
    
        return softmax,logp, mu, log_var, input_chord



