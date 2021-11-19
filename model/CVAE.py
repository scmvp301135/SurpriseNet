import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from constants import Constants_CVAE, Constants_CVAE_all_chords 


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_size=Constants_framewise.NUM_CHORDS + Constants.BEAT_RESOLUTION * 2 * 12, 
                hidden_size = self.encoder_hidden_size , 
                num_layers=self.encoder_num_layers,
                batch_first=True, 
                bidirectional=True)

        self.encoder_output2mean = nn.Linear(self.encoder_hidden_size * 2, self.latent_size)
        self.encoder_output2logv = nn.Linear(self.encoder_hidden_size * 2, self.latent_size)
        
    def forward(self, input_chord, melody_condition, length):
 
        input_seq = torch.cat([input_chord,melody_condition], dim=-1)
        # Pack data to encoder
        packed_x = pack_padded_sequence(input_seq, length, batch_first=True, enforce_sorted=False)
        encoder_output , (hidden, _) = self.encoder(packed_x)
        
        # Pad back
        encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True, total_length=Constants.MAX_SEQUENCE_LENGTH)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.encoder_output2mean(encoder_output)
        log_var = self.encoder_output2logv(encoder_output)

        return mu, log_var


class Latent(nn.Module):
    def __init__(self, L, N):
        super(Latent, self).__init__()

        # Latent to decoder
        self.latent2decoder_input = nn.Linear(self.latent_size + Constants.BEAT_RESOLUTION * 2 * 12, decoder_hidden_size // 2)
        
    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # If cuda
        if torch.cuda.is_available():
            eps = eps.to(self.device)
            
        z = eps * std + mu
        
        return z


class Decoder(nn.Module):
    def __init__(self, L, N):
        super(Latent, self).__init__()
        
        self.decoder = nn.LSTM(input_size=self.decoder_hidden_size // 2, 
                               hidden_size =self.decoder_hidden_size, 
                               num_layers=self.decoder_num_layers, 
                               batch_first=True, 
                               dropout=0.2, 
                               bidirectional=True)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(self.decoder_hidden_size * 2,Constants.NUM_CHORDS)


    def forward(self, mu, logvar):
        z = torch.cat([z,melody_condition],dim=-1)
        
        # Latent to hidden 
        decoder_input = self.latent2decoder_input(z)
        decoder_output, _ = self.decoder(decoder_input)
        
        # Reconstruct to one-hot chord
        result = self.outputs2chord(decoder_output)
        
        # Softmax
        softmax = F.softmax(result,dim=-1)
        
        return result, softmax

class CVAE(nn.Module):
    def __init__(self,
                 encoder_hidden_size = encoder_hidden_size, 
                 decoder_hidden_size = decoder_hidden_size,
                 latent_size = latent_size, 
                 encoder_num_layers = encoder_num_layers,
                 decoder_num_layers = decoder_num_layers, 
                 batch_size = 512, 
                 device = 'cpu',
                 hidden_size_factor = 1,
                 num_layers_factor = 1,
                 latent_size_factor = 1):
        
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


class MusicCVAE(nn.Module):
    def __init__(self,
                 encoder_hidden_size = encoder_hidden_size, 
                 decoder_hidden_size = decoder_hidden_size,
                 latent_size = latent_size, 
                 encoder_num_layers = encoder_num_layers,
                 decoder_num_layers = decoder_num_layers, 
                 batch_size = 512, 
                 device = 'cpu',
                 hidden_size_factor = 1,
                 num_layers_factor = 1,
                 latent_size_factor = 1):
        
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
                 encoder_hidden_size = encoder_hidden_size, 
                 decoder_hidden_size = decoder_hidden_size,
                 latent_size = latent_size, 
                 encoder_num_layers = encoder_num_layers,
                 decoder_num_layers = decoder_num_layers, 
                 batch_size = 512, 
                 device = 'cpu',
                 hidden_size_factor = 1,
                 num_layers_factor = 1,
                 latent_size_factor = 1):
        
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



