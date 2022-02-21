import torch
from torch import nn
from torch.nn.functional import softplus
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from constants import Constants
import numpy as np

class MusicCVAE(nn.Module):
    def __init__(self,
                 teacher_forcing, 
                 eps_i,
                 encoder_hidden_size = Constants.ENCODER_HIDDEN_SIZE, 
                 conductor_hidden_size = Constants.CONDUCTOR_HIDDEN_SIZE,
                 decoder_hidden_size = Constants.DECODER_HIDDEN_SIZE,
                 latent_size = Constants.LATENT_SIZE, 
                 encoder_num_layer = Constants.ENCODER_NUM_LAYER,
                 conductor_num_layer = Constants.CONDUCTOR_NUM_LAYER,
                 decoder_num_layer = Constants.DECODER_NUM_LAYER, 
                 prenet_input = 2 * 12 * Constants.BEAT_RESOLUTION,
                 prenet_hidden_size = Constants.PRENET_HIDDEN_SIZE,
                 prenet_num_layer = Constants.PRENET_NUM_LAYER,
                 batch_size = 512, 
                 device = 'cpu',
                 hidden_size_factor = 1,
                 num_layers_factor = 1,
                 latent_size_factor = 1,
                 ):
        
        super(MusicCVAE, self).__init__()
        
        self.batch_size = batch_size
        self.device = device
        self.teacher_forcing = teacher_forcing
        self.eps_i = eps_i
        self.encoder_hidden_size = encoder_hidden_size * hidden_size_factor
        self.conductor_hidden_size = conductor_hidden_size * hidden_size_factor
        self.decoder_hidden_size = decoder_hidden_size * hidden_size_factor
        self.encoder_num_layer = encoder_num_layer * num_layers_factor
        self.conductor_num_layer = conductor_num_layer * num_layers_factor
        self.decoder_num_layer = decoder_num_layer * num_layers_factor
        self.latent_size = latent_size * latent_size_factor

        # data goes into bidirectional encoder
        self.encoder = nn.GRU(input_size=Constants.NUM_CHORDS, 
                              hidden_size=self.encoder_hidden_size, 
                              num_layers=self.encoder_num_layer, 
                              batch_first=True, 
                              bidirectional=True)

        # Encoder to latent
        self.hidden2mean = nn.Linear(self.encoder_hidden_size * self.encoder_num_layer * 2, self.latent_size)
        self.hidden2logv = nn.Linear(self.encoder_hidden_size * self.encoder_num_layer * 2, self.latent_size)
        
        # Melody prenet
        self.melody_prenet = nn.GRU(input_size=2 * 12 * Constants.BEAT_RESOLUTION, 
                              hidden_size=prenet_hidden_size, 
                              num_layers=prenet_num_layer, 
                              batch_first=True, 
                              bidirectional=True)
        
        # Latent to decoder
        self.latent2conductor_input = nn.Linear(self.latent_size, Constants.NUM_CHORDS)  
        self.latent2conductor_hidden = nn.Linear(self.latent_size, self.conductor_hidden_size)  
        
        self.dropout = nn.Dropout(p=0.2)

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

    # Coin toss to determine whether to use teacher forcing on a note(Scheduled sampling)
    # Will always be True for eps_i = 1.
    def use_teacher_forcing(self):
        with torch.no_grad():
            tf = np.random.rand(1)[0] <= self.eps_i
        return tf
    
    def encode(self,chord,length):
        # Pack data to encoder
        # encoder_output,(hidden,c) = self.encoder(input)
        packed_chord = pack_padded_sequence(chord, length, batch_first=True, enforce_sorted=False)
        encoder_output, encoder_hidden = self.encoder(packed_chord)
        
        # flatten hidden state
        encoder_hidden = encoder_hidden.transpose_(0, 1).contiguous()
        encoder_hidden = encoder_hidden.view(self.batch_size, -1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.hidden2mean(encoder_hidden)
        log_var = self.hidden2logv(encoder_hidden)

        return mu, log_var
    
    def melody_embedding(self,melody,length):
        # Pack data to encoder
        # encoder_output,(hidden,c) = self.encoder(input)
        packed_melody = pack_padded_sequence(melody, length, batch_first=True, enforce_sorted=False)
        _ ,hidden = self.melody_prenet(packed_melody)
        
        # flatten hidden state
        hidden = hidden.transpose_(0, 1).contiguous()
        hidden = hidden.view(self.batch_size, -1)

        return hidden
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
#         eps = torch.randn(std.shape)
        
        # If cuda
        if torch.cuda.is_available():
            eps = eps.to(self.device)
            
        z = eps * std + mu
        
        return z
      
    def decode(self, z, melody_token, input_chord_seqs, input_melody_seqs):

        conductor_input = self.latent2conductor_input(z).unsqueeze(1)
        melody_token = melody_token.unsqueeze(1)
        conductor_input = torch.cat([conductor_input,melody_token],dim=-1)
        counter = 0
    
        if self.conductor_num_layer > 1:
            conductor_hidden = self.latent2conductor_hidden(z).unsqueeze(0).repeat(self.conductor_num_layer,1,1)
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
                decoder_input = torch.cat([embedding, input_chord_seqs[:,idx,:], input_melody_seqs[:,idx,:]],dim=-1)
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

    def forward(self, input_chord_seqs, input_melody_seqs, length):

        # Batch size
        self.batch_size, _, _ = input_chord_seqs.shape
        # Encode
        mu, log_var = self.encode(input_chord_seqs,length)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
         # Melody condition through prenet
        melody_embedding = self.melody_embedding(input_melody_seqs,length)
        # Decode
        output,softmax = self.decode(z, melody_embedding, input_chord_seqs, input_melody_seqs)
        # Log Softmax
        logp = F.log_softmax(output, dim=-1)
        
        return softmax, logp, mu, log_var, input_chord_seqs