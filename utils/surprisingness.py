"""
Author
    * Yi Wei Chen 2021
"""

import numpy as np
from tqdm import tqdm

# Markov chain
class markov_chain(): 
    """
    Create surprisingness sequence from integer numpy array with shape of (batch, time, 1(index number)).
    Example: 

    Input: 
        num_chords = 5
        seq = np.random.randint(num_chords, size=(10,40,1))

    Output:
        surprisingness_seqs, TM = surprisingness.markov_chain(seq, num_chords).create_surprisingness_seqs()

    """
    def __init__(self, chord_seqs, chord_nums, all_chords=False):
        
        # Shape of chord_seqs(numpy array) : (batch, time, 1(index number))
        self.chord_seqs = chord_seqs
        self.states = [x for x in range(chord_nums)]
        self.num_state = chord_nums #number of states 
        self.M = [[0] * self.num_state for _ in range(self.num_state)]
  
    # Calculate transition_probability
    def transition_probability(self, seq):
        # Convert seq to index seq
        index_seq = np.squeeze(seq, axis=-1).tolist()

        for (i,j) in zip(index_seq, index_seq[1:]):
            self.M[i][j] += 1

        # Convert to probabilities:
        for row in self.M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
    
    # Create transition matrix from one chord sequence
    def create_transition_matrix(self, seq):
        self.transition_probability(seq)
        return np.array(self.M)
    
    # Calculate surprisingness
    def calculate_surprisingness(self, seq, t, TM):
        
        current = seq[t]
        i_ = self.states.index(current)

        previous = seq[t - 1]
        j_ = self.states.index(previous)

        if TM[i_][j_] == 0:
            surprisingness = -np.log(TM[i_][j_] + 1e-4)
        else:
            surprisingness = -np.log(TM[i_][j_])
            
        return surprisingness
    
    # Create surprisingness sequences
    def create_surprisingness_seqs(self):
    
        surprisingness_seqs = []
        batch = len(self.chord_seqs)
        
        # Calculate surprisingness for chord sequences 
        for i in tqdm(range(batch)):
            seq = self.chord_seqs[i]
            timesteps = range(1, len(seq))
            surprisingness_seq = [0]

            for step in timesteps:
                TM = self.create_transition_matrix(seq[:step]).transpose()
                surprisingness = self.calculate_surprisingness(seq, step, TM)
                surprisingness_seq.append(surprisingness)

                # Re-initiate a new transition matrix for next sequence
                self.M = [[0] * self.num_state for _ in range(self.num_state)]
                   
            surprisingness_seqs.append(np.asarray(surprisingness_seq))
        
        # Pad 0 to the positions if the length of the chord sequence is smaller than max length               
        for i in tqdm(range(len(surprisingness_seqs))):
            surprisingness_seqs[i] = np.pad(surprisingness_seqs[i], (0, 272 - surprisingness_seqs[i].shape[0]),'constant', constant_values = 0)
       
        # Convert all lists to np arrays
        surprisingness_seqs = np.asarray(surprisingness_seqs)
        surprisingness_seqs = np.expand_dims(surprisingness_seqs, axis=-1)

        return surprisingness_seqs, TM  # surprisingness_seqs (batch, max_seq_length, 1), TM (num_state, num_state)

