import argparse
import numpy as np
from tqdm import tqdm
from constants import Constants

## Markov chain
class markov_chain(): 
    def __init__(self,chord_seqs,all_chords=False):
        
        self.chord_seqs = chord_seqs
#         self.length = length
        
        # All chords or not
        if all_chords:
            self.states = [x for x in range(Constants.ALL_NUM_CHORDS)]
            self.num_state = Constants.ALL_NUM_CHORDS #number of states
            
        else:
            self.states = [x for x in range(Constants.NUM_CHORDS)]
            self.num_state = Constants.NUM_CHORDS #number of states
            
        self.M = [[0]*self.num_state for _ in range(self.num_state)]
#         self.blank_M = [[0]*self.num_state for _ in range(self.num_state)]
    
    # Input one seq
    def transition_probability(self,seq):
#         M = [[0]*self.num_state for _ in range(self.num_state)]
        
        # Convert seq to index seq
#         index_seq = [self.states.index(i) for i in seq]
        index_seq = np.squeeze(seq,axis=-1).tolist()

        for (i,j) in zip(index_seq,index_seq[1:]):
            self.M[i][j] += 1

        #now convert to probabilities:
        for row in self.M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
    
    # Input one seq
    def create_transition_matrix_by_one_seq(self,seq):
        self.transition_probability(seq)
        return np.array(self.M)
    
    # Input seqs
    def create_transition_matrix_by_many_seqs(self):
        for seq in self.chord_seqs:
            self.transition_probability(seq)
        return np.array(self.M)
    
    # Input one seq
    def calculate_surprisingness(self,seq,t,TM):
        
        current = seq[t]
        i_ = self.states.index(current)

        previous = seq[t - 1]
        j_ = self.states.index(previous)

        if TM[i_][j_] == 0:
            surprisingness = -np.log(TM[i_][j_] + 1e-4)
        else:
            surprisingness = -np.log(TM[i_][j_])
            
        return surprisingness
    
    def create_surprisingness_seqs(self,average_TM=False):
    
        surprisingness_seqs = []
        n = len(self.chord_seqs)
        
        if average_TM:
            TM = self.create_transition_matrix_by_many_seqs().transpose()
            
        for i in tqdm(range(n)):
            seq = self.chord_seqs[i]
#             T = range(1,self.length[i])
            T = range(1,Constants.MAX_SEQUENCE_LENGTH)
            surprisingness_seq = [0]

            if average_TM:
                for t in T:
                    surprisingness = self.calculate_surprisingness(seq,t,TM)
                    surprisingness_seq.append(surprisingness)
                
            else:
                for t in T:
                    TM = self.create_transition_matrix_by_one_seq(seq[:t]).transpose()
                    surprisingness = self.calculate_surprisingness(seq,t,TM)
                    surprisingness_seq.append(surprisingness)
                    self.M = [[0]*self.num_state for _ in range(self.num_state)]
                   
            surprisingness_seqs.append(np.asarray(surprisingness_seq))
        
        # Pad 0 to the positions if the length of chord sequence is smaller than max length               
#         for i in tqdm(range(len(surprisingness_seqs))):
#             surprisingness_seqs[i] = np.pad(surprisingness_seqs[i], (0, Constants.MAX_SEQUENCE_LENGTH - surprisingness_seqs[i].shape[0]),'constant', constant_values = 0)
       
        # Convert all lists to np arrays
        surprisingness_seqs = np.asarray(surprisingness_seqs)
        surprisingness_seqs = np.expand_dims(surprisingness_seqs,axis=-1)

        return surprisingness_seqs, TM
    
# #     

## Main
def main():
    ''' 
    Usage:
    python create_surprisingness.py  ///
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 
    
    parser.add_argument('-average_TM', default=False) 
    parser.add_argument('-all_chords', default=False) 
    parser.add_argument('-filename', type=str, required=True) 
    args = parser.parse_args()
    
    length = np.load('./data/length.npy')
    
    if args.all_chords:
        chord_seqs = np.load('./data/chord_indices.npy')
        
    else:
        chord_seqs = np.load('./data/number_96_2_beat.npy')
    
    surprisingness_seqs, TM = markov_chain(chord_seqs,
                                           args.all_chords).create_surprisingness_seqs(args.average_TM)
    np.save('./data/' + args.filename , surprisingness_seqs)
    
if __name__ == '__main__':
    main()

    
