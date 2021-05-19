import math
import numpy as np
import pickle

# Load chord symbol data
f = open('./data/symbol_and_fingerring', 'rb')
symbol_and_fingerring = pickle.load(f)
f.close()

f = open('./data/indices_to_symbol', 'rb')
INDICES_TO_SYMBOL = pickle.load(f)

INDICES_TO_PIANOROLL = list(symbol_and_fingerring.values())
CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_number_to_note(midi_number):
    return CHROMATIC[midi_number % 12]

def pianoroll_to_note(pianoroll):
    notes = []
    for i in range(128):
        if pianoroll[i] != 0:
            notes.append(midi_number_to_note(i))

    return notes

# Pianoroll (midi) to index numbers
def pianoroll2number(pianoroll):
    midi_number = []
    for i in range(128):
        if pianoroll[i] != 0:
            midi_number.append(i)
    
    for i in range(len(midi_number)):
        midi_number[i] = midi_number[i]%12
    
    return midi_number

# Index to 12 notes onehot 
def number2onehot(number):
    onehot = [0 for i in range(12)]
    for i in number:
        onehot[i] = 1

    return onehot


# Note string to 12 onehot index
def note2number(notes):
    note2num = {'C':0,
                'C#':1, 'Db':1,
                'D':2,
                'D#': 3, 'Eb': 3,
                'E':4,
                'F':5,
                'F#': 6, 'Gb': 6,
                'G':7,
                'G#': 8, 'Ab': 8,
                'A':9,
                'A#': 10, 'Bb': 10,
                'B':11}
    num = []
    for note in notes:
        num.append(note2num[note])
    return num

# Calculate tonal coordinate in tonal space
def tonal_centroid(notes):
    fifths_lookup = {9:[1.0, 0.0], 2:[math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)], 7:[math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0:[0.0, 1.0], 5:[math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)], 10:[math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3:[-1.0, 0.0], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 1:[math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6:[0.0, -1.0], 11:[math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    minor_thirds_lookup = {3:[1.0, 0.0], 7:[1.0, 0.0], 11:[1.0, 0.0],
                           0:[0.0, 1.0], 4:[0.0, 1.0], 8:[0.0, 1.0],
                           1:[-1.0, 0.0], 5:[-1.0, 0.0], 9:[-1.0, 0.0],
                           2:[0.0, -1.0], 6:[0.0, -1.0], 10:[0.0, -1.0]}
    major_thirds_lookup = {0:[0.0, 1.0], 3:[0.0, 1.0], 6:[0.0, 1.0], 9:[0.0, 1.0],
                           2:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 5:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 11:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           1:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 7:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 10:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    r1 =1
    r2 =1
    r3 = 0.5
    if notes:
        for note in notes:
            for i in range(2):
                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]
        for i in range(2):
            fifths[i] /= len(notes)
            minor[i] /= len(notes)
            major[i] /= len(notes)

    return fifths + minor + major

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#
def neg_dist(x, y):
    dis = 0
    for i, j in zip(x, y):
        dis += math.pow(i-j, 2)
    return -math.sqrt(dis)


## Profile function
class contour_type():
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
        return y

    def type2(self):   
        x = torch.arange(0,self.length,1)
        y = - self.norm / (1 + torch.exp(-( x - self.length / 2 ))) + self.norm
        return y

    def type3(self):   
        x = torch.Tensor([0] * self.length)
        return x

    def type4(self):   
        x = torch.Tensor([self.norm] * self.length)
        return x

    def type5(self):   
        mu, sigma = self.length / 2, self.length / 8 # mean and standard deviation
        x = torch.arange(0,self.length + 1,1)
        y = 1 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        max_value = max(y)
        ratio = self.norm / max_value
        y *= ratio
        return y

    def type6(self):   
        mu, sigma = self.length / 2, self.length / 8 # mean and standard deviation
        x = torch.arange(0,self.length + 1,1)
        y = 1 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        max_value = max(y)
        ratio = self.norm / max_value
        y = -y * ratio + self.norm
        return y

