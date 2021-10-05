import math
import numpy as np
import pickle
import torch

# Load symbol_and_fingerring data
f = open('./utils/symbol_and_fingerring.pkl', 'rb')
symbol_and_fingerring = pickle.load(f)
f.close()

# Load symbol_and_fingerring data
f = open('./utils/indices_to_symbol.pkl', 'rb')
INDICES_TO_SYMBOL = pickle.load(f)

INDICES_TO_PIANOROLL = list(symbol_and_fingerring.values())
CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midinum_to_note(midi_number):
    return CHROMATIC[midi_number % 12]

def pianoroll_to_note(pianoroll):
    notes = []
    for i in range(128):
        if pianoroll[i] != 0:
            notes.append(midinum_to_note(i))

    return notes

# Pianoroll (midi) to index numbers
def pianoroll_to_number(pianoroll):
    midi_number = []
    for i in range(128):
        if pianoroll[i] != 0:
            midi_number.append(i)
    
    for i in range(len(midi_number)):
        midi_number[i] = midi_number[i]%12
    
    return midi_number

# Index to 12 notes onehot 
def number_to_onehot(number):
    onehot = [0 for i in range(12)]
    for i in number:
        onehot[i] = 1

    return onehot

# Note string to 12 onehot index
def note_to_number(notes):
    note_to_num = {'C':0,
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
        num.append(note_to_num[note])
    return num

def symbol_to_number96(symbol):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    order = 0
    char_list = list(symbol)

    if 'o' in char_list or 'ø' in char_list:
        # dim
        order = 4
    else:
        if '#' in char_list and '5' in char_list:
            # aug
            order = 3
        else:
            if 's' in char_list and 'u' in char_list:
                # sus
                order = 5
            else:
                if '7' in char_list or '9' in char_list or '11' in char_list:
                    #7th chord
                    if 'm' in char_list:
                        #major7, minor7
                        if 'j' in char_list:
                            #maj7
                            order = 6
                        else:
                            #minor7
                            order = 7
                    else:
                        #dominant7
                        order = 8
                else:
                    #major. minor
                    if 'm' in char_list and 'j' not in char_list:
                        #minor
                        order = 2
                    else:
                        #major
                        order = 1
    index = 0
    if len(char_list) == 1:
        if char_list[0] == 'A' or char_list[0] == 'a':
            index = 9
        if char_list[0] == 'B' or char_list[0] == 'b':
            index = 11
        if char_list[0] == 'C' or char_list[0] == 'c':
            index = 0
        if char_list[0] == 'D' or char_list[0] == 'd':
            index = 2
        if char_list[0] == 'E' or char_list[0] == 'e':
            index = 4
        if char_list[0] == 'F' or char_list[0] == 'f':
            index = 5
        if char_list[0] == 'G' or char_list[0] == 'g':
            index = 7
    else:
        if char_list[1] == 'b':
            #flat
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 8
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 10
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 1
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 3
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 6
        else:
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 9
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 11
            if char_list[0] == 'C' or char_list[0] == 'c':
                index = 0
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 2
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 4
            if char_list[0] == 'F' or char_list[0] == 'f':
                index = 5
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 7

    return np.asarray([index * 8 + (order - 1)])

def symbol_to_onehot96(symbol):
    # order: major, minor. aug, dim, sus, major7, minor7, dominant7
    order = 0
    char_list = list(symbol)

    if 'o' in char_list or 'ø' in char_list:
        # dim
        order = 4
    else:
        if '#' in char_list and '5' in char_list:
            # aug
            order = 3
        else:
            if 's' in char_list and 'u' in char_list:
                # sus
                order = 5
            else:
                if '7' in char_list or '9' in char_list or '11' in char_list:
                    #7th chord
                    if 'm' in char_list:
                        #major7, minor7
                        if 'j' in char_list:
                            #maj7
                            order = 6
                        else:
                            #minor7
                            order = 7
                    else:
                        #dominant7
                        order = 8
                else:
                    #major. minor
                    if 'm' in char_list and 'j' not in char_list:
                        #minor
                        order = 2
                    else:
                        #major
                        order = 1
    index = 0
    if len(char_list) == 1:
        if char_list[0] == 'A' or char_list[0] == 'a':
            index = 9
        if char_list[0] == 'B' or char_list[0] == 'b':
            index = 11
        if char_list[0] == 'C' or char_list[0] == 'c':
            index = 0
        if char_list[0] == 'D' or char_list[0] == 'd':
            index = 2
        if char_list[0] == 'E' or char_list[0] == 'e':
            index = 4
        if char_list[0] == 'F' or char_list[0] == 'f':
            index = 5
        if char_list[0] == 'G' or char_list[0] == 'g':
            index = 7
    else:
        if char_list[1] == 'b':
            #flat
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 8
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 10
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 1
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 3
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 6
        else:
            if char_list[0] == 'A' or char_list[0] == 'a':
                index = 9
            if char_list[0] == 'B' or char_list[0] == 'b':
                index = 11
            if char_list[0] == 'C' or char_list[0] == 'c':
                index = 0
            if char_list[0] == 'D' or char_list[0] == 'd':
                index = 2
            if char_list[0] == 'E' or char_list[0] == 'e':
                index = 4
            if char_list[0] == 'F' or char_list[0] == 'f':
                index = 5
            if char_list[0] == 'G' or char_list[0] == 'g':
                index = 7

    s = [0 for i in range(96)]
    s[index * 8 + (order - 1)] = 1
    return s


# Calculate tonal coordinate in tonal space
# def tonal_centroid(notes):
#     fifths_lookup = {9:[1.0, 0.0], 2:[math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)], 7:[math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
#                      0:[0.0, 1.0], 5:[math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)], 10:[math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
#                      3:[-1.0, 0.0], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 1:[math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
#                      6:[0.0, -1.0], 11:[math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
#     minor_thirds_lookup = {3:[1.0, 0.0], 7:[1.0, 0.0], 11:[1.0, 0.0],
#                            0:[0.0, 1.0], 4:[0.0, 1.0], 8:[0.0, 1.0],
#                            1:[-1.0, 0.0], 5:[-1.0, 0.0], 9:[-1.0, 0.0],
#                            2:[0.0, -1.0], 6:[0.0, -1.0], 10:[0.0, -1.0]}
#     major_thirds_lookup = {0:[0.0, 1.0], 3:[0.0, 1.0], 6:[0.0, 1.0], 9:[0.0, 1.0],
#                            2:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 5:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 11:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
#                            1:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 7:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 10:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

#     fifths = [0.0, 0.0]
#     minor = [0.0, 0.0]
#     major = [0.0, 0.0]
#     r1 =1
#     r2 =1
#     r3 = 0.5
#     if notes:
#         for note in notes:
#             for i in range(2):
#                 fifths[i] += r1 * fifths_lookup[note][i]
#                 minor[i] += r2 * minor_thirds_lookup[note][i]
#                 major[i] += r3 * major_thirds_lookup[note][i]
#         for i in range(2):
#             fifths[i] /= len(notes)
#             minor[i] /= len(notes)
#             major[i] /= len(notes)

#     return fifths + minor + major

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)

# #
# def neg_dist(x, y):
#     dis = 0
#     for i, j in zip(x, y):
#         dis += math.pow(i-j, 2)
#     return -math.sqrt(dis)

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

