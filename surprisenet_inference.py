import argparse
from tqdm import tqdm
import numpy as np
import torch
import pickle
from decode import *
from model.surprise_CVAE_all_chords import CVAE
from metrics_all_chords import CHE_and_CC, CTD, CTnCTR, PCS, MCTD
from constants import Constants, Constants_framewise
from sklearn.metrics import accuracy_score
from utils import contour_type

class InferenceVAE():
    def __init__(self,args):
        
        self.seed = args.seed
        self.cuda = args.cuda
        self.device = torch.device('cuda:' + self.cuda) if torch.cuda.is_available() else 'cpu'
        self.max_seq_len = Constants.MAX_SEQUENCE_LENGTH
        self.model_path = args.model_path
        self.inference_size = args.inference_size
        self.outputdir = args.outputdir
        self.decode_to_pianoroll = args.decode_to_pianoroll
        self.random_sample = args.random_sample
        self.latent_size_factor = args.latent_size_factor
        self.hidden_size_factor = args.hidden_size_factor
        self.num_layers_factor = args.num_layers_factor
        self.prenet_size_factor = args.prenet_size_factor
        
        
    def load_data(self):
        # Load data
        print('loading data...')

        melody_data = np.load('./data/melody_data.npy')
        chord_groundtruth = np.load('./data/chord_groundtruth.npy')
        melody_condition = np.load('./data/melody_baseline.npy')
        lengths = np.load('./data/length.npy')

        f = open('./data/tempos', 'rb')
        tempos = pickle.load(f)
        f.close()
        f = open('./data/downbeats', 'rb')
        downbeats = pickle.load(f)
        f.close()

        print('splitting testing set...')
        melody_data = melody_data[:self.inference_size]
        chord_groundtruth = chord_groundtruth[:self.inference_size]
        val_melody_condition = torch.from_numpy(melody_condition[:self.inference_size]).float()
        val_length = torch.from_numpy(lengths[:self.inference_size])

        tempos = tempos[:self.inference_size]
        downbeats = downbeats[:self.inference_size]
        
        return melody_data, val_melody_condition, chord_groundtruth, val_length, tempos, downbeats
        
    ## Calculate objective metrics
    def cal_objective_metrics(self,melody,chord_pred,length):

        f = open('metrics/' + self.outputdir + '.txt', 'w')
        m = [0 for i in range(6)]
        for i in range(self.inference_size):
            chord_pred_part = chord_pred[i][:length[i]]
            melody_part = melody[i][:length[i]]
        #     print(chord_pred_part.shape)
        #     print(melody_part.shape)

            che, cc = CHE_and_CC(chord_pred_part, chord_num=96)
            ctd = CTD(chord_pred_part, chord_num=96)
            ctnctr = CTnCTR(melody_part, chord_pred_part, chord_num=96)
            pcs = PCS(melody_part, chord_pred_part, chord_num=96)
            mctd = MCTD(melody_part, chord_pred_part, chord_num=96)
            m[0] += che
            m[1] += cc
            m[2] += ctd
            m[3] += ctnctr
            m[4] += pcs
            m[5] += mctd
            f.write(str(che) + " " + str(cc) + " " + str(ctd) + " " + str(ctnctr) + " " + str(pcs) + " " + str(mctd) + '\n')
        f.close()

        print('CHE: ', m[0]/self.inference_size)
        print('CC: ', m[1]/self.inference_size)
        print('CTD: ', m[2]/self.inference_size)
        print('CTnCTR: ', m[3]/self.inference_size)
        print('PCS: ', m[4]/self.inference_size)
        print('MCTD: ', m[5]/self.inference_size)
    
    ## Reconstruction rate (accuracy):
    def cal_reconstruction_rate(self,y_true,y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        acc = accuracy_score(y_true,y_pred)
        print('Accuracy:' + f'{acc:.2f}')
    
    def load_model(self,model_path):
        # Load model
        print('building model...')
        model = CVAE(device = self.device,
                     hidden_size_factor = self.hidden_size_factor,
                     num_layers_factor = self.num_layers_factor,
                     latent_size_factor = self.latent_size_factor,
                     prenet_size_factor = 1).to(self.device)
        model.load_state_dict(torch.load('output_models/' + model_path + '.pth'))
        
        return model
        
    def sample_one_by_one(self, model, melody_data, chord_groundtruth, tempos, downbeats):
        
        np.random.seed(self.seed)
        indices = np.random.randint(500, size=self.inference_size)
#         print(indices)

        for index in indices:
            melody_truth = np.expand_dims(melody_data[index], axis=0)
            chord_truth = np.expand_dims(chord_groundtruth[index], axis=0)
            tempo = [tempos[index]]
            downbeat = [downbeats[index]]

            melody = val_melody_condition[index].unsqueeze(dim=0)
            melody = melody.view(1,-1)
            inference_length = torch.Tensor([val_length[index]]).long()

            # Sampling
            torch.manual_seed(self.seed)
            latent = torch.randn(1,Constants_framewise.LATENT_SIZE).to(self.device)

    #         z = torch.cat((latent,melody1,r_pitch,r_rhythm), dim=-1)
            z = torch.cat((latent,melody), dim=-1)

            _, sample = model.decode(z)

            ########## Random sampling ###########
            # Proceed chord decode
            print('proceed chord decode...')
            decode_length = inference_length
            joint_prob = sample.cpu().detach().numpy()

            # Append argmax index to get pianoroll array
            accompany_pianoroll = argmax2pianoroll(joint_prob)

            # augment chord into frame base
            accompany_pianoroll_frame, chord_groundtruth_frame = sequence2frame(accompany_pianoroll, chord_truth, BEAT_RESOLUTION=Constants.BEAT_RESOLUTION, BEAT_PER_CHORD=Constants.BEAT_PER_CHORD)
            
            # length into frame base
            decode_length = decode_length * Constants.BEAT_RESOLUTION * Constants.BEAT_PER_CHORD

            # write pianoroll
            result_dir = 'results/' + self.outputdir
    #         filename = str(index) + '-pitch-' + str(args.pitch_ratio) + '-rhythm-' + str(args.rhythm_ratio)
            filename = str(index)
            print(result_dir)
            print(result_dir + '/' + filename + '.mid')
            write_one_pianoroll(result_dir, filename ,melody_truth, accompany_pianoroll_frame,chord_groundtruth_frame, decode_length, tempo,downbeat)
        
    def sample_from_latent(self, model, melody_condition):
        
        # Sampling
        z = torch.randn(self.inference_size,Constants.MAX_SEQUENCE_LENGTH,Constants_framewise.LATENT_SIZE).to(self.device)
#         z = torch.cat((latent,input_melody), dim=-1)
        _,samples = model.decode(z,melody_condition)
        
        return samples
    
    def decode2pianoroll(self, melody_data, val_length, accompany_pianoroll, chord_groundtruth, tempos, downbeats):
        
        # augment chord into frame base
        accompany_pianoroll_frame, chord_groundtruth_frame = sequence2frame(accompany_pianoroll, chord_groundtruth)
 
        # length into frame base
        length = val_length * Constants.BEAT_RESOLUTION * Constants.BEAT_PER_CHORD

        # write pianoroll
        result_dir = 'results/' + self.outputdir
        write_pianoroll(result_dir, melody_data, accompany_pianoroll_frame,chord_groundtruth_frame, length, tempos,downbeats)

    ## Model inference
    def run(self):
        
        melody_data, val_melody_condition, chord_groundtruth, val_length, tempos, downbeats = self.load_data()
        val_melody_condition, val_length = val_melody_condition.to(self.device), val_length.to(self.device).squeeze()
        val_length = val_length.cpu().detach().numpy()
        
        model = self.load_model(self.model_path)
        model.eval()  
        
        if self.random_sample:
            self.sample_one_by_one(model, melody_data, chord_groundtruth, tempos, downbeats)
        
        else:
            ########## Sampling ###########
            with torch.no_grad():
                samples = self.sample_from_latent(model, val_melody_condition)

            # Proceed chord decode
            print('proceed chord decode...')
            samples = samples.cpu().detach().numpy()
            joint_prob = samples

            # Append argmax index to get pianoroll array
            accompany_pianoroll = argmax2pianoroll(joint_prob)

            # Calculate accuracy
            self.cal_reconstruction_rate(chord_groundtruth,accompany_pianoroll)

            # cal metrics
            val_melody_condition = val_melody_condition.cpu().detach().numpy()
            self.cal_objective_metrics(val_melody_condition, samples, val_length)
            
            # Decode to pianoroll or not
            if self.decode_to_pianoroll:
                self.decode2pianoroll(melody_data, val_length, accompany_pianoroll, chord_groundtruth, tempos, downbeats)

## Main
def main():
    ''' 
    Usage:
    python cvae_inference.py -model_path ///
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 
    
    parser.add_argument('-inference_size', type=int, default=500) 
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-outputdir', type=str, default='new_results')
    parser.add_argument('-cuda', type=str, default='0')
    parser.add_argument('-seed', default=30, type=str, help='random seed')
    parser.add_argument('-decode_to_pianoroll', default=False)
    parser.add_argument('-random_sample', default=False)
    parser.add_argument('-latent_size_factor', type=int, default=1)
    parser.add_argument('-hidden_size_factor', type=int, default=1)
    parser.add_argument('-num_layers_factor', type=int, default=1)
    parser.add_argument('-prenet_size_factor', type=int, default=1)
    
    args = parser.parse_args()
    
    inference = InferenceVAE(args)
    inference.run()
    
if __name__ == '__main__':
    main()

