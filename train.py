import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.dataloader import HLSDDataset
from model.CVAE import CVAE, MusicCVAE, SurpriseNet
from sklearn.metrics import accuracy_score

class TrainingCVAE():
    def __init__(self, args, step=0, k=0.0025, x0=2500):
        
        self.batch_size = args.batch_size
        self.val_size = args.val_size
        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.cuda = args.cuda
        self.device = torch.device('cuda:' + self.cuda) if torch.cuda.is_available() else 'cpu'
        self.step = step
        self.k = k
        self.x0 = x0
        self.training_loss = 0
        self.validation_loss = 0
        self.save_model = args.save_model
        self.weight = args.weight
        self.latent_size_factor = args.latent_size_factor
        self.hidden_size_factor = args.hidden_size_factor
        self.num_layers_factor = args.num_layers_factor
        self.input_surprise = args.input_surprise
        
    # Loss function
    def loss_fn(self,loss_function, logp, target, length, mean, log_var, anneal_function, step, k, x0):

        # Negative Log Likelihood
        NLL_loss = loss_function(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        KL_weight = self.kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    # Annealing function 
    def kl_anneal_function(self,anneal_function, step, k, x0):
            if anneal_function == 'logistic':
                return float(1/(1+np.exp(-k*(step-x0))))
            elif anneal_function == 'linear':
                return min(1, step/x0) 

    def load_data(self):

        # Create dataloader
        print('Creating dataloader...')
        dataset = HLSDDataset()
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        train_indices = indices[:self.val_size]
        valid_indices = indices[self.val_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, drop_last=True, sampler=train_sampler)
        valid_dataloader = DataLoader(dataset, sampler=valid_sampler)

        return train_dataloader, valid_dataloader
    
    # Reconstruction rate (accuracy):
    def cal_reconstruction_rate(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        acc = accuracy_score(y_true, y_pred)
        print('Accuracy:' + f'{acc:.4f}')
        
    
    def train(self, model, optimizer, dataloader, step, k, x0, loss_function):  
        ########## Training mode ###########
            model.train()
            training_loss = self.training_loss

            for chord_onehots, length, melody, surprise, chord_indices in dataloader:

                # melody (512, 272, 12 * 24 * 2)
                # chord (512, 272, 1) 
                # length (512, 1)
                # chord_onehot (512, 272, 96)
        
                melody, length, chord_onehots = melody.to(self.device), length.to(self.device).squeeze(), chord_onehots.to(self.device)
                optimizer.zero_grad()

                # Model prediction
                if self.input_surprise:
                    pred, logp ,mu, log_var, _ = model(chord_onehots, length, melody, surprise)

                else:
                    pred, logp ,mu, log_var, _ = model(chord_onehots, length, melody)

                # Arrange 
                pred_flatten = []
                groundtruth_flatten = []
                logp_flatten = []
                length = length.squeeze()

                for i in range(self.batch_size):

                    # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                    logp_flatten.append(logp[i][:length[i]])

                    # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,12 * 24 * 2)
                    pred_flatten.append(pred[i][:length[i]])

                    # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
                    groundtruth_flatten.append(chord_onehots[i][:length[i]])

                # Rearrange for loss calculation
                logp_flatten = torch.cat(logp_flatten, dim=0)
                pred_flatten = torch.cat(pred_flatten, dim=0)

                # Loss calculation
                NLL_loss, KL_loss, KL_weight = self.loss_fn(loss_function = loss_function, logp = logp_flatten, target = chord_indices, length = length, mean = mu, log_var = log_var, anneal_function='logistic', step=step, k=k, x0=x0)
                self.step += 1
                loss = (NLL_loss + KL_weight * KL_loss)
                training_loss += loss.item()

                loss.backward()
                optimizer.step()

            print('training_loss: ', training_loss / (17505 // self.batch_size))

    def eval(self, model, dataloader, step, k, x0, loss_function):
        ########## Evaluation mode ###########
            model.eval()
            validation_loss = self.validation_loss

            for chord_onehots, length, melody, surprise, chord_indices in dataloader:
                
                melody, length, chord_onehot = val_melody.to(self.device), val_length.to(self.device).squeeze(), val_chord_onehot.to(self.device)
       
                # Model prediction
                pred, logp ,mu, log_var, _ = model(chord_onehot,melody,length)

                # Arrange 
                pred_flatten = []
                groundtruth_flatten = []
                logp_flatten = []
                length = length.squeeze()

                for i in range(self.val_size):
                    
                    # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                    logp_flatten.append(logp[i][:length[i]])

                    # Get predicted softmax chords by length of the song (cutting off padding 0), (1,length,96)
                    pred_flatten.append(pred[i][:length[i]])

                    # Get groundtruth chords by length of the song (cutting off padding 0), (1,length)
                    groundtruth_flatten.append(chord_onehot[i][:length[i]])

                # Rearrange for loss calculatio
                logp_flatten = torch.cat(logp_flatten, dim=0)
                pred_flatten = torch.cat(pred_flatten, dim=0)
                pred_index = torch.max(pred_flatten,1).indices
                groundtruth_flatten = torch.cat(groundtruth_flatten,dim=0).long()
                groundtruth_index = torch.max(groundtruth_flatten,1).indices

                # Loss calculation
                # Add weight to NLL also
                NLL_loss, KL_loss, KL_weight = self.loss_fn(loss_function = loss_function, logp = logp_flatten, target = groundtruth_index, length = length, mean = mu, log_var = log_var,anneal_function='logistic', step=step, k=k, x0=x0)
                loss = (NLL_loss + KL_weight * KL_loss) 
                validation_loss += loss.item()

                print('validation_loss: ', validation_loss)
                self.cal_reconstruction_rate(groundtruth_index.cpu(),pred_index.cpu())

    # Model training  
    def run(self):
        
        epochs = self.epoch
        # Load data
        train_dataloader, valid_dataloader = self.load_data()

        # Model
        print('building model...')
        model = CVAE(device = self.device,
                     hidden_size_factor = self.hidden_size_factor,
                     num_layers_factor = self.num_layers_factor,
                     latent_size_factor = self.latent_size_factor).to(self.device)
        print(model)

        # Training parameters
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        lambda1 = lambda epoch: 0.995 ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        if self.weight:
            self.weight = weight_chord
        loss_function = torch.nn.NLLLoss(weight = self.weight)

        # Define annealing parameters
        step = self.step
        k = self.k
        x0 = self.x0
        
        print('start training...')
        for epoch in tqdm(range(epochs)):
            print('epoch: ', epoch + 1)
            
            self.train(model,
                       optimizer,
                       dataloader,
                       step,k,x0,
                       loss_function
                      )
            
            self.eval(model,
                      val_melody,
                      val_chord_onehot,
                      val_length,
                      step,k,x0,
                      loss_function
                     )

        # Save recontructed results
        # np.save('reconstructed_one_hot_chords.npy', chord_pred.cpu().detach().numpy()) 

        # Save model
        model_dir = 'output_models/' + self.save_model
        torch.save(model.state_dict(), model_dir + '.pth')

## Main
def main():
    ''' 
    Usage:
    python train.py -save_model "/output_models/"
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 

    parser.add_argument('-learning_rate', type=float, default=0.001)   
    parser.add_argument('-val_size', default=500)    
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=512)
    parser.add_argument('-save_model', type=str, default="output_models/")
    parser.add_argument('-cuda', type=str, default='0')
    parser.add_argument('-weight', type=bool, default=None)
    parser.add_argument('-latent_size_factor', type=int, default=1)
    parser.add_argument('-hidden_size_factor', type=int, default=1)
    parser.add_argument('-num_layers_factor', type=int, default=1)
    parser.add_argument('-input_surprise', type=bool, default=False)
    
    args = parser.parse_args()
    
    train = TrainingCVAE(args)
    train.run()
    
if __name__ == '__main__':
    main()
