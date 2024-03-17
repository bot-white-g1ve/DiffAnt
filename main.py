import os
import copy
import ivtmetrics
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, set_random_seed
from utils import get_convert_matrix
import random
import pdb

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, causal, 
        num_targets, sample_rate, temporal_aug, set_sampling_seed, guidance_matrices, ant_range, device):

        assert(sample_rate == 1)
        assert(ant_range > 0)

        self.device = device
        self.num_targets = num_targets
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.guidance_matrices = guidance_matrices
        self.ant_range = ant_range

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, causal, self.num_targets, self.guidance_matrices, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, val_test_dataset, loss_weights, class_weighting,
              num_epochs, batch_size, learning_rate, weight_decay, result_dir, log_freq, log_train_results, log_APs,
              evaluation_protocol):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting: # To be checked
            class_weights = train_train_dataset.get_class_weights(class_weighting)
            class_weights = class_weights.float().to(device)
        else:
            class_weights = None
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):

                feature, label, video = data
                feature, label = feature.to(device), label.to(device)

                # ant_rid = random.randint(0, self.ant_range * 2) # a<=x<=b
                # ant_rid = random.randint(0, 10) # a<=x<=b
                ant_rid = random.randint(0, self.ant_range) # a<=x<=b

                if ant_rid > 0:
                    feature = feature[:,:,:-ant_rid]
                    label = label[:,:,ant_rid:]

                loss_dict = self.model.get_training_loss(feature, 
                    event_gt=label,  # 1, C, T
                    class_weights=class_weights,
                    ant_range=ant_rid,
                )

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, C, T])
                # # output    torch.Size([1, C, T]) 
                # ##################

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        
            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:

                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')

                # for mode in ['encoder-agg', 'decoder-agg', 'ensemble-agg']: 
                # for mode in ['encoder-agg', 'decoder-agg']: 
                for mode in ['decoder-agg']: 

                    if test_test_dataset:

                        test_result_dict = self.test(
                            test_test_dataset, mode, log_APs, evaluation_protocol, model_path=None)

                        if result_dir:
                            for k,v in test_result_dict.items():
                                logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                            np.save(os.path.join(result_dir, 
                                f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                        for k,v in test_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


                    if val_test_dataset:

                        val_result_dict = self.test(
                            val_test_dataset, mode, log_APs, evaluation_protocol, model_path=None)

                        if result_dir:
                            for k,v in val_result_dict.items():
                                logger.add_scalar(f'Val-{mode}-{k}', v, epoch)

                            np.save(os.path.join(result_dir, 
                                f'val_results_{mode}_epoch{epoch}.npy'), val_result_dict)

                        for k,v in val_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Val-{k} {v}')


                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, log_APs, evaluation_protocol, model_path=None)

                        if result_dir:
                            for k,v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        for k,v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if result_dir:
            logger.close()


    def test_single_video(self, video_idx, test_dataset, mode, model_path=None):

        device = self.device  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder-noagg', 'encoder-agg', 'decoder-noagg', 'decoder-agg', 'ensemble-noagg', 'ensemble-agg'])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature, label, video = test_dataset[video_idx]

            feature = [i[:,:,:-self.ant_range] for i in feature]
            label = label[:,:,self.ant_range:]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, C, Original T])
            # output: [torch.Size([1, C, Sampled T])]
            
            label = label.squeeze(0).cpu().numpy()

            # Rethink the aggregation, mean is not good for CholecT50
            if mode in ['encoder-agg', 'decoder-agg', 'ensemble-agg']:
                
                if mode == 'encoder-agg':
                    output = [self.model.encoder(feature[i].to(device)) 
                           for i in range(len(feature))] # output is a list of tuples
                    output = [F.sigmoid(i).cpu() for i in output]
                if mode == 'decoder-agg':
                    output = [self.model.ddim_sample(feature[i].to(device), self.ant_range, seed) 
                               for i in range(len(feature))] # output is a list of tuples
                    output = [i.cpu() for i in output]

                if mode == 'ensemble-agg':

                    output_encoder = [self.model.encoder(feature[i].to(device)) 
                           for i in range(len(feature))]
                    output_encoder = [F.sigmoid(i).cpu() for i in output_encoder]

                    output_decoder = [self.model.ddim_sample(feature[i].to(device), self.ant_range, seed) 
                               for i in range(len(feature))] 
                    output_decoder = [i.cpu() for i in output_decoder]

                    output = [(output_encoder[i] + output_decoder[i]) / 2 # TO DO: maybe change combination weights
                        for i in range(len(feature))]

                assert(output[0].shape[0] == 1)
                agg_output = np.zeros(label.shape) # C x T
                for offset in range(self.sample_rate):
                    agg_output[:,offset::self.sample_rate] = output[offset].squeeze(0).numpy()
                output = agg_output

            if mode in ['encoder-noagg', 'decoder-noagg', 'ensemble-noagg']: # temporal aug must be true

                if mode == 'encoder-noagg':
                    output = self.model.encoder(feature[len(feature)//2].to(device)) 
                    output = F.sigmoid(output).cpu()
                if mode == 'decoder-noagg':
                    output = self.model.ddim_sample(feature[len(feature)//2].to(device), self.ant_range, seed) 
                    output = output.cpu() 
                if mode == 'ensemble-noagg':
                    
                    output_encoder = self.model.encoder(feature[len(feature)//2].to(device)) 
                    output_encoder = F.sigmoid(output_encoder).cpu()

                    output_decoder = self.model.ddim_sample(feature[len(feature)//2].to(device), self.ant_range, seed) 
                    output_decoder = output_decoder.cpu() 

                    output = (output_encoder + output_decoder) / 2 # TO DO: maybe change combination weights

                # TO DO
                assert(output.shape[0] == 1) # 1xCxT
                output = F.interpolate(output, size=label.shape[1])
                output = output.squeeze(0).numpy()
                
            assert(output.shape == label.shape) # C x T
            assert(output.max() <= 1 and output.min() >= 0)

            return video, output.T, label.T

    def test(self, test_dataset, mode, log_APs, evaluation_protocol, model_path=None):
        device = self.device

        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        # ivtmetrics: TO DO to be checked, reproduce existing ones
        recognize = ivtmetrics.Recognition(100)
        recognize.reset_global()
        if evaluation_protocol != "Challenge":
            recognize_i = ivtmetrics.Recognition(6)
            recognize_v = ivtmetrics.Recognition(10)
            recognize_t = ivtmetrics.Recognition(15)
            recognize_i.reset_global()
            recognize_v.reset_global()
            recognize_t.reset_global()
            recognize_dict = {'ivt':recognize, 'iv':recognize, 'it':recognize, 'i':recognize_i, 'v':recognize_v, 't':recognize_t} # no rec for iv and it

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        all_acc = []

        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, model_path)

                # pred.shape == label.shape  # T x C

                pred = test_dataset.split_components(pred)
                label = test_dataset.split_components(label)

                # ivt (3415, 100)
                # i (3415, 6)
                # v (3415, 10)
                # t (3415, 15)

                ###########################################
                pred_ivt = pred['ivt']
                label_ivt = label['ivt']
                ###########################################

                # ivtmetrics
                if evaluation_protocol == 'Challenge':
                    recognize.update(label_ivt, pred_ivt)
                else:
                    for ap in log_APs:
                        if ap in ['i', 'v', 't', 'ivt']:
                            recognize_dict[ap].update(label[ap], pred[ap])
                
                if evaluation_protocol == 'Challenge':
                    recognize.video_end()
                else:
                    for ap in log_APs:
                        if ap in ['i', 'v', 't', 'ivt']:
                            recognize_dict[ap].video_end()
                
                # A naive ACC is used here now
                correct = 0
                for i in range(pred_ivt.shape[0]):
                    correct += int(np.all((pred_ivt[i] > 0.5) == label_ivt[i]))
                acc = correct / pred_ivt.shape[0]

                all_acc.append(acc)

        APs = {'i':{'mAP':0}, 'v':{'mAP':0}, 't':{'mAP':0}, 'iv':{'mAP':0}, 'it':{'mAP':0}, 'ivt':{'mAP':0}}
        TopNs = {1:0, 5:0, 10:0, 20:0}

        if evaluation_protocol == 'Challenge':
            for ap in log_APs:
                APs[ap] = recognize.compute_video_AP(ap, ignore_null=True)
            
            if 'ivt' in log_APs:
                for n in TopNs.keys():
                    TopNs[n] = recognize.topK(n, 'ivt')
        else:
            for ap in log_APs:
                if ap in ['ivt', 'it', 'iv']:
                    APs[ap] = recognize_dict[ap].compute_video_AP(ap, ignore_null=False)
                else:
                    APs[ap] = recognize_dict[ap].compute_video_AP(ignore_null=False)

            if 'ivt' in log_APs:
                for n in TopNs.keys():
                    TopNs[n] = recognize_dict['ivt'].topK(n, 'ivt')

        result_dict = {
            'Acc': np.mean(all_acc),
            'AP_I': APs['i']['mAP'],
            'AP_V': APs['v']['mAP'],
            'AP_T': APs['t']['mAP'],
            'AP_IV': APs['iv']['mAP'],
            'AP_IT': APs['it']['mAP'] ,
            'AP_IVT': APs['ivt']['mAP'],
            'Top-1': TopNs[1],
            'Top-5': TopNs[5],
            'Top-10': TopNs[10],
            'Top-20': TopNs[20],
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    mapping_file = os.path.join(root_data_dir, dataset_name, 'label_mapping.txt')

    train_data_dict = get_data_dict(
        root_data_dir=root_data_dir, 
        dataset_name=dataset_name, 
        feature_subdir=feature_subdir,
        mapping_file=mapping_file,
        target_components=target_components,
        video_list=train_video_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        ant_range=ant_range,
    )

    train_train_dataset = VideoFeatureDataset(train_data_dict, target_components, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, target_components, mode='test')

    if test_video_list:
        test_data_dict = get_data_dict(
            root_data_dir=root_data_dir, 
            dataset_name=dataset_name, 
            feature_subdir=feature_subdir,
            mapping_file=mapping_file,
            target_components=target_components,
            video_list=test_video_list, 
            sample_rate=sample_rate, 
            temporal_aug=temporal_aug,
            ant_range=ant_range,
        )
        test_test_dataset = VideoFeatureDataset(test_data_dict, target_components, mode='test')
    else:
        test_test_dataset = None

    if val_video_list:
        val_data_dict = get_data_dict(
            root_data_dir=root_data_dir, 
            dataset_name=dataset_name, 
            feature_subdir=feature_subdir,
            mapping_file=mapping_file,
            target_components=target_components,
            video_list=val_video_list, 
            sample_rate=sample_rate, 
            temporal_aug=temporal_aug,
            ant_range=ant_range,
        )
        val_test_dataset = VideoFeatureDataset(val_data_dict, target_components, mode='test')
    else:
        val_test_dataset = None


    num_targets = train_train_dataset.num_targets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    guidance_matrices = {}
    guidance_matrices['i_ivt'] = get_convert_matrix(mapping_file, 'i', 'ivt')
    guidance_matrices['v_ivt'] = get_convert_matrix(mapping_file, 'v', 'ivt')
    guidance_matrices['t_ivt'] = get_convert_matrix(mapping_file, 't', 'ivt')
    guidance_matrices['i_ivt'] = torch.from_numpy(guidance_matrices['i_ivt']).to(device).float().unsqueeze(0)
    guidance_matrices['v_ivt'] = torch.from_numpy(guidance_matrices['v_ivt']).to(device).float().unsqueeze(0)
    guidance_matrices['t_ivt'] = torch.from_numpy(guidance_matrices['t_ivt']).to(device).float().unsqueeze(0)


    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        causal, num_targets, sample_rate, temporal_aug, set_sampling_seed, guidance_matrices, ant_range,
        device=device
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, val_test_dataset,
                      loss_weights, class_weighting, num_epochs, batch_size, learning_rate, weight_decay,
                      result_dir=os.path.join(result_dir, naming),
                      log_freq=log_freq, log_train_results=log_train_results, log_APs=log_APs,
                      evaluation_protocol=evaluation_protocol
                      )