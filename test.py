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
import pdb
from main import Trainer

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
    )

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
        )
        val_test_dataset = VideoFeatureDataset(val_data_dict, target_components, mode='test')
    else:
        val_test_dataset = None


    num_targets = train_test_dataset.num_targets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    guidance_matrices = {}
    guidance_matrices['i_ivt'] = get_convert_matrix(mapping_file, 'i', 'ivt')
    guidance_matrices['v_ivt'] = get_convert_matrix(mapping_file, 'v', 'ivt')
    guidance_matrices['t_ivt'] = get_convert_matrix(mapping_file, 't', 'ivt')
    guidance_matrices['i_ivt'] = torch.from_numpy(guidance_matrices['i_ivt']).to(device).float().unsqueeze(0)
    guidance_matrices['v_ivt'] = torch.from_numpy(guidance_matrices['v_ivt']).to(device).float().unsqueeze(0)
    guidance_matrices['t_ivt'] = torch.from_numpy(guidance_matrices['t_ivt']).to(device).float().unsqueeze(0)


    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        causal, num_targets, sample_rate, temporal_aug, set_sampling_seed, guidance_matrices,
        device=device
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    eval_result_dir = os.path.join(result_dir, naming)

    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)

    mode = evalonly_params['mode']

    for epoch in evalonly_params['epochs']:

        model_path = os.path.join(result_dir, evalonly_params['pretrain_naming'], f'epoch-{epoch}.model')

        print('Evaluating:', model_path)

        if test_test_dataset:

            test_result_dict = trainer.test(
                test_test_dataset, mode, log_APs, evaluation_protocol, model_path=model_path)

            np.save(os.path.join(eval_result_dir, 
                f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

            for k,v in test_result_dict.items():
                print(f'Epoch {epoch} - {mode}-Test-{k} {v}')

        if val_test_dataset:

            val_result_dict = trainer.test(
                val_test_dataset, mode, log_APs, evaluation_protocol, model_path=model_path)

            np.save(os.path.join(eval_result_dir, 
                f'val_results_{mode}_epoch{epoch}.npy'), val_result_dict)

            for k,v in val_result_dict.items():
                print(f'Epoch {epoch} - {mode}-Val-{k} {v}')

        if log_train_results:

            train_result_dict = trainer.test(
                train_test_dataset, mode, log_APs, evaluation_protocol, model_path=model_path)

            np.save(os.path.join(eval_result_dir, 
                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)

            for k,v in train_result_dict.items():
                print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
