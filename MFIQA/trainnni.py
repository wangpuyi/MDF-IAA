import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MFIQAdatatest import MFIQADataset 
from MFIQA import MFIQAModule           
from tqdm import tqdm
from utils import performance_fit
import numpy as np
import os
import argparse
import nni
from nni.utils import merge_parameter
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def val(model, val_loader, device):
    model.eval()

    with torch.no_grad():
        predictions = []
        targets = []
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze()
            outputs = model(inputs).squeeze()
            # print('outputs: ', outputs)
            predictions.extend(outputs.cpu().tolist())
            targets.extend(labels.cpu().tolist())
        predictions = np.array(predictions)
        targets = np.array(targets)

        srcc, plcc, krcc, rmse_value = performance_fit(predictions, targets)
        print(f'Validation Metrics - SRCC: {srcc:.4f}, PRCC: {plcc:.4f}, KRCC: {krcc:.4f}, RMSE: {rmse_value:.4f}')

    return srcc

def train(args):
    dataset = args['dataset']
    type_train = args['type_train']
    type_val = args['type_val']
    # learning_rate = args['learning_rate']
    # batch_size = args['batch_size']
    # num_epochs = args['num_epochs']
    decay_epoch = args['decay_epoch']
    decay_rate = args['decay_rate']

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Load Data
    train_dataset = MFIQADataset(dataset, type_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)

    val_dataset = MFIQADataset(dataset, type_val) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Model
    model = MFIQAModule().to(device)
    # model.load_state_dict(torch.load('module_checkpoint/ImageReward/best_model_way1_99.pth'))

    # Loss and optimizer
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=decay_rate)

    # Training loop
    best_srcc = 0
    for epoch in range(args['num_epochs']):
        model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0: #取100(每6400张pring一次)
                num_epochs = args['num_epochs']
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        scheduler.step()
        # Validate after each epoch
        val_srcc = val(model, val_loader, device)

        if dataset == 'ImageReward':
            root_dir = 'module_checkpoint'+ '/' + str(dataset)
            checkpoint_dir = root_dir + '/' + 'best_model_way1_{}.pth'.format(epoch)
        elif dataset == 'LAION':
            root_dir = 'module_checkpoint'+ '/' + str(dataset)
            checkpoint_dir = root_dir + '/' + 'best_model_way1_{}.pth'.format(epoch)
        elif dataset == 'test':
            root_dir = 'module_checkpoint'+ '/' + str(dataset)
            checkpoint_dir = root_dir + '/' + 'best_model_way1_{}.pth'.format(epoch)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        if val_srcc > best_srcc:
            best_srcc = val_srcc
            torch.save(model.state_dict(), checkpoint_dir)
            print('Best model saved.')
        
        nni.report_intermediate_result(val_srcc)
    nni.report_final_result(val_srcc)

    print('Training finished.')

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ImageReward')
    parser.add_argument('--type_train', type=str, default='train')
    parser.add_argument('--type_val', type=str, default='validation')
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=20)
    parser.add_argument('--decay_rate', type=float, default=0.9)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # main()
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(get_params(), tuner_params))
    train(params)
