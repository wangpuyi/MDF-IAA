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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.tensorboard import SummaryWriter

def val(model, val_loader, device, best_srcc, dataset, epoch):
    model.eval()
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

    n_val = len(val_loader)
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
        # print('predictions: ', predictions.shape)
        # print('targets: ', targets.shape)
        srcc, plcc, krcc, rmse_value = performance_fit(predictions, targets)
        print(f'Validation Metrics - SRCC: {srcc:.4f}, PRCC: {plcc:.4f}, KRCC: {krcc:.4f}, RMSE: {rmse_value:.4f}')    
        if srcc > best_srcc:
            best_srcc = srcc
            torch.save(model.state_dict(), checkpoint_dir)
            print('Best model saved.')
    return best_srcc

def train(dataset, type_train, type_val, learning_rate, batch_size, num_epochs, decay_epoch, decay_rate):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Load Data
    train_dataset = MFIQADataset(dataset, type_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MFIQADataset(dataset, type_val) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = MFIQAModule().to(device)
    # model.load_state_dict(torch.load('module_checkpoint/ImageReward/best_model_way1_99.pth'))

    # Loss and optimizer
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=decay_rate)

    # Training loop
    best_srcc = 0
    for epoch in range(num_epochs):
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
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        scheduler.step()
        # Validate after each epoch
        best_srcc = val(model, val_loader, device, best_srcc, dataset, epoch)

    print('Training finished.')

def main():
    # Hyperparameters
    dataset = 'ImageReward'
    type_train = 'train'
    type_val = 'validation'
    learning_rate = 0.00001
    batch_size = 64
    num_epochs = 100
    decay_epoch = 20
    decay_rate = 0.9
    
    # Test the model
    train(dataset, type_train, type_val, learning_rate, batch_size, num_epochs, decay_epoch, decay_rate)


if __name__ == '__main__':
    main()
