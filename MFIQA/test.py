import torch
from torch.utils.data import DataLoader
from MFIQAdatatest import MFIQADataset
from MFIQA import MFIQAModule
from utils import performance_fit
import numpy as np

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze()
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().tolist())
            targets.extend(labels.cpu().tolist())

        predictions = np.array(predictions)
        targets = np.array(targets)
        srcc, plcc, krcc, rmse_value = performance_fit(predictions, targets)
        print(f'Test Metrics - SRCC: {srcc:.4f}, PLCC: {plcc:.4f}, KRCC: {krcc:.4f}, RMSE: {rmse_value:.4f}')    

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'AGIQA-3k'
    checkpoints_dir = ''

    # Load test data
    test_dataset = MFIQADataset(dataset, 'data')
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Model
    model = MFIQAModule().to(device)

    # Load trained model weights
    model.load_state_dict(torch.load(checkpoints_dir))
    
    # Test the model
    test(model, test_loader, device)

if __name__ == '__main__':
    main()
