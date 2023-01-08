import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from minipeak.utils import load_training_dataset, save_wrong_pred_to_image, filter_data_window

# Write a python script using pytorch to train a CNN model that will detect the
# present of discriminative minis in a time window. The traning data is a set of
# panda frame having an 'amplitude' field and a 'minis' for each timestamp. The size
# of the time window is 100ms and the sampling rate is 1000Hz.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN model to detect discriminative minis in a time window")
    parser.add_argument('csv_folder', type=Path, help='path to the training data')
    parser.add_argument('exp_folder', type=Path, help='path to save experiment data')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--window-size', type=int, default=100, help='window size in ms')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')
    return parser.parse_args()


# Define the CNN model (ChatGPT)
class CNN(nn.Module):
    def __init__(self, window_size: int):
        super(CNN, self).__init__()
        self.window_size = window_size
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(64 * int(window_size / 2), 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, x):
        # Convolution layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def main() -> None:
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # create experiment folder if doesn't exist
    args.exp_folder.mkdir(parents=True, exist_ok=True)
    image_path = args.exp_folder / f'wrong_pred'
    image_path.mkdir(parents=True, exist_ok=True)

    # Create a pytorch Dataset from list of experiments csv files in folder. The 
    # timeseries will be split into chunks of data called 'windows'. The windows are
    # overlapping to make sure that the peaks are not truncated in a way that would
    # make it difficult for the model to detect them.
    all_X, all_y = load_training_dataset(args.csv_folder, args.window_size)
    # We want to have the same amount of windows that contain a minis and windows
    # that don't contain a minis to balance the learning.
    all_X, all_y = filter_data_window(all_X, all_y)

    all_X = torch.from_numpy(all_X).float()
    all_y = torch.from_numpy(all_y).float()
    dataset = torch.utils.data.TensorDataset(all_X, all_y)

    # Split dataset into training group and testing group. As we are using a small set
    # of data -> 80% training, 20% testing.
    training_size = int(0.8 * len(dataset))
    testing_size = len(dataset) - training_size
    train_data, test_data = \
        torch.utils.data.random_split(dataset, [training_size, testing_size])

    # Create dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize the model and optimizer
    model = CNN(window_size=args.window_size)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    
    # Define the loss function and evaluation metric
    loss_fn = nn.BCELoss() # nn.BCEWithLogitsLoss()
    def accuracy(y_pred, y_true):
        # y_pred = torch.sigmoid(y_pred)
        correct = (y_pred > 0.5) == (y_true > 0.5)
        return correct.float().mean()

    # train model
    for epoch in range(args.epochs):
        # Initialize the accuracy and loss for the epoch
        epoch_loss = 0
        epoch_acc = 0
        
        # Set the model to training mode
        model.train()
        
        # Loop over the training data
        for batch_idx, (X, y) in enumerate(train_data_loader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X)
            
            # Check if a minipeak is part of this batch
            y = torch.any(y[:, :, :], dim=2).float()
        
            # Compute loss and accuracy
            loss = loss_fn(y_pred, y)
            acc = accuracy(y_pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update the epoch loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # Print the epoch loss and accuracy
        print(f'Epoch {epoch+1}: loss={epoch_loss / (batch_idx+1):.4f}, '
              f'accuracy={epoch_acc / (batch_idx+1):.4f}')

    # Set the model to evaluation mode
    model.eval()

    # Initialize the validation loss and accuracy
    val_loss = 0
    val_acc = 0

    # Loop over the validation data
    for i, (x, y) in enumerate(test_data_loader):
        # Forward pass
        y_pred = model(x)

        # Check if a minipeak is part of this batch
        y = torch.any(y[:, :, :], dim=2).float()

        # Compute the loss and accuracy
        loss = loss_fn(y_pred, y)
        acc = accuracy(y_pred, y)

        # Update the validation loss and accuracy
        val_loss += loss.item()
        val_acc += acc.item()
        
        # Save the wrong prediction to image for investigation
        save_wrong_pred_to_image(X, y_pred, y, image_path)

    # Print the validation loss and accuracy
    print(f'Validation: loss={val_loss / (i+1):.4f}, accuracy={val_acc / (i+1):.4f}')

    # save model
    model_file = args.exp_folder / 'model.pt'
    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    main()
