import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from minipeak.utils import \
    load_training_dataset, save_false_positives_to_image, save_false_negatives_to_image, \
    filter_data_window, create_experiment_folder, save_experiment_to_json, \
    save_training_resultsto_csv


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


# Define the 1D CNN model
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


@dataclass
class ValidationResults:
    nb_samples: int = 0
    sum_loss: float = 0
    sum_accuracy: float = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    
    def add_results(self, loss: float, accuracy: float, true_positives: int,
                    false_positives: int, true_negatives: int, false_negatives: int):
        self.nb_samples += 1
        self.sum_loss += loss
        self.sum_accuracy += accuracy
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.true_negatives += true_negatives
        self.false_negatives += false_negatives
    
    def loss(self) -> float:
        return self.sum_loss / self.nb_samples
    
    def accuracy(self) -> float:
        return self.sum_accuracy / self.nb_samples
    
    def precision(self) -> float:
        return float(self.true_positives) / (self.true_positives + self.false_positives)
    
    def recall(self) -> float:
        return float(self.true_positives) / (self.true_positives + self.false_negatives)


def main() -> None:
    args = _parse_args()
    device = \
        torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')

    experiment_folder = create_experiment_folder(args.exp_folder)

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
    
    no_peaks_zone = int(args.window_size/4)
    def contains_peaks(peak_window, no_peaks_padding: int):
        return torch.any(peak_window[:, :, no_peaks_padding:-no_peaks_padding],
                         dim=2).float()

    # Training results
    training_df = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])

    # Train model
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
            y = contains_peaks(y, no_peaks_padding=no_peaks_zone)
        
            # Compute loss and accuracy
            loss = loss_fn(y_pred, y)
            acc = accuracy(y_pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update the epoch loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # Epoch loss and accuracy
        training_df.loc[epoch] = [epoch+1, epoch_loss / (batch_idx+1), epoch_acc / (batch_idx+1)]
        print(f'Epoch {epoch+1}: loss={epoch_loss / (batch_idx+1):.4f}, '
              f'accuracy={epoch_acc / (batch_idx+1):.4f}')

    # Save training results to csv for plotting
    save_training_resultsto_csv(experiment_folder, training_df)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the validation loss, accuracy, recall and precision
    validation = ValidationResults()

    # Loop over the validation data
    for x, y in test_data_loader:
        # Forward pass
        y_pred = model(x)

        # Check if a minipeak is part of this batch
        y = contains_peaks(y, no_peaks_padding=no_peaks_zone)

        # Compute the loss and accuracy
        loss = loss_fn(y_pred, y)
        acc = accuracy(y_pred, y)
        true_pos, false_pos = save_false_positives_to_image(experiment_folder, x, y_pred, y)
        true_neg, false_neg = save_false_negatives_to_image(experiment_folder, x, y_pred, y)

        # Update the validation loss and accuracy
        validation.add_results(loss.item(), acc.item(), true_pos, false_pos,
                               true_neg, false_neg)

    # Compute final validation loss, accuracy, precision and recall
    print(f'\nValidation results:')
    print(f'loss      = {validation.loss():.4f}')
    print(f'accuracy  = {validation.accuracy():.4f}')
    print(f'precision = {validation.precision():.4f}')
    print(f'recall    = {validation.recall():.4f}')

    # Save the training hyperparameters and validation results
    save_experiment_to_json(experiment_folder, args.epochs, args.learning_rate,
                            args.weight_decay, args.window_size,
                            validation.loss(), validation.accuracy(),
                            validation.precision(), validation.recall())

    # save model
    model_file = args.exp_folder / 'model.pt'
    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    main()
