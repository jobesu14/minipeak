import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from minipeak.models.cnn_model import CNN
from minipeak.preprocessing import load_training_datasets
from minipeak.utils import \
    save_false_positives_to_image, save_false_negatives_to_image, \
    filter_windows, create_experiment_folder, save_experiment_to_json, \
    save_training_results_to_csv
# from minipeak.visualization import plot_training_curves


def _parse_args() -> argparse.Namespace:
    """ Parse the arguments provided through the command line. """
    parser = argparse.ArgumentParser(
        description="Train a CNN model to detect discriminative minis in a time window")
    parser.add_argument('csv_folder', type=Path, help='path to the training data')
    parser.add_argument('exp_folder', type=Path, help='path to save experiment data')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--window-size', type=int, default=80, help='window size in ms')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')
    return parser.parse_args()


@dataclass
class ValidationResults:
    """
    Store the results of the validation of the model to compute the validation set
    metrics.
    """
    nb_samples: int = 0
    sum_loss: float = 0
    sum_accuracy: float = 0
    sum_pos_err: float = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    true_negatives: int = 0

    def add_results(self, loss: float, accuracy: float, pos_err: float,
                    true_positives: int, false_positives: int, true_negatives: int,
                    false_negatives: int) -> None:
        self.nb_samples += 1
        self.sum_loss += loss
        self.sum_accuracy += accuracy
        self.sum_pos_err += pos_err
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.true_negatives += true_negatives
        self.false_negatives += false_negatives

    def loss(self) -> float:
        return self.sum_loss / self.nb_samples

    def accuracy(self) -> float:
        return self.sum_accuracy / self.nb_samples

    def position_error(self) -> float:
        return self.sum_pos_err / self.nb_samples

    def precision(self) -> float:
        return float(self.true_positives) / (self.true_positives + self.false_positives)

    def recall(self) -> float:
        return float(self.true_positives) / (self.true_positives + self.false_negatives)


def _training_data(csv_folder: Path, window_size: int, batch_size: int) \
        -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create a pytorch Dataset from list of experiments csv files in folder. The
    timeseries will be split into chunks of data called 'windows'. The windows are
    overlapping to make sure that the peaks are not truncated in a way that would
    make it difficult for the model to detect them.

    :param csv_folder: path to the folder containing the training csv files with the
    following columns: 'time', 'amplitude', 'mins' (= peak amplitude)
    :param window_size: size of the window used to predict the peak positions
    :batch_size: size of the batch used for training
    :return: a tuple of two dataloaders, one for the training set and one for the
    evaluation set.
    """
    all_x, all_y = load_training_datasets(csv_folder, window_size)
    # We want to have the same amount of windows that contain a minis and windows
    # that don't contain a minis to balance the learning.
    all_x, all_y = filter_windows(all_x, all_y)

    all_x = torch.from_numpy(all_x).float()
    all_y = torch.from_numpy(all_y).float()
    dataset = torch.utils.data.TensorDataset(all_x, all_y)

    # Split dataset into training group and testing group. As we are using a small set
    # of data -> 80% training, 20% testing.
    training_size = int(0.8 * len(dataset))
    testing_size = len(dataset) - training_size
    train_data, test_data = \
        torch.utils.data.random_split(dataset, [training_size, testing_size])

    # Create dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_data_loader, test_data_loader


def _peaks_info(peak_window: torch.Tensor, no_peaks_padding: int) -> torch.Tensor:
    """
    Add a "dead zone" around the edges of the window to avoid detecting peaks that
    are truncated by the window's borders.

    :param peak_window: tensor containing the windowed data of the timeseries.
    :param no_peaks_padding: number of samples to ignore at the edges of the window.
    :return: a tensor containing the filtered peaks information within the winodw.
    """
    has_peaks = \
        torch.any(peak_window[:, :, no_peaks_padding:-no_peaks_padding], dim=2).float()
    peak_pos = \
        torch.argmax(peak_window[:, :, no_peaks_padding:-no_peaks_padding], dim=2).float()
    peak_pos = (no_peaks_padding + peak_pos) / peak_window.shape[2]
    return torch.stack((has_peaks, peak_pos), dim=1).reshape(-1, 2)


def _loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss of the model. The loss is composed of two parts:
    - the class loss (window contains a peak or not)
    - the mean squared error (peak position in window) if a peak is present

    :param y_pred: tensor containing the predictions of the model (peak or no peak)
    :param y_true: tensor containing the ground truth (peak or no peak)
    :return: the loss of the model
    """
    # Class loss (window contains a peak or not).
    loss = nn.BCELoss()(y_pred[:, 0], y_true[:, 0])

    # Mean squared loss (peak position in window) if a peak is present.
    loss += nn.MSELoss()(y_pred[:, 1], y_true[:, 1])
    return loss


def _class_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """ Compute the accuracy of the model on the class prediction. """
    correct = (y_pred[:, 0] > 0.5) == (y_true[:, 0] > 0.5)
    return correct.float().mean()


def _position_error(x: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) \
        -> torch.Tensor:
    """
    Compute the peak position prediction error within the window. The root mean squared
    error is computed on the percentage of the window where the peak is located.
    Note: the window not containing a peak has its peak position set to 0 (within window
    peak dead zone) to make sure it doesn't impact the positions lost.

    :param x: tensor containing the windowed data of the timeseries
    :param y_pred: tensor containing the predictions of the model (peak position)
    :param y_true: tensor containing the ground truth (peak position)
    """
    # Get the percentage of the window where the peaks are located.
    t_pred_perc = y_pred[:, 1]
    t_true_perc = y_true[:, 1]

    # Get the amplitude values of the peaks.
    batch_size = x.shape[0]
    index_pred = torch.mul(t_pred_perc, x.shape[2]).reshape([batch_size, 1, 1]).long()
    index_true = torch.mul(t_true_perc, x.shape[2]).reshape([batch_size, 1, 1]).long()
    amp_pred = torch.take_along_dim(x, indices=index_pred, dim=2).reshape(batch_size)
    amp_true = torch.take_along_dim(x, indices=index_true, dim=2).reshape(batch_size)

    # Combine the time/amplitude of the predictions and targets.
    pred = torch.stack((t_pred_perc, amp_pred), dim=1)
    target = torch.stack((t_true_perc, amp_true), dim=1)

    # Compute the mean squared error between the time/amplitude coordinates of the
    # predicted peak and the target peak.
    error = nn.MSELoss()(pred, target)
    return error.float().mean()


def _train(experiment_folder: Path, model: nn.Module, optimizer: optim.Optimizer,
           train_loader: torch.utils.data.DataLoader, epochs: int, no_peaks_zone: int,
           device: torch.device) -> None:
    """
    Main loop to train the model.

    :param experiment_folder: folder where the training results will be saved into a csv
    file containing the model loss and accuracy for each epoch.
    :param model: model to train.
    :param optimizer: optimizer to use for the training.
    :param train_loader: data loader for the training data.
    :param epochs: number of epochs to train the model.
    :param no_peaks_zone: number of samples to ignore at both edges of the window.
    :param device: device to use for the training.
    """
    # Training results
    training_df = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])

    # Set the model to training mode
    model.train()

    # Train model
    for epoch in range(epochs):
        # Initialize the accuracy and loss for the epoch
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_pos_err = 0.0

        # Loop over the training data
        for batch_idx, (x, y) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Move data to device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x)

            # Check if a minipeak is part of this batch
            y = _peaks_info(y, no_peaks_padding=no_peaks_zone)

            # Move data to cpu
            x = x.cpu()
            y = y.cpu()
            y_pred = y_pred.cpu()

            # Compute loss and accuracy
            loss = _loss(y_pred, y)
            acc = _class_accuracy(y_pred, y)
            pos_err = _position_error(x, y_pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the epoch loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_pos_err += pos_err.item()

        # Epoch loss and accuracy
        training_df.loc[epoch] = [epoch+1, epoch_loss / (batch_idx+1),
                                  epoch_acc / (batch_idx+1)]
        logging.info(f'Epoch {epoch+1}: loss={epoch_loss / (batch_idx+1):.4f}, '
                     f'accuracy={epoch_acc / (batch_idx+1):.4f}, '
                     f'position error={epoch_pos_err / (batch_idx+1):.4f}')

    # Plot and save training results
    save_training_results_to_csv(experiment_folder, training_df)
    # plot_training_curves(training_df)


def _evaluate(experiment_folder: Path, model: nn.Module,
              test_data_loader: torch.utils.data.DataLoader, no_peaks_zone: int,
              device: torch.device) -> ValidationResults:
    """
    Main loop to evaluate the model.

    :param experiment_folder: folder where the false positive windows and the false
    negative windows will be saved.
    :param model: model to train.
    :param test_data_loader: data loader for the training data.
    :param no_peaks_zone: number of samples to ignore at both edges of the window.
    :param device: device to use for the training.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize the validation loss, accuracy, recall and precision
    validation = ValidationResults()

    # Loop over the validation data
    for x, y in test_data_loader:
        # Move data to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model(x)

        # Check if a minipeak is part of this batch
        y = _peaks_info(y, no_peaks_padding=no_peaks_zone)

        # Move data to cpu
        x = x.cpu()
        y = y.cpu()
        y_pred = y_pred.cpu()

        # Compute the loss and accuracy
        loss = _loss(y_pred, y)
        acc = _class_accuracy(y_pred, y)
        pos_err = _position_error(x, y_pred, y)
        true_pos, false_pos = save_false_positives_to_image(experiment_folder, x,
                                                            y_pred, y.numpy())
        true_neg, false_neg = save_false_negatives_to_image(experiment_folder, x,
                                                            y_pred, y.numpy())

        # Update the validation loss and accuracy
        validation.add_results(loss.item(), acc.item(), pos_err.item(), true_pos,
                               false_pos, true_neg, false_neg)

    # Compute final validation loss, accuracy, precision and recall
    logging.info('\nValidation results:')
    logging.info(f'loss      = {validation.loss():.4f}')
    logging.info(f'accuracy  = {validation.accuracy():.4f}')
    logging.info(f'pos error = {validation.position_error():.4f}')
    logging.info(f'precision = {validation.precision():.4f}')
    logging.info(f'recall    = {validation.recall():.4f}')

    return validation


def main() -> None:
    """ Main application when run from the command line interface. """
    logging.basicConfig(level='INFO')
    args = _parse_args()

    device = \
        torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f'Using device: {device}')

    experiment_folder = create_experiment_folder(args.exp_folder)

    # Create the training and validation datasets
    train_data_loader, test_data_loader = \
        _training_data(args.csv_folder, args.window_size, args.batch_size)

    # Initialize the model, optimizer and loss function
    model = CNN(window_size=args.window_size).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    # Define window area where peaks are ignored
    no_peaks_zone = int(args.window_size/4)

    # Train the model
    _train(experiment_folder, model, optimizer, train_data_loader, args.epochs,
           no_peaks_zone, device)

    # Evaluate the model
    validation = _evaluate(experiment_folder, model, test_data_loader, no_peaks_zone,
                           device)

    # Save the training hyperparameters and validation results
    save_experiment_to_json(experiment_folder, args.epochs, args.learning_rate,
                            args.weight_decay, args.window_size,
                            validation.loss(), validation.accuracy(),
                            validation.position_error(), validation.precision(),
                            validation.recall())

    # Save model
    model_file = experiment_folder / 'model.pt'
    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    """ Main application when run as a script. """
    main()
