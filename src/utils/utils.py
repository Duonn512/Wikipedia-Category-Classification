import torch

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, return_results=False):
    """
    Args:
    model: The model to train
    train_loader: The DataLoader for the training set
    val_loader: The DataLoader for the validation set
    loss_fn: The loss function to use
    optimizer: The optimizer to use
    num_epochs: The number of epochs to train for
    device: The device to train the model on
    return_results: If True, return the training results

    Returns:
    If return_results is True, return a dictionary containing the following keys:
    """
    model.to(device)
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        # Train loop
        train_avg_loss = 0
        train_correct_pred = 0
        train_total_pred = 0

        for idx, (description, label) in enumerate(train_loader):
            optimizer.zero_grad()
            description = description.to(device)
            label = label.to(device)
            y_pred = model(description)
            batch_loss = loss_fn(y_pred, label.long())
            batch_loss.backward()
            optimizer.step()

            train_avg_loss += batch_loss.item()
            train_correct_pred += (torch.argmax(y_pred, 1) == label).sum().item()
            train_total_pred += len(label)
        
        train_avg_loss /= len(train_loader)
        train_losses.append(train_avg_loss)
        train_accs.append(train_correct_pred / train_total_pred)
        
        # Validation loop
        val_avg_loss = 0
        val_correct_pred = 0
        val_total_pred = 0

        with torch.no_grad():
            for idx, (description, label) in enumerate(val_loader):
                description = description.to(device)
                label = label.to(device)
                y_pred = model(description)
                batch_loss = loss_fn(y_pred, label.long())

                val_avg_loss += batch_loss.item()
                val_correct_pred += (torch.argmax(y_pred, 1) == label).sum().item()
                val_total_pred += len(label)

        val_avg_loss /= len(val_loader)
        val_losses.append(val_avg_loss)
        val_accs.append(val_correct_pred / val_total_pred)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Train Acc: {train_correct_pred / train_total_pred:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_correct_pred / val_total_pred:.4f}")

    if return_results:
        results = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs
        }
        return results

def evaluate_model(model, test_loader, device, plot_confusion_matrix=True):
    model.eval()
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for description, label in test_loader:
            description = description.to(device)
            label = label.to(device)
            outputs = model(description)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(label.to('cpu').numpy())
            all_predictions.extend(predicted.to('cpu').numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    metrics = {
        'accuracy': accuracy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }
    
    if plot_confusion_matrix:
        cm = confusion_matrix(all_labels, all_predictions)        
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True  labels')
        plt.xticks(np.arange(4) + 0.5, ["Khoa học tự nhiên","Khoa học xã hội","Kỹ thuật","Văn hóa"], rotation=30)
        plt.yticks(np.arange(4) + 0.5, ["Khoa học tự nhiên","Khoa học xã hội","Kỹ thuật","Văn hóa"], rotation=30)
        plt.show()

    
    return metrics
