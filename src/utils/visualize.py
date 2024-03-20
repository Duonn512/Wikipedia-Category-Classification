import matplotlib.pyplot as plt

def visualize_result(results, model_name, save_path=None):
    """
    results: A dictionary containing the following keys:
        - train_losses: A list of training losses
        - train_accs: A list of training accuracies
        - val_losses: A list of validation losses
        - val_accs: A list of validation accuracies
    model_name: The name of the model (string)
    save_path: If not None, save the plot to this path
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(results['train_losses'], label='Train')
    axs[0].plot(results['val_losses'], label='Validation')
    axs[0].set_title(f'Loss over epochs for {model_name}')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(results['train_accs'], label='Train')
    axs[1].plot(results['val_accs'], label='Validation')
    axs[1].set_title(f'Accuracy over epochs for {model_name}')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    if save_path:
        fig.savefig(save_path)