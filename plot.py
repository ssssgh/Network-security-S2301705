import argparse
from Experiment import Experiment
import json
from utils import AttributeAccessibleDict
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_density(ax, data_path, threshold, model_name):
    data = np.load(data_path)
    recon_errors = data[:, 0]
    labels = data[:, 1]
    normal_idx = labels == 0
    attack_idx = labels == 1
    normals = recon_errors[normal_idx]
    attacks = recon_errors[attack_idx]
    d = {
        "normal": normals,
        "attack": attacks
    }
    df = pd.DataFrame.from_dict(dict([ (k,pd.Series(v)) for k,v in d.items() ]))

    sns.histplot(df, x="normal", stat="density", kde=True, ax=ax, label="normal")
    sns.histplot(df, x="attack", stat="density", bins=70, kde=True, ax=ax, label="attack")
    ax.set_xlabel("reconstruction error", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.axvline(x=threshold, ymin=0, ymax=5, linestyle="--", color="red", label="threshold")
    ax.set_title(f"Density of the reconstruction errors by {model_name}", fontsize=12)
    ax.legend()

def plot_confusion_matrix(ax, data_path, threshold, model_name):
    data = np.load(data_path)
    recon_errors = data[:, 0]
    labels = data[:, 1]
    predictions = np.zeros_like(labels)
    predictions[recon_errors > threshold] = 1
    cm = confusion_matrix(labels, predictions)
    print(cm)
    cm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, ax=ax, square=True)
    ax.set_ylabel("True Labels", fontsize=12)
    ax.set_xlabel("Predicted Labels", fontsize=12)
    ax.set_title(f"Confusion Matrix normalized along each row ({model_name})")
    print(cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Anomaly Detection Exp")
    parser.add_argument("--config_file", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    config = AttributeAccessibleDict(config)
    
    # Load Exp class
    exp = Experiment(args=config)
    
    exp.model.load_state_dict(torch.load(f"{exp.save_dir}{exp.model_name}.pth", map_location=torch.device("cpu")))

    with torch.no_grad():
        recon_errors, labels = exp.get_recon_errors_and_labels(validate=True)
        epoch_f1, threshold = exp.compute_best_f1(0,recon_errors,labels, return_threshold=True)
        recon_errors, labels = exp.get_recon_errors_and_labels(validate=False)
        pred_data = torch.stack([recon_errors, labels], dim=1)
    pred_data = pred_data.cpu().numpy()
    np.save("vae_reconstruction.npy", pred_data)
    print((labels==1).sum())
    print((labels == 0).sum())
    print(epoch_f1)
    print(threshold)

    # ae_threshold = 2.9
    # vae_threshold = 5.34
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # plot_density(axes[0], "reconstruction.npy", ae_threshold, model_name="AE")
    # plot_density(axes[1], "vae_reconstruction.npy", vae_threshold, model_name="VAE")
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("./plots/" + "Reconstruction density.jpg",format='jpg')


    # ae_threshold = 2.9
    # vae_threshold = 5.34
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # plot_confusion_matrix(axes[0], "reconstruction.npy", ae_threshold, model_name="AE")
    # plot_confusion_matrix(axes[1], "vae_reconstruction.npy", vae_threshold, model_name="VAE")
    # plt.tight_layout()
    # plt.savefig("./plots/" + "Confusion matrix.jpg",format='jpg')