from dataset import NB15Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from models import AutoEncoder,VAE
import torch

def load_dataset(train_path, test_path, scalar_path, scaling_method):
    train_set = NB15Dataset(train_path, scalar_path, scaling_method, benign_only=False)
    test_set = NB15Dataset(test_path, scalar_path, scaling_method, benign_only=False)
    feature = pd.read_csv(train_path)
    feature_name = feature.columns
    return train_set.X, train_set.y, test_set.X, test_set.y,feature_name

def compute_all_metrics(pred_y, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=pred_y, average="binary")
    accuracy = accuracy_score(labels, pred_y)
    print("Best result in this experiment: ")
    print(f"precision: {precision:.4f}\t recall: {recall:.4f}\t F1: {f1:.4f}\t accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # configuration
    train_path = "./data/train_set.csv"
    test_path = "./data/test_set.csv"
    scalar_path = "./data/minmax_scalar.npy"
    scaling_method = "minmax"

    # load data
    train_X, train_y, test_X, test_y,feature_names= load_dataset(train_path, test_path, scalar_path, scaling_method)

    # load model 
    model = AutoEncoder()
    model.load_state_dict(torch.load("./models/ae.pth"))
    
    feature_names = feature_names[:-2]
    explainer = LimeTabularExplainer(train_X,
                                     feature_names=feature_names,
                                     class_names=['Reconstruction Error'],
                                     mode = 'regression')
    i = 25
    def predictfn(input_data):
        # Ensure the model is in evaluation mode
        model.eval()
    
        # Convert the input_data to torch Tensor
        input_data = torch.from_numpy(input_data.astype(np.float32))
    
        # Check if the input_data has the correct number of dimensions
        # if not, add an extra dimension for batch
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)
    
        # Get the model's predictions
        output_data = model(input_data)
    
        # The predictions are returned as a torch Tensor, but they need to be a numpy array
        # for lime. Also, squeeze the output to remove the batch dimension.
        return output_data.detach().numpy().squeeze()

    #Explain the instance's prediction using the predict function
    exp = explainer.explain_instance(test_X[i], predictfn, num_features=7)
    prediction = predictfn(test_X[i])
    
    # Plot the explanation and save it as a figure
    fig = exp.as_pyplot_figure()
    ax = plt.gca()
    ax.set_title(f"LIME Explanation. Model prediction: {prediction:.3f}")
    # Iterate through the bars (patches) and add text
    for rect in ax.patches:
        y_value = rect.get_y() + rect.get_height() / 2
        x_value = rect.get_width()
        ax.text(x_value, y_value, f'{x_value:.3f}', va='center')
    plt.tight_layout()
    plt.savefig("./plots/" + "AE_LIME.jpg", format='jpg')

