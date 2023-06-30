from dataset import NB15Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from models import AutoEncoder,VAE
import torch

def load_dataset(train_path, test_path, scalar_path, scaling_method):
    train_set = NB15Dataset(train_path, scalar_path, scaling_method, benign_only=False)
    test_set = NB15Dataset(test_path, scalar_path, scaling_method, benign_only=False)
    feature = pd.read_csv(train_path)
    feature_name = feature.columns
    return train_set.X, train_set.y, test_set.X, test_set.y,feature_name

if __name__ == "__main__":
    # configuration
    train_path = "./data/train_set.csv"
    test_path = "./data/test_set.csv"
    scalar_path = "./data/minmax_scalar.npy"
    scaling_method = "minmax"

    # load data
    train_X, train_y, test_X, test_y,feature_names= load_dataset(train_path, test_path, scalar_path, scaling_method)

    # load model 
    model = VAE()
    model.load_state_dict(torch.load("./models/vae.pth"))
    
    feature_names = feature_names[:-2]

# explain the model's predictions using SHAP
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
    
    # use kmeans to summarize the dataset
    X_summary = shap.kmeans(train_X, 10)

# use the summary as background dataset for KernelExplainer
    explainer = shap.KernelExplainer(predictfn, X_summary)

# compute the SHAP values
    shap_values = explainer.shap_values(test_X[:3000], nsamples=100)

# visualize the explanation
    shap.summary_plot(shap_values, test_X[:3000], feature_names=feature_names)

    shap.force_plot(
        explainer.expected_value, shap_values[2], test_X[2],
        feature_names=feature_names,
        matplotlib=True, show=False
    )
    plt.savefig("./plots/" + "VAE_SHAP_0.jpg", format='jpg')

