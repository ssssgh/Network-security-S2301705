from dataset import NB15Dataset
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

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
    train_X, train_y, test_X, test_y,feature_names = load_dataset(train_path, test_path, scalar_path, scaling_method)

    # fit and test
    clf = RandomForestClassifier()
    clf = clf.fit(train_X, train_y)

    feature_names = feature_names[:-2]

    X_summary = shap.kmeans(train_X, 10)

    def predict_proba_class_1(X):
        return clf.predict_proba(X)[:, 1]  

# use the summary as background dataset for KernelExplainer
    explainer = shap.KernelExplainer(predict_proba_class_1, X_summary)

# compute the SHAP values
    shap_values = explainer.shap_values(test_X[:2000],nsamples = 100)

# visualize the explanation
    shap.summary_plot(shap_values, test_X[:2000],feature_names = feature_names)

    shap.force_plot(
        explainer.expected_value, shap_values[2], test_X[2],
        feature_names=feature_names,
        matplotlib=True, show=False
    )
    plt.savefig("./plots/" + "RF_SHAP_0.jpg", format='jpg')
