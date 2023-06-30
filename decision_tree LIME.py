from dataset import NB15Dataset
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

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

    # fit and test
    clf = DecisionTreeClassifier()
    clf = clf.fit(train_X, train_y)
    
    pred_y = []
    for x in tqdm(test_X, desc="Predicting", unit="samples"):
        y_pred = clf.predict([x])
        pred_y.append(y_pred[0])
    compute_all_metrics(pred_y, test_y)

    feature_names = feature_names[:-2]
    explainer = LimeTabularExplainer(train_X,
                                     feature_names=feature_names, # the names of the features
                                     class_names=['Normal', 'Attack'])
    i = 25

    # Explain the instance's prediction
    exp = explainer.explain_instance(test_X[i], clf.predict_proba, num_features=10)

    # Plot the explanation and save it as a figure
    fig = exp.as_pyplot_figure()
    ax = plt.gca()

    # Iterate through the bars (patches) and add text
    for rect in ax.patches:
        y_value = rect.get_y() + rect.get_height() / 2
        x_value = rect.get_width()
        ax.text(x_value, y_value, f'{x_value:.3f}', va='center')
    plt.tight_layout()
    plt.savefig("./plots/" + "DT_LIME.jpg", format='jpg')

