from dataset import NB15Dataset
import pandas as pd
import matplotlib.pyplot as plt
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

    fig, ax = plt.subplots()
    labels = ['Precision', 'Recall', 'F1 Score','Accuracy']
    colors = ['blue', 'green', 'orange', 'red']
    values = [precision, recall, f1, accuracy]
    x_pos = range(len(labels))
    ax.bar(x_pos, values, align='center', alpha=0.5,color = colors)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('Precision, Recall, F1 Score and Accuracy of Decision Tree')
    
    ax.set_ylabel('Scores')
    plt.tight_layout()
    plt.savefig("./plots/" + "DecisionTree_metrics.jpg",format='jpg')

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

    feature_names = feature_names[:-2]
    feature_importances = clf.feature_importances_
    features_df = pd.DataFrame(data={'feature': feature_names, 'importance': feature_importances})

    # Sort the DataFrame in descending order of importance
    features_df = features_df.sort_values('importance', ascending=False)

    #    Print the seven most important features
    top_features_df = features_df.head(7)

    # Create the bar plot
    plt.figure(figsize=(10,5))
    bars = plt.barh(top_features_df['feature'], top_features_df['importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 7 Important Features DecisionTree')
    plt.gca().invert_yaxis()
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.3f}', 
                 va='center', ha='left')
    plt.tight_layout()
    plt.savefig("./plots/" + "DecisionTree_feature.jpg",format='jpg')
    
    pred_y = []
    for x in tqdm(test_X, desc="Predicting", unit="samples"):
        y_pred = clf.predict([x])
        pred_y.append(y_pred[0])
    compute_all_metrics(pred_y, test_y)

