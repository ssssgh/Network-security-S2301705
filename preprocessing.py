import numpy as np
import pandas as pd

biased_features = ["srcip", "sport", "dstip", "dsport", "stcpb", "dtcpb", "Stime", "Ltime"]
onehot_features = ["proto", "state", "service"]


def insert_feature_name_to_dataset(raw_file_path, feature_name_path, output_path):
    with open(feature_name_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    feature_names = [line.strip() for line in lines]
    feature_line = ",".join(feature_names) + "\n"
    with open(output_path, 'w',encoding='utf-8') as out_f:
        out_f.write(feature_line)
        in_f = open(raw_file_path, 'r',encoding='utf-8')
        while line := in_f.readline():
            out_f.write(line)
        in_f.close()


def clean_dataset(dataset_path, data_dir):
    dataset = pd.read_csv(dataset_path)

    # drop biased columns
    print(f"num of column: {len(dataset.columns)}")
    dataset.drop(biased_features, axis=1, inplace=True, errors="ignore")
    print(f"num of column after dropping: {len(dataset.columns)}")

    # one-hot encoding
    onehot_df = dataset[onehot_features]
    numerical_df = dataset.drop(onehot_features, axis=1)
    dfs = []
    for col_name in onehot_df.columns:
        print(f"found {len(onehot_df[col_name].unique())} categories in {col_name}")
        print(onehot_df[col_name].unique())
        encoded_df = pd.get_dummies(onehot_df[col_name])
        dfs.append(encoded_df)
    dfs.append(numerical_df)
    encoded_dataset = pd.concat(dfs, axis=1)
    print(f"num of column after encoding: {len(encoded_dataset.columns)}")

    # replace nan in attack_cat with "Normal"
    encoded_dataset["attack_cat"] = encoded_dataset["attack_cat"].replace(np.nan, 'Normal')
 
    # train/eval/test split
    encoded_dataset = encoded_dataset.sample(frac=1).reset_index(drop=True)
    train_prop = 0.7
    eval_prop = test_prop = 0.15
    train_num = int(len(encoded_dataset) * train_prop)
    eval_num = int(len(encoded_dataset) * eval_prop)
    test_num = len(encoded_dataset) - train_num - eval_num
    print("performing train/eval/test split")
    print(f"train_num: {train_num}\t eval_num: {eval_num}\t test_num: {test_num}")
    train_set = encoded_dataset.iloc[:train_num]
    eval_set = encoded_dataset.iloc[train_num:train_num+eval_num]
    test_set = encoded_dataset.iloc[train_num+eval_num:]

    print("saving datasets...")
    train_set.to_csv(data_dir + "train_set.csv", index=False)
    eval_set.to_csv(data_dir + "eval_set.csv", index=False)
    test_set.to_csv(data_dir + "test_set.csv", index=False)

    # compute min-max scalar
    data = encoded_dataset.iloc[:, :-2].to_numpy(dtype=np.float32)
    x_max = data.max(axis=0).reshape((1, -1))
    x_min = data.min(axis=0).reshape((1, -1))
    print(f"scalar dim: {x_max.shape}")
    scalar = np.concatenate([x_max, x_min], axis=0)
    np.save(data_dir + "minmax_scalar.npy", scalar)

    # compute standard scalar
    data = encoded_dataset.iloc[:, :-2].to_numpy(dtype=np.float32)
    x_mean = data.mean(axis=0).reshape((1, -1))
    x_std = data.std(axis=0).reshape((1, -1))
    scalar = np.concatenate([x_mean, x_std], axis=0)
    np.save(data_dir + "standard_scalar.npy", scalar)


if __name__ == "__main__":
    data_dir = "./data/"
    raw_file_path = "./data/UNSW-NB15_1.csv"
    feature_path = "./data/feature_names.txt"
    dataset_path = "./data/dataset_with_feature_names.csv"
    insert_feature_name_to_dataset(raw_file_path, feature_path, dataset_path)
    clean_dataset(dataset_path, data_dir)
