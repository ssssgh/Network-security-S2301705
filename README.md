### Dependencies
```
pytorch
numpy
matplotlib
tqdm
scikit-learn
pandas
lime
shap
```

### How to run AE and VAE
Preprocess the dataset by running 
```shell
python3 preprocessing.py
```
Ensure `UNSW-NB15_1.csv` and `feature_names.txt` 
is available under `./data/`.

The config files of AE and VAE is saved under `./configs`.
Replace `<ROOT_DIR>` in each config file with the actual path of this
directory and then run
```shell
python3 main.py --config_file ./configs/<filename>
```
If GPU is not available, it is recommended to use a smaller `batch_size`, 
e.g., 32 or 64, to avoid large memory consumption.

`main.py` runs a standard experiment and stores the model 
with the best F1 score on validation set. At the end of 
the experiment, the script loads the saved the model, test it on the testing set, 
and report precision, recall, F1 score and accuracy.
Since the dataset is imbalanced, accuracy can be very misleading.


Ensure that you have preprocessed the dataset by
```shell
python3 preprocessing.py
```
and then run the following.

### How to run Decision Tree and Decision Tree SHAP
```shell
python3 decision_tree.py
```
```shell
python3 decision_tree SHAP.py
```

### How to run Random Forest and Random Forest SHAP
```shell
python3 Randomforest.py
```
```shell
python3 Randomforest SHAP.py
```

### How to run SVM and SVM SHAP
Notice that for SVM it will take a lot of time to train
the model,so be patience.
```shell
python3 SVM.py
```
```shell
python3 SVM SHAP.py
```

### How to run SHAP for AE and VAE
This become a little bit complex.Firstly, after you have trained the AE and VAE model, 
you can see two `.pth` file under `./models/`,these contain optimal parameter for AE
and VAE model.Since AE and VAE are not classification model,we need to change the output
in our model to process LIME.Find `models.py`, you can see `forward` function in both AE
and VAE, we need to return the reconstruction error instead of `Xhat` and `X`.In the code 
I have written comments.

Next,we pay attention to `SHAP_AEorVAE.py`, you can see there is code like:
`model = AutoEncoder()`
`model.load_state_dict(torch.load("./models/ae.pth"))`
Here you can change the model between `AutoEncoder()/VAE()` and `ae.pth/vae.pth` to choose which
model you want to apply in LIME.
Dont forget to change the picture name at last!
`plt.savefig("./plots/" + "AE_LIME.jpg", format='jpg')`
`AE_SHAP/VAE_SHAP`
