import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Timer
from models import VAE, AutoEncoder
from dataset import NB15Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_recall_curve

class Experiment:
    def __init__(self, args):
        config_timer = Timer("Exp initialization")
        self.model_name = args.model_name
        self.save_dir = args.ROOT_DIR + args.save_dir
        self.plot_dir = args.ROOT_DIR + args.plot_dir
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.scaling_method = args.scaling_method

        self.model = None
        self.optimizer = None

        self.train_loader, self.val_loader, self.test_loader = self.init_loaders(
            args.ROOT_DIR + args.train_path,
            args.ROOT_DIR + args.val_path,
            args.ROOT_DIR + args.test_path,
            args.ROOT_DIR + args.scalar_path,
            args.scaling_method
        )
        self.model, self.optimizer = self.init_model_optimizer()

        self.train_losses = {
            "loss": [],
            "CE": [],
            "KLD": []
        }

        self.test_losses = {
            "loss": [],
            "CE": [],
            "KLD": []
        }

        config_timer.end()

    def init_model_optimizer(self):
        if self.model_name == "vae":
            model = VAE()
        elif self.model_name == "ae":
            model = AutoEncoder()
        else:
            raise ValueError(f"unknown model name: {self.model_name}")
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return model, optimizer

    def init_loaders(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        scalar_path: str,
        scaling_method: str
    ):
        train_loader = DataLoader(
            NB15Dataset(train_path, scalar_path, scaling_method, benign_only=True),
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True
        )
        val_loader = DataLoader(
            NB15Dataset(val_path, scalar_path, scaling_method, benign_only=False),
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True
        )
        test_loader = DataLoader(
            NB15Dataset(test_path, scalar_path, scaling_method, benign_only=False),
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True
        )
        return train_loader, val_loader, test_loader

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        ce_loss = 0
        kld_loss = 0
        counter = 0
        for model_input in tqdm(self.train_loader):
            counter += 1
            model_input = self.to_device(model_input)
            self.optimizer.zero_grad()
            model_output = self.feed_to_model(model_input)
            loss = self.loss_function(model_output)
            total_loss = loss["loss"]
            train_loss += total_loss.item()
            total_loss.backward()
            self.optimizer.step()

            # log other terms
            if self.model_name == "vae":
                ce_loss += loss["CE"].item()
                kld_loss += loss["KLD"].item()

        train_loss /= counter
        loss_str = f'====> Epoch: {epoch} Average loss: {train_loss:.4f}'
        if self.model_name == "vae":
            ce_loss /= counter
            kld_loss /= counter
            loss_str += f", CE loss: {ce_loss:.4f}, KLD: {kld_loss:.4f} \n"
            print(loss_str)
            return train_loss, ce_loss, kld_loss
        else:
            loss_str += "\n"
            print(loss_str)
            return train_loss

    def test(self, validate=True):
        self.model.eval()
        loader = self.val_loader if validate else self.test_loader
        test_loss = 0
        ce_loss = 0
        kld_loss = 0
        counter = 0
        with torch.no_grad():
            for model_input in tqdm(loader):
                counter += 1
                model_input = self.to_device(model_input)
                model_output = self.feed_to_model(model_input)
                loss = self.loss_function(model_output)
                total_loss = loss["loss"]
                test_loss += total_loss.item()

                # log other terms
                if self.model_name == "vae":
                    ce_loss += loss["CE"].item()
                    kld_loss += loss["KLD"].item()

        test_loss /= counter
        loss_str = f'====> Test average loss: {test_loss:.4f}'
        if self.model_name == "vae":
            ce_loss /= counter
            kld_loss /= counter
            loss_str += f", CE loss: {ce_loss:.4f}, KLD: {kld_loss:.6f} \n"
            print(loss_str)
            return test_loss, ce_loss, kld_loss
        else:
            loss_str += "\n"
            print(loss_str)
            return test_loss

    def loss_function(self, model_output):
        if self.model_name == "vae":
            X_hat, X, mu, log_var = model_output
            # MSE is cross entropy between Gaussian
            ce = F.mse_loss(X_hat, X)
            kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            return {"loss": ce+kld, "CE": ce, "KLD": kld}
        elif self.model_name == "ae":
            X_hat, X = model_output
            recon_loss = F.mse_loss(X_hat, X)
            return {"loss": recon_loss}
        else:
            raise ValueError("unknown model name")

    def feed_to_model(self, model_input):
        X, y, attack_name = model_input
        return self.model(X)

    def to_device(self, model_input):
        X, y, attack_name = model_input
        X = X.to(self.device)
        y = y.to(self.device)
        return X, y, attack_name

    def get_y(self, model_input):
        X, y, attack_name = model_input
        return y

    def compute_recon_error(self, model_output):
        if self.model_name == "vae":
            X_hat, X, mu, log_var = model_output
        elif self.model_name == "ae":
            X_hat, X = model_output
        else:
            raise ValueError("unknown model name")
        return torch.sum((X_hat - X) ** 2, dim=1)

    def log_losses(self, losses, train=True):
        log_dict = self.train_losses if train else self.test_losses
        if self.model_name == "vae":
            loss, ce, kld = losses
            log_dict["loss"].append(loss)
            log_dict["CE"].append(ce)
            log_dict["KLD"].append(kld)
        elif self.model_name == "ae":
            loss = losses
            log_dict["loss"].append(loss)
        else:
            raise ValueError("unknown model name")

    def get_recon_errors_and_labels(self, validate=True):
        self.model.eval()
        loader = self.val_loader if validate else self.test_loader
        recon_errors = []
        labels = []
        with torch.no_grad():
            for model_input in tqdm(loader):
                model_input = self.to_device(model_input)
                model_output = self.feed_to_model(model_input)
                batch_errors = self.compute_recon_error(model_output)
                recon_errors.append(batch_errors)
                labels.append(self.get_y(model_input))

        recon_errors = torch.cat(recon_errors, dim=0).view(-1)
        labels = torch.cat(labels, dim=0).view(-1)
        return recon_errors, labels

    
    def compute_best_f1(self,epoch,recon_errors, labels, return_threshold=True):
        recon_errors = recon_errors.cpu().numpy()
        labels = labels.cpu().numpy()
        precision, recall, threshold = precision_recall_curve(labels, recon_errors)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        thresholds = threshold.tolist()
        thresholds.append(max(threshold)+0.1)
        threshold = np.array(thresholds)

        max_f1 = np.max(f1)
        best_threshold = threshold[np.argmax(f1)]
        if epoch == self.epoch_num:
            plt.figure(figsize=(10,6))
            plt.plot(threshold, precision, label='Precision')
            plt.plot(threshold, recall, label='Recall')
            plt.plot(threshold, f1, label='F1 Score')
            plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold ({best_threshold:.4f})')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            if self.model_name == "vae":
                plt.title('Precision, Recall, F1 Score vs Threshold for VAE')
            else:
                plt.title('Precision, Recall, F1 Score vs Threshold for AE')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(self.plot_dir + f"{self.model_name}_threshold.jpg",format='jpg')

        if return_threshold:
            return max_f1, best_threshold
        else:
            return max_f1
        
    def compute_all_metrics(self, recon_errors, labels, threshold):
        recon_errors = recon_errors.cpu().numpy()
        labels = labels.cpu().numpy()
        y_pred = np.zeros_like(labels)
        y_pred[recon_errors > threshold] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=y_pred, average="binary")
        accuracy = accuracy_score(labels, y_pred)

        fig, ax = plt.subplots()
        labels = ['Precision', 'Recall', 'F1 Score','Accuracy']
        colors = ['blue', 'green', 'orange', 'red']
        values = [precision, recall, f1, accuracy]
        x_pos = range(len(labels))
        ax.bar(x_pos, values, align='center', alpha=0.5,color = colors)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)

        ax.set_ylabel('Scores')
        if self.model_name == "vae":
            ax.set_title('Precision, Recall, F1 Score and Accuracy of VAE')
        else:
            ax.set_title('Precision, Recall, F1 Score and Accuracy of AE')
        plt.tight_layout()
        plt.savefig(self.plot_dir + f"{self.model_name}_metrics.jpg",format='jpg')

        print("Best result in this experiment: ")
        print(f"precision: {precision:.4f}\t recall: {recall:.4f}\t F1: {f1:.4f}\t accuracy: {accuracy:.4f}")

    def visualize_losses(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x_pos = np.arange(self.epoch_num)

        # plot losses
        axes[0].plot(x_pos, self.train_losses["loss"], label="total_loss")
        axes[1].plot(x_pos, self.test_losses["loss"], label="total_loss")
        if self.model_name == "vae":
            axes[0].plot(x_pos, self.train_losses["CE"], label="CE")
            axes[0].plot(x_pos, self.train_losses["KLD"], label="KLD")
            axes[1].plot(x_pos, self.test_losses["CE"], label="CE")
            axes[1].plot(x_pos, self.test_losses["KLD"], label="KLD")
            axes[0].set_title("train loss of VAE")
            axes[1].set_title("test loss of VAE")
        else:
            axes[0].set_title("train loss of AE")
            axes[1].set_title("test loss of AE")
        axes[0].set_xlabel("epoch num")
        axes[1].set_xlabel("epoch num")
        axes[0].legend()
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.plot_dir + f"{self.model_name}_loss.jpg",format='jpg')

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def start_experiment(self):
        single_exp_timer = Timer("Single Exp Timer")
        best_f1 = 0
        best_threshold = 0
        for epoch in range(1, self.epoch_num + 1):
            losses = self.train(epoch)
            self.log_losses(losses, train=True)
            losses = self.test(validate=True)
            self.log_losses(losses, train=False)
            recon_errors, labels = self.get_recon_errors_and_labels(validate=True)
            epoch_f1, threshold = self.compute_best_f1(epoch,recon_errors, labels, return_threshold=True)
            print(f"F1 score: {epoch_f1:.4f}\t at threshold: {threshold:.4f}")
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_threshold = threshold
                # save model
                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), f"{self.save_dir}{self.model_name}.pth")
                else:
                    torch.save(self.model.state_dict(), f"{self.save_dir}{self.model_name}.pth")

        # load the best model
        self.model.load_state_dict(torch.load(f"{self.save_dir}{self.model_name}.pth"))
        recon_errors, labels = self.get_recon_errors_and_labels(validate=False)
        self.compute_all_metrics(recon_errors, labels, best_threshold)
        self.visualize_losses()
        single_exp_timer.end()