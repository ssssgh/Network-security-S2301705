import torch
from torch import nn
from typing import List, Tuple

class VAE(nn.Module):
    def __init__(
        self,
        in_dim: int = 200,
        latent_dim: int = 20,
        hidden_dims: List = None
    ) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [512, 256, 64, 32]

        # Build Encoder
        for idx, h_dim in enumerate(hidden_dims):
            last_dim = in_dim if idx == 0 else hidden_dims[idx - 1]
            modules.append(
                nn.Sequential(
                    nn.Linear(last_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim)
                )
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        hidden_dims.reverse()
        for idx, h_dim in enumerate(hidden_dims):
            last_dim = latent_dim if idx == 0 else hidden_dims[idx - 1]
            modules.append(
                nn.Sequential(
                    nn.Linear(last_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim)
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], in_dim),
            nn.Sigmoid()
        )

    def encode(self, X: torch.Tensor) -> Tuple:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        encoded = self.encoder(X)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes onto the original feature space
        """
        decoded = self.decoder(z)
        x_hat = self.final_layer(decoded)
        return x_hat

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, X: torch.Tensor) -> Tuple:
        mu, log_var = self.encode(X)
        z = self.reparameterize(mu, log_var)
        X_hat = self.decode(z)
        return X_hat, X, mu, log_var
        # here to change the return
        #return torch.sum((X - X_hat) ** 2, dim=1)



class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 200, hidden_dims: List = None) -> None:
        super(AutoEncoder, self).__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [512, 256, 64, 32]

        # Build Encoder
        for idx, h_dim in enumerate(hidden_dims):
            last_dim = in_dim if idx == 0 else hidden_dims[idx - 1]
            modules.append(
                nn.Sequential(
                    nn.Linear(last_dim, h_dim),
                    nn.ReLU()
                )
            )

        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        hidden_dims.reverse()
        for idx, h_dim in enumerate(hidden_dims):
            out_dim = hidden_dims[idx + 1]
            modules.append(
                nn.Sequential(
                    nn.Linear(h_dim, out_dim),
                    nn.ReLU()
                )
            )
            if idx == len(hidden_dims) - 2:
                break
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], in_dim),
            nn.Sigmoid()
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        X_hat = self.final_layer(decoded)
        # here to change the return
        #return torch.sum((X - X_hat) ** 2, dim=1)
        return X_hat, X

if __name__ == "__main__":
    X = torch.randn(16, 200)
    model = VAE()
    X_hat, _ = model(X)
    print(X_hat.size())


