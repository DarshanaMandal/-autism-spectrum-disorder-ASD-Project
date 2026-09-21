"""
MultiSense GAN: Conditional Recurrent Pose Sequence Generator
Synthesizes temporal skeletal trajectories (B, 60, 132) conditioned on behavior class.
"""

import torch
import torch.nn as nn


class ConditionalPoseGenerator(nn.Module):
    def __init__(self, latent_dim=64, num_classes=3, feature_dim=132, seq_len=60, hidden_dim=128, num_layers=2):
        super(ConditionalPoseGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Project concatenated [noise, class_onehot]
        self.input_fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Recurrent temporal generator
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output projection back to skeletal landmark feature space (132)
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, z, c_onehot):
        """
        Args:
            z: (B, latent_dim) Gaussian noise
            c_onehot: (B, num_classes) one-hot conditioning
        Returns:
            fake_seq: (B, seq_len, feature_dim)
        """
        B = z.size(0)
        zc = torch.cat([z, c_onehot], dim=1)  # (B, latent_dim + num_classes)
        h = self.input_fc(zc)  # (B, hidden_dim)
        
        # Expand across sequence time dimension
        h_expanded = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, hidden_dim)

        rnn_out, _ = self.rnn(h_expanded)  # (B, seq_len, hidden_dim * 2)
        fake_seq = self.output_fc(rnn_out)  # (B, seq_len, feature_dim)
        return fake_seq

    @torch.no_grad()
    def sample(self, num_samples, class_id, device="cpu"):
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        c = torch.zeros(num_samples, self.num_classes, device=device)
        c[:, class_id] = 1.0
        fake_seq = self(z, c)
        return fake_seq.cpu().numpy()
