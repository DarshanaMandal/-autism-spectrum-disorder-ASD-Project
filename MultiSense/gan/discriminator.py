"""
MultiSense GAN: Conditional Recurrent Pose Sequence Discriminator
Evaluates temporal authenticity of pose sequences conditioned on class.
"""

import torch
import torch.nn as nn


class ConditionalPoseDiscriminator(nn.Module):
    def __init__(self, num_classes=3, feature_dim=132, seq_len=60, hidden_dim=128, num_layers=2):
        super(ConditionalPoseDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.seq_len = seq_len

        # Recurrent feature extractor on conditioned sequence
        self.rnn = nn.GRU(
            input_size=feature_dim + num_classes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Discriminator classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, c_onehot):
        """
        Args:
            x: (B, seq_len, feature_dim) pose sequence
            c_onehot: (B, num_classes) one-hot conditioning
        Returns:
            validity: (B, 1) scalar realism score
        """
        B, T, _ = x.shape
        # Broadcast conditioning across all time steps
        c_expanded = c_onehot.unsqueeze(1).repeat(1, T, 1)  # (B, T, num_classes)
        xc = torch.cat([x, c_expanded], dim=2)  # (B, T, feature_dim + num_classes)

        rnn_out, _ = self.rnn(xc)  # (B, T, hidden_dim * 2)
        # Global temporal average pooling + max pooling for robust action recognition
        avg_pool = torch.mean(rnn_out, dim=1)  # (B, hidden_dim * 2)
        validity = self.fc(avg_pool)  # (B, 1)
        return validity
