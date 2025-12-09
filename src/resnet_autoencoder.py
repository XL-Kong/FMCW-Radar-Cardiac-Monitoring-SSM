from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, encoded_space_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, encoded_space_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_cnn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, encoded_space_dim: int = 512, output_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(encoded_space_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 130, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(130),
            nn.ReLU(True),
            nn.ConvTranspose2d(130, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        return x


class ResNetAutoencoder(nn.Module):
    def __init__(self, encoded_space_dim: int = 512, output_channels: int = 1, dropout: float = 0.4):
        super().__init__()
        self.encoder = ResNetEncoder(encoded_space_dim, dropout)
        self.decoder = ResNetDecoder(encoded_space_dim, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiChannelResNetAutoencoder(nn.Module):
    """
    Adapter + SE block stacked in front of a frozen pre-trained autoencoder.
    """

    def __init__(
        self,
        input_channels_n: int,
        pretrained_path: str,
        encoded_space_dim: int = 512,
        output_channels: int = 1,
        adapter_channels: int = 16,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels=input_channels_n, out_channels=adapter_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(adapter_channels),
            nn.ReLU(inplace=True),
            SEBlock(channel=adapter_channels, reduction=se_reduction),
            nn.Conv2d(in_channels=adapter_channels, out_channels=1, kernel_size=1, padding=0),
        )

        pretrained = ResNetAutoencoder(encoded_space_dim, output_channels)
        try:
            pretrained.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unable to load pre-trained weights: {exc}") from exc

        self.pretrained_encoder = pretrained.encoder
        self.pretrained_decoder = pretrained.decoder
        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False
        for param in self.pretrained_decoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(x)
        encoded = self.pretrained_encoder(adapted)
        decoded = self.pretrained_decoder(encoded)
        return decoded

