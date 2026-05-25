import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=1,
        num_classes=4,
        dropout=0.3
    ):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        features = self.feature_extractor(x)
        features = features.flatten(1)

        features = features.view(B, T, 512)

        lstm_out, _ = self.lstm(features)

        final_feature = lstm_out[:, -1, :]

        logits = self.classifier(final_feature)

        return logits

model = CNNLSTM()

dummy = torch.randn(4, 16, 3, 224, 224)

out = model(dummy)

print(out.shape)