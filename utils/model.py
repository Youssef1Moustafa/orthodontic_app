import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

class TabularEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32,64),
            nn.ReLU()
        )

    def forward(self,x):
        return self.model(x)

class OrthodonticModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder()

        self.class_embedding = nn.Embedding(2,8)
        self.treatment_embedding = nn.Embedding(2,8)

        self.shared = nn.Sequential(
            nn.Linear(512+64+16,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )

        self.angle_head = nn.Linear(128,2)
        self.landmark_head = nn.Linear(128,136)

    def forward(self,image,tabular,class_id,treatment_id):

        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_encoder(tabular)

        class_emb = self.class_embedding(class_id)
        treat_emb = self.treatment_embedding(treatment_id)

        fused = torch.cat([img_feat,tab_feat,class_emb,treat_emb],dim=1)

        shared = self.shared(fused)

        angle = self.angle_head(shared)
        landmark = self.landmark_head(shared)

        return angle, landmark