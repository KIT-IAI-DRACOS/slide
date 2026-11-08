import torch
import torch.nn as nn


class StaticFeatureEmbedding(nn.Module):
    def __init__(self, cardinalities, embed_dims) -> None:
        super().__init__()

        self.cardinalities = cardinalities
        self.embed_dims = embed_dims
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embed_dim) for cardinality, embed_dim in zip(cardinalities, embed_dims, strict=True)]
        )

    def forward(self, x):
        # x: B x num_static_ex
        embedded_features = [self.embeddings[i](x[:, i]) for i in range(len(self.cardinalities))]
        # Concatenate along the feature dimension
        embedded = torch.cat(embedded_features, dim=-1)  # B x (num_static_ex * embed_dim)
        return embedded
