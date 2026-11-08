import numpy as np
import torch


class IdentityScaler:
    def fit(self, data):
        return self
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

class StandardScaler:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, data) -> None:
        if torch.is_tensor(data):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)

        self.mean = np.array(arr.mean(axis=0))
        self.std = np.array(arr.std(axis=0))

        self.std = np.where(self.std == 0, 1, self.std)  # Avoid division by zero

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler: you must call .fit(...) before .transform(...)")
        if torch.is_tensor(data):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std  = torch.as_tensor(self.std,  dtype=data.dtype, device=data.device)
            return (data - mean) / std
        
        else:
            arr = np.asarray(data)
            return (arr - self.mean) / self.std

    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler: you must call .fit(...) before .inverse_transform(...)")

        if torch.is_tensor(data):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std  = torch.as_tensor(self.std,  dtype=data.dtype, device=data.device)

            return data * std + mean

        else:
            arr = np.asarray(data)
            return arr * self.std + self.mean
