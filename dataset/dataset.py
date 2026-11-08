import logging
import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset

from dataset.ex_noise import add_horizon_ex_noise
from dataset.scaler import IdentityScaler, StandardScaler

logger = logging.getLogger(__name__)


SCALER = {
    "standard": StandardScaler,
    "none":     IdentityScaler,
}
class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.batch_size = config.model.batch_size
        self.num_workers = getattr(config, 'num_workers', 1)
        self.pin_memory = getattr(config, 'pin_memory', False)

        self.target_scaler = SCALER[config.scalers.target]()
        self.feature_scaler = SCALER[config.scalers.feature]()

        self.data = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
 
    def prepare_data(self) -> None:
        data_path = os.path.join(
            self.config.dataset.data_directory,
            self.config.dataset.data_name,
        )
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)

        if len(self.data.columns) > 0:
            first = self.data.columns[0]
            first_name = "" if first is None else str(first).strip()
            if first_name == "" or first_name.startswith("Unnamed"):
                self.data = self.data.rename(columns={first: "Date"})

        # use 'date' or 'Date' as index
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
        elif 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)

        if self.config.dataset.start is not None:
            if type(self.config.dataset.start) is int:
                self.data = self.data.iloc[self.config.dataset.start:]
            elif type(self.config.dataset.start) is str:
                self.data = self.data.loc[pd.to_datetime(self.config.dataset.start):]
            else:
                raise ValueError("config.dataset.start should be int or str")

        if self.config.dataset.end is not None:
            if type(self.config.dataset.end) is int:
                self.data = self.data.iloc[:self.config.dataset.end]
            elif type(self.config.dataset.end) is str:
                self.data = self.data.loc[:pd.to_datetime(self.config.dataset.end)]
            else:
                raise ValueError("config.dataset.end should be int or str")


    def setup(self, stage=None) -> None:
        if self.data is None:
            self.prepare_data()

        if type(self.config.dataset.val_split) is float:
            val_split = int(len(self.data) * self.config.dataset.val_split)
        elif type(self.config.dataset.val_split) is int:
            val_split = self.config.dataset.val_split
        elif type(self.config.dataset.val_split) is str:
            val_split_time = pd.to_datetime(self.config.dataset.val_split)
            val_split = self.data.index.get_indexer([val_split_time], method="ffill")[0]
        else:
            raise ValueError("config.dataset.val_split should be float, int or str")

        if type(self.config.dataset.test_split) is float:
            test_split = int(len(self.data) * self.config.dataset.test_split)
        elif type(self.config.dataset.test_split) is int:
            test_split = self.config.dataset.test_split
        elif type(self.config.dataset.test_split) is str:
            test_split_time = pd.to_datetime(self.config.dataset.test_split)
            test_split = self.data.index.get_indexer([test_split_time], method="ffill")[0]
        else:
            raise ValueError("config.dataset.test_split should be float, int or str")


        if stage in (None, 'fit'):         
            train_df = self.data.iloc[:val_split]
            overlap_val = self.config.dataset.get('overlap_val', True)
            if overlap_val:
                val_df = self.data.iloc[val_split - self.config.sample.input_length:test_split]
            else:
                val_df = self.data.iloc[val_split:test_split]

            train_df, train_feature_df = self._split_features(train_df)
            val_df,   val_feature_df   = self._split_features(val_df)
            
            # target scaling
            self.target_scaler.fit(train_df)
            if train_feature_df is not None:
                self.feature_scaler.fit(train_feature_df)
            train_df = pd.DataFrame(
                self.target_scaler.transform(train_df),
                index=train_df.index,
                columns=train_df.columns,
            )
            val_df   = pd.DataFrame(
                self.target_scaler.transform(val_df),
                index=val_df.index,
                columns=val_df.columns,
            )

            if train_feature_df is not None:
                train_feature_df = pd.DataFrame(
                    self.feature_scaler.transform(train_feature_df),
                    index=train_feature_df.index,
                    columns=train_feature_df.columns,
                )
                val_feature_df = pd.DataFrame(
                    self.feature_scaler.transform(val_feature_df),
                    index=val_feature_df.index,
                    columns=val_feature_df.columns,
                )

            self.train_dataset = self._create_dataset(train_df, future=train_feature_df, stride=self.config.sample.get('train_step_size', None))
            self.val_dataset = self._create_dataset(val_df, future=val_feature_df)


        if stage in (None, 'test', 'predict'):
            overlap_test = self.config.dataset.get('overlap_test', True)
            if overlap_test:
                test_df = self.data.iloc[test_split - self.config.sample.input_length:]
            else:
                test_df = self.data.iloc[test_split:]

            test_df, test_feature_df = self._split_features(test_df)
            test_df = pd.DataFrame(
                self.target_scaler.transform(test_df),
                index=test_df.index,
                columns=test_df.columns,
            )
            if test_feature_df is not None:
                test_feature_df = pd.DataFrame(
                    self.feature_scaler.transform(test_feature_df),
                    index=test_feature_df.index,
                    columns=test_feature_df.columns,
                )
            self.test_dataset = self._create_dataset(
                test_df, future=test_feature_df
            )


    def _create_dataset(
        self,
        data: pd.DataFrame,
        future: pd.DataFrame,
        stride = None
    ):
        return TimeSeriesDataset(
            df=data,
            history_length=self.config.sample.input_length,
            forecast_length=self.config.sample.target_length,
            stride=self.config.sample.step_size if stride is None else stride,
            future_features=future,
            horizon_ex_noise=bool(self.config.dataset.get("horizon_ex_noise", False)),
            horizon_ex_noise_scale=float(self.config.dataset.get("horizon_ex_noise_scale", 0.5)),
        )

    def _split_features(self, df: pd.DataFrame):
        target_cols = self._target_cols()

        if target_cols is None:
            target_df = df
            feature_df = None
        else:
            missing = [c for c in target_cols if c not in df.columns]
            if missing:
                raise KeyError(f"Target column(s) not found in data: {missing}")

            target_df = df[target_cols]

            feature_cols = [c for c in df.columns if c not in target_cols]
            feature_df = df[feature_cols] if len(feature_cols) > 0 else None

        return target_df, feature_df


    def _target_cols(self):
        tgt = self.config.dataset.target
        if tgt is None:
            return None

        if isinstance(tgt, str):
            return [tgt]

        if isinstance(tgt, (list, tuple, ListConfig)):
            cols = list(tgt)
            if not all(isinstance(c, str) for c in cols):
                raise ValueError(f"All targets must be strings, got: {cols}")
            return cols

        raise ValueError(
            f"config.dataset.target must be None, a string, or a list of strings; got {type(tgt)}"
    )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self.test_dataloader()


    def _dataloader(self, dataset, shuffle: bool=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df,
        history_length: int,     
        forecast_length: int, 
        stride: int,
        future_features=None,
        horizon_ex_noise: bool = False,
        horizon_ex_noise_scale: float = 0.5,
    ) -> None:
        self.data_np       = df.to_numpy()
        self.feature_data_np = future_features.to_numpy() if future_features is not None else None
        self.horizon_ex_noise = horizon_ex_noise
        self.horizon_ex_noise_scale = horizon_ex_noise_scale

        self.timestamps    = df.index      # shape (T,)
        self.x_size = history_length
        self.y_size   = forecast_length

        self.window_size   = self.x_size + self.y_size
        self.stride         = stride

        # Figure out all window starts
        L = len(self.data_np)
        W = self.window_size
        if L < W:
            raise ValueError(f"Data length {L} < window size {W}")
        total_windows = (L - W) // self.stride + 1
        all_starts = np.arange(0, total_windows * self.stride, self.stride)

        logger.info(f"Total windows: {len(all_starts)}")
 
        self.starts = all_starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the given index,
        extracting a history window, a forecast window, plus any exogenous
        and static features.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple (inp, out)
            inp : dict of Tensors
                Contains:
                    past_target     — Tensor of shape (x_size, num_targets)
                    past_features   — Tensor of shape (feature_lookback, num_past_features),
                                      or empty Tensor if no past features
                    future_features — Tensor of shape (feature_lookback + feature_lookahead,
                                      num_future_features), or empty Tensor if no future features
                    static          — Tensor of shape (num_static_features),
                                      or empty Tensor if no static features

            out : Tensor
                Tensor of shape (y_size, num_targets) containing the
                future target values to predict.

        Examples
        --------
        # Per-sample
        inp = {
            "past_target":     Tensor[x_size, num_targets],
            "past_features":   Tensor[feature_lookback, num_past_features],
            "future_features": Tensor[feature_lookback+feature_lookahead, num_future_features],
            "static":          Tensor[num_static_features]
        }
        out = Tensor[y_size, num_targets]

        # Batched (DataLoader)
        inp = {
            "past_target":     Tensor[batch_size, x_size, num_targets],
            "past_features":   Tensor[batch_size, feature_lookback, num_past_features],
            "future_features": Tensor[batch_size, feature_lookback+feature_lookahead, num_future_features],
            "static":          Tensor[batch_size, num_static_features]
        }
        out = Tensor[batch_size, y_size, num_targets]
        """
        
        # Compute slice bounds
        s = self.starts[idx]
        e = s + self.window_size

        # Core input/target
        x_np = self.data_np[s : s + self.x_size]
        y_np = self.data_np[s + self.x_size : e]
        inp  = {"past_target": torch.from_numpy(x_np).float()}

        if self.feature_data_np is not None:
            xf_np = self.feature_data_np[s : e]
            if self.horizon_ex_noise:
                xf_np = add_horizon_ex_noise(
                    xf_np, self.x_size, self.horizon_ex_noise_scale
                )
            inp["future_features"] = torch.from_numpy(xf_np).float()

        inp["static"] = torch.empty(0)
        inp['past_features'] = torch.empty(0)

        # final target tensor
        out = torch.from_numpy(y_np).float()
        return inp, out


