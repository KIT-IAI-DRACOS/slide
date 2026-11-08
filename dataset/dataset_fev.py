import logging

import datasets
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from datasets import Array2D, Features, Sequence, Value
from torch.utils.data import DataLoader, Dataset

from dataset.scaler import IdentityScaler, StandardScaler

logger = logging.getLogger(__name__)


SCALER = {
    "standard": StandardScaler,
    "none":     IdentityScaler,
}

class FevDataModule(pl.LightningDataModule):
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
        logger.info(f"Loading data from {self.config.dataset.path}")
        
        self.data = datasets.load_dataset(
            self.config.dataset.path,
            self.config.dataset.data_name,
            split="train"
        )
        string_to_int_cols = self.config.dataset.get("string_to_int_cols", None)
        if string_to_int_cols is not None:
            self.data = self._dynamic_string_to_int(
                self.data,
                string_to_int_cols,
            )
        log_cols = self.config.dataset.get("log_columns", None)
        if log_cols is not None:
            self.data = self._apply_log_transform(
                self.data,
                log_cols,
            )

        if self.config.dataset.add_time_features:
            self.data = self._add_cyclical_time_features(self.data)

        # Cast string static columns to integer ClassLabel codes once so that
        # downstream numpy/tensor formatting works without string-to-float
        # conversion issues and the vocabulary stays consistent across splits.
        if len(getattr(self.config.dataset, "static_columns", [])) > 0:
            self.data = self._static_string_to_int(
                self.data,
                self.config.dataset.static_columns,
            )

        if self.config.dataset.start is not None:
            raise NotImplementedError("Slicing by start date/index is not implementedin fev datamodule")

        if self.config.dataset.end is not None:
            raise NotImplementedError("Slicing by end date/index is not implemented in fev datamodule")

    def setup(self, stage: str | None = None) -> None:
        if self.data is None:
            self.prepare_data()

        need_fit = stage in (None, 'fit')
        need_test = stage in (None, 'test', 'predict')

        # Slice once for all splits to avoid repeated slow HuggingFace .map passes.
        train_ds, val_ds, test_ds = self._slice_all_splits(
            need_train=need_fit,
            need_val=need_fit,
            need_test=need_test,
        )

        if stage in (None, 'fit'):
            # target scaling
            if self.config.scalers.scaling_mode == "global":
                self.target_scaler = self._fit_global_scaler(
                    train_ds,
                    [self.config.dataset.target],
                    scaler_key=self.config.scalers.target,
                )
            elif self.config.scalers.scaling_mode == "per_id":
                self.target_scaler = self._fit_per_item_scalers(
                    train_ds,
                    cols=[self.config.dataset.target],
                    id_col=self.config.dataset.id_column,
                    scaler_key=self.config.scalers.target,
                )
            else:
                raise ValueError(f"Unknown scaling mode: {self.config.scalers.scaling_mode}")
            
            # apply scaling
            train_ds = self._apply_target_scaling(train_ds)
            val_ds = self._apply_target_scaling(val_ds)

            # feature scaling
            if len(self.config.dataset.past_columns) + len(self.config.dataset.future_columns) > 0:
                if self.config.dataset.get("scale_columns", None) is not None:
                    feature_cols = self.config.dataset.scale_columns
                else:
                    feature_cols = self.config.dataset.past_columns + self.config.dataset.future_columns

                feature_mode = getattr(self.config.scalers, "feature_scaling_mode", "global")

                if feature_mode == "global":
                    self.feature_scaler = self._fit_global_scaler(
                        train_ds,
                        feature_cols,
                        scaler_key=self.config.scalers.feature,
                    )
                    print("feature scaler:", self.feature_scaler.mean, self.feature_scaler.std)
                elif feature_mode == "per_id":
                    self.feature_scaler = self._fit_per_item_scalers(
                        train_ds,
                        cols=feature_cols,
                        id_col=self.config.dataset.id_column,
                        scaler_key=self.config.scalers.feature,
                    )
                elif feature_mode == "none":
                    self.feature_scaler = None
                else:
                    raise ValueError(f"Unknown feature scaling mode: {feature_mode}")

                train_ds = self._scale_feature_rows(train_ds)
                val_ds = self._scale_feature_rows(val_ds)

            self.train_dataset = self._create_dataset(train_ds, train_step_size=True)
            self.val_dataset = self._create_dataset(val_ds)


        if stage in (None, 'test', 'predict'):
            # apply scaling
            if self.config.scalers.scaling_mode == "global":
                test_ds = self._apply_target_scaling(test_ds)
            elif self.config.scalers.scaling_mode == "per_id":
                test_ds = self._apply_target_scaling(test_ds)
            
            # feature scaling
            if len(self.config.dataset.past_columns) + len(self.config.dataset.future_columns) > 0:
                test_ds = self._scale_feature_rows(test_ds)
            
            if test_ds is not None:
                self.test_dataset = self._create_dataset(test_ds)

    def _create_dataset(
        self,
        data,
        train_step_size=False
    ):
        data = self._precompute_windows(data, stride=self.config.sample.step_size if not train_step_size else self.config.sample.get('train_step_size', self.config.sample.step_size))
        return FevDataset(
            dataset=data,
            history_length=self.config.sample.input_length,
            forecast_length=self.config.sample.target_length,
            stride=self.config.sample.step_size if not train_step_size else self.config.sample.get('train_step_size', self.config.sample.step_size),
            target_col=self.config.dataset.target,
            id_col=self.config.dataset.id_column,
            past_cols=self.config.dataset.past_columns,
            future_cols=self.config.dataset.future_columns,
            static_cols=self.config.dataset.static_columns,
        )

    def _precompute_windows(self, ds, stride):
        """
        Materialize per-series metadata (starts, stacked feature matrices, etc.)
        in parallel via Dataset.map to avoid Python-level looping in FevDataset.
        """
        if ds is None:
            return None

        past_cols = self.config.dataset.past_columns
        future_cols = self.config.dataset.future_columns
        static_cols = self.config.dataset.static_columns
        target_col = self.config.dataset.target
        id_col = self.config.dataset.id_column
        window_size = self.config.sample.input_length + self.config.sample.target_length

        def process_batch(batch):
            out = {
                "timestamp": [],
                id_col: [],
                "starts": [],
                "target_arr": [],
                "past_mat": [],
                "future_mat": [],
                "static_vec": [],
                "valid": [],
            }

            bsz = len(batch["timestamp"])
            for i in range(bsz):
                ts = np.asarray(batch["timestamp"][i])
                L = len(ts)
                max_start = L - window_size
                is_valid = max_start >= 0
                out["valid"].append(is_valid)
                out["timestamp"].append(ts)
                out[id_col].append(str(batch[id_col][i]))

                if not is_valid:
                    out["starts"].append(np.empty((0,), dtype=np.int64))
                    out["target_arr"].append(np.empty((0,), dtype=np.float32))
                    out["past_mat"].append(None)
                    out["future_mat"].append(None)
                    out["static_vec"].append(np.empty((0,), dtype=np.float32))
                    continue

                out["starts"].append(np.arange(0, max_start + 1, stride, dtype=np.int64))
                out["target_arr"].append(np.asarray(batch[target_col][i], dtype=np.float32))

                if past_cols:
                    past_list = [
                        np.asarray(batch[col][i], dtype=np.float32) for col in past_cols
                    ]
                    out["past_mat"].append(np.stack(past_list, axis=-1))
                else:
                    out["past_mat"].append(None)

                if future_cols:
                    fut_list = [
                        np.asarray(batch[col][i], dtype=np.float32) for col in future_cols
                    ]
                    out["future_mat"].append(np.stack(fut_list, axis=-1))
                else:
                    out["future_mat"].append(None)

                if static_cols:
                    static_vals = []
                    for col in static_cols:
                        v_arr = np.asarray(batch[col][i], dtype=np.float32)
                        if v_arr.ndim == 0:
                            static_vals.append(float(v_arr))
                        elif v_arr.ndim == 1 and v_arr.shape[0] == 1:
                            static_vals.append(float(v_arr[0]))
                        else:
                            raise ValueError(
                                f"Static column {col} should be scalar or length-1, got shape {v_arr.shape}"
                            )
                    out["static_vec"].append(np.asarray(static_vals, dtype=np.float32))
                else:
                    out["static_vec"].append(np.empty((0,), dtype=np.float32))

            return out

        # Declare feature types so map stores Arrow arrays (faster to format to numpy).
        features = ds.features.copy()
        features.update(
            {
                id_col: Value("string"),
                "starts": Sequence(Value("int64")),
                "target_arr": Sequence(Value("float32")),
                "past_mat": Array2D(
                    shape=(None, len(past_cols)),
                    dtype="float32",
                ) if past_cols else Value("null"),
                "future_mat": Array2D(
                    shape=(None, len(future_cols)),
                    dtype="float32",
                ) if future_cols else Value("null"),
                "static_vec": Sequence(Value("float32")),
                "valid": Value("bool"),
            }
        )

        mapped = ds.map(
            process_batch,
            batched=True,
            num_proc=1,
            desc="Precompute windows",
            features=Features(features),
            load_from_cache_file=False,
        )

        filtered = mapped.filter(
            lambda valid: valid,
            input_columns="valid",
            desc="Filter short series",
        )

        return filtered.remove_columns(["valid"])

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


    def _fit_global_scaler(self, train_ds, cols, scaler_key):
        """
        Fit a global scaler using streaming stats via Dataset.map (no batching).
        """
        scaler = SCALER[scaler_key]()
        if isinstance(scaler, IdentityScaler):
            return scaler

        if scaler_key == "standard":
            def row_stats(row):
                mat = np.stack([np.asarray(row[c]) for c in cols], axis=-1)
                if mat.ndim == 1:
                    mat = mat[:, None]
                return {
                    "sum": mat.sum(axis=0),
                    "sumsq": (mat * mat).sum(axis=0),
                    "count": mat.shape[0],
                }

            stats_ds = train_ds.map(
                row_stats,
                batched=False,
                num_proc=self.num_workers,
                remove_columns=train_ds.column_names,
                desc="Fit scaler (global)",
                load_from_cache_file=False,
            )

            total_sum = np.sum(np.stack(stats_ds["sum"], axis=0), axis=0)
            total_sumsq = np.sum(np.stack(stats_ds["sumsq"], axis=0), axis=0)
            total_count = int(np.sum(stats_ds["count"]))

            mean = total_sum / total_count
            var = total_sumsq / total_count - mean**2
            std = np.sqrt(np.maximum(var, 0.0))
            std = np.where(std == 0, 1, std)

            scaler.mean = mean
            scaler.std = std

        elif scaler_key == "minmax":
            def row_stats(row):
                mat = np.stack([np.asarray(row[c]) for c in cols], axis=-1)
                if mat.ndim == 1:
                    mat = mat[:, None]
                return {
                    "min": mat.min(axis=0),
                    "max": mat.max(axis=0),
                }

            stats_ds = train_ds.map(
                row_stats,
                batched=False,
                num_proc=1,
                remove_columns=train_ds.column_names,
                desc="Fit scaler (global)",
                load_from_cache_file=False,
            )

            global_min = np.min(np.stack(stats_ds["min"], axis=0), axis=0)
            global_max = np.max(np.stack(stats_ds["max"], axis=0), axis=0)
            data_range = global_max - global_min
            data_range[data_range == 0.0] = 1.0

            scaler.min = global_min
            scaler.max = global_max
            scaler.data_range_ = data_range

        else:
            raise ValueError(f"Unsupported scaler type: {scaler_key}")

        return scaler

    def _fit_per_item_scalers(self, train_ds, cols, id_col, scaler_key):
        """
        Fit a scaler per series using streaming stats via Dataset.map (no batching).
        """
        cols = [cols] if isinstance(cols, str) else list(cols)
        scaler_cls = SCALER[scaler_key]
        if scaler_cls is IdentityScaler:
            ids = set()
            for row in train_ds:
                ids.add(str(row[id_col]))
            return {i: scaler_cls() for i in ids}

        if scaler_key == "standard":
            def row_stats(row):
                arr = np.stack([np.asarray(row[c]) for c in cols], axis=-1)
                if arr.ndim == 1:
                    arr = arr[:, None]
                return {
                    "id": str(row[id_col]),
                    "sum": arr.sum(axis=0),
                    "sumsq": (arr * arr).sum(axis=0),
                    "count": arr.shape[0],
                }
        elif scaler_key == "minmax":
            def row_stats(row):
                arr = np.stack([np.asarray(row[c]) for c in cols], axis=-1)
                if arr.ndim == 1:
                    arr = arr[:, None]
                return {
                    "id": str(row[id_col]),
                    "min": arr.min(axis=0),
                    "max": arr.max(axis=0),
                }
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_key}")

        stats_ds = train_ds.map(
            row_stats,
            batched=False,
            num_proc=1,
            remove_columns=train_ds.column_names,
            desc="Fit scaler (per id)",
            load_from_cache_file=False,
        )

        by_item = {}
        for i in range(len(stats_ds["id"])):
            item_id = stats_ds["id"][i]
            if item_id not in by_item:
                if scaler_key == "standard":
                    by_item[item_id] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
                else:
                    by_item[item_id] = {"min": None, "max": None}

            if scaler_key == "standard":
                by_item[item_id]["sum"] += np.asarray(stats_ds["sum"][i], dtype=np.float64)
                by_item[item_id]["sumsq"] += np.asarray(stats_ds["sumsq"][i], dtype=np.float64)
                by_item[item_id]["count"] += int(stats_ds["count"][i])
            else:
                cur_min = np.asarray(stats_ds["min"][i], dtype=np.float64)
                cur_max = np.asarray(stats_ds["max"][i], dtype=np.float64)
                if by_item[item_id]["min"] is None:
                    by_item[item_id]["min"] = cur_min
                    by_item[item_id]["max"] = cur_max
                else:
                    by_item[item_id]["min"] = np.minimum(by_item[item_id]["min"], cur_min)
                    by_item[item_id]["max"] = np.maximum(by_item[item_id]["max"], cur_max)

        scalers = {}
        for item_id, stats in by_item.items():
            scaler = scaler_cls()
            if scaler_key == "standard":
                mean = stats["sum"] / stats["count"]
                var = stats["sumsq"] / stats["count"] - mean**2
                std = np.sqrt(np.maximum(var, 0.0))
                std = np.where(std == 0, 1, std)
                scaler.mean = mean
                scaler.std = std
            else:
                data_range = stats["max"] - stats["min"]
                data_range[data_range == 0.0] = 1.0
                scaler.min = stats["min"]
                scaler.max = stats["max"]
                scaler.data_range_ = data_range
            scalers[item_id] = scaler

        return scalers


    def _slice_all_splits(self, need_train=True, need_val=True, need_test=True):

        base_cols = self.data.column_names
        input_len = self.config.sample.input_length
        val_spec = self.config.dataset.val_split
        test_spec = self.config.dataset.test_split

        def compute_split_indices(timestamps):
            ts_array = np.asarray(timestamps)
            L = len(ts_array)

            def to_idx(cfg):
                if isinstance(cfg, float):
                    return int(L * cfg)
                if isinstance(cfg, int):
                    if cfg < 0:
                        return max(L + cfg, 0)
                    else:
                        return min(cfg, L)
                if isinstance(cfg, str):
                    ts = np.datetime64(pd.to_datetime(cfg))
                    return int(np.searchsorted(ts_array, ts, side="right") - 1)
                raise ValueError("config.dataset split values should be float, int or str")

            val_idx = to_idx(val_spec)
            test_idx = to_idx(test_spec)
            val_idx = max(0, min(val_idx, L))
            test_idx = max(val_idx, min(test_idx, L))
            return val_idx, test_idx

        def slice_row(row, start=None, end=None):
            out = {}
            for k, v in row.items():
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    out[k] = v[start:end]
                elif isinstance(v, np.ndarray) and v.ndim == 2:
                    out[k] = v[start:end, :]
                elif isinstance(v, list) and len(v) > 0:
                    out[k] = v[start:end]
                else:
                    out[k] = v
            return out

        def slice_batch(batch):
            out = {}
            bsz = len(batch["timestamp"])

            def init(prefix):
                return {f"{prefix}__{c}": [] for c in base_cols}

            tr_out = init("train") if need_train else None
            vl_out = init("val") if need_val else None
            te_out = init("test") if need_test else None

            for i in range(bsz):
                row = {k: batch[k][i] for k in batch}
                val_idx, test_idx = compute_split_indices(row["timestamp"])

                if need_train:
                    tr_row = slice_row(row, end=val_idx)
                    for c in base_cols:
                        tr_out[f"train__{c}"].append(tr_row[c])

                if need_val:
                    vl_row = slice_row(
                        row,
                        start=max(val_idx - input_len, 0),
                        end=test_idx,
                    )
                    for c in base_cols:
                        vl_out[f"val__{c}"].append(vl_row[c])

                if need_test:
                    te_row = slice_row(
                        row,
                        start=max(test_idx - input_len, 0),
                    )
                    for c in base_cols:
                        te_out[f"test__{c}"].append(te_row[c])

            if need_train:
                out.update(tr_out)
            if need_val:
                out.update(vl_out)
            if need_test:
                out.update(te_out)
            return out

        split_ds = self.data.map(
            slice_batch,
            batched=True,
            num_proc=1,
            desc="Slice splits",
            load_from_cache_file=False,
        )

        def build_split(prefix):
            cols = [c for c in split_ds.column_names if c.startswith(f"{prefix}__")]
            if not cols:
                return None
            renamed = {c: c.split("__", 1)[1] for c in cols}
            return split_ds.select_columns(cols).rename_columns(renamed)

        train_ds = build_split("train") if need_train else None
        val_ds = build_split("val") if need_val else None
        test_ds = build_split("test") if need_test else None

        return train_ds, val_ds, test_ds

    def _apply_target_scaling(self, ds):
        """Apply already-fitted target scaler via Dataset.map."""
        if ds is None:
            return None

        mode = self.config.scalers.scaling_mode
        target_col = self.config.dataset.target
        id_col = self.config.dataset.id_column

        if mode == "global":
            scaler = self.target_scaler

            def scale_batch(batch):
                scaled = []
                for arr in batch[target_col]:
                    arr_np = np.asarray(arr)
                    scaled.append(scaler.transform(arr_np).astype(np.float32))
                batch[target_col] = scaled
                return batch

        elif mode == "per_id":
            scaler_dict = self.target_scaler

            def scale_batch(batch):
                scaled = []
                for item_id, arr in zip(batch[id_col], batch[target_col], strict=True):
                    scaler = scaler_dict[str(item_id)]
                    arr_np = np.asarray(arr)
                    scaled.append(scaler.transform(arr_np).astype(np.float32))
                batch[target_col] = scaled
                return batch

        elif mode == "none":
            return ds
        else:
            raise ValueError(f"Unknown scaling mode: {mode}")

        return ds.map(
            scale_batch,
            batched=True,
            num_proc=1,
            desc="Scale target",
            load_from_cache_file=False,
        )

    def _scale_feature_rows(self, ds):
        """Scale past/future features for a split via Dataset.map."""
        if ds is None:
            return None

        if self.config.dataset.get("scale_columns", None) is not None:
            cols = self.config.dataset.scale_columns
        else:
            cols = self.config.dataset.past_columns + self.config.dataset.future_columns
        if not cols:
            return ds

        mode = getattr(self.config.scalers, "feature_scaling_mode", "global")

        if mode == "none" or self.feature_scaler is None:
            return ds

        if mode == "global":
            scaler = self.feature_scaler

            def scale_batch(batch):
                scaled_cols = {c: [] for c in cols}
                for i in range(len(batch[cols[0]])):
                    mat = np.stack([np.asarray(batch[c][i]) for c in cols], axis=-1)
                    mat_scaled = scaler.transform(mat)
                    for j, c in enumerate(cols):
                        scaled_cols[c].append(mat_scaled[:, j].astype(np.float32))
                batch.update(scaled_cols)
                return batch

        elif mode == "per_id":
            scaler_dict = self.feature_scaler
            id_col = self.config.dataset.id_column

            def scale_batch(batch):
                scaled_cols = {c: [] for c in cols}
                for item_id, *row_vals in zip(batch[id_col], *[batch[c] for c in cols], strict=True):
                    mat = np.stack([np.asarray(v) for v in row_vals], axis=-1)
                    scaler = scaler_dict[str(item_id)]
                    mat_scaled = scaler.transform(mat)
                    for j, c in enumerate(cols):
                        scaled_cols[c].append(mat_scaled[:, j].astype(np.float32))
                batch.update(scaled_cols)
                return batch
        else:
            raise ValueError(f"Unknown feature scaling mode: {mode}")

        return ds.map(
            scale_batch,
            batched=True,
            num_proc=1,
            desc="Scale features",
            load_from_cache_file=False,
        )

    def _static_string_to_int(self, ds, static_cols):

        features = ds.features.copy()

        for col in static_cols:
            if isinstance(features[col], datasets.Value) and features[col].dtype == "string":
                vocab = sorted(set(ds[col]))  # extract vocabulary
                features[col] = datasets.ClassLabel(names=vocab)
            elif isinstance(features[col], datasets.Value) and features[col].dtype in ["int64", "float32", "float64"]:
                col_vals = np.array(ds[col])

                if col_vals.dtype.kind == "f":
                    if not np.all(np.isfinite(col_vals)):
                        raise ValueError(f"Column {col} contains NaN/Inf but missing handling is disabled.")
                    if not np.all(col_vals == np.round(col_vals)):
                        raise ValueError(f"Column {col} has non-integer float values; cannot treat as categorical IDs safely.")

                uniq = np.unique(col_vals.astype(np.int64))
                mapping = {int(v): i for i, v in enumerate(uniq)}  # dense 0..K-1
                num_classes = len(uniq)

                def map_fn(x, *, col, mapping):
                    v = x[col]
                    # no missing handling: if v is bad, this will raise
                    return {col: mapping[int(v)]}

                ds = ds.map(map_fn, fn_kwargs={"col": col, "mapping": mapping})
                features[col] = datasets.ClassLabel(num_classes=num_classes)
            else:
                raise NotImplementedError(f"Static column {col} is not a string Value column.")

        return ds.cast(features)
    
    def _dynamic_string_to_int(self, ds, string_cols):
        features = ds.features.copy()

        for col in string_cols:
            def fn(batch, *, col):
                out = []
                for seq in batch[col]:
                    out.append([0 if (x is None or x == "") else 1 for x in seq])
                return {col: out}
        
            ds = ds.map(
                fn,
                batched=True,
                num_proc=1,
                fn_kwargs={"col": col},
                desc=f"Convert string to int for column {col}",
                load_from_cache_file=False,
            )
            features[col] = Sequence(Value("int64"))
        return ds.cast(features)

    def _add_cyclical_time_features(
        self,
        dataset,
        timestamp_key="timestamp",
    ):
        def _map_fn(example):
            ts = example[timestamp_key]  # list[datetime.datetime]
            day_of_week = np.array([t.weekday() for t in ts])              # 0-6
            day_of_month = np.array([t.day - 1 for t in ts])               # 0-30
            day_of_year = np.array([t.timetuple().tm_yday - 1 for t in ts])# 0-364
            hour_of_day = np.array([t.hour for t in ts])                  # 0-23

            return {
                "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7).tolist(),
                "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7).tolist(),

                "day_of_month_sin": np.sin(2 * np.pi * day_of_month / 31).tolist(),
                "day_of_month_cos": np.cos(2 * np.pi * day_of_month / 31).tolist(),

                "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25).tolist(),
                "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25).tolist(),

                "hour_of_day_sin": np.sin(2 * np.pi * hour_of_day / 24).tolist(),
                "hour_of_day_cos": np.cos(2 * np.pi * hour_of_day / 24).tolist(),
            }
        
        return dataset.map(_map_fn, batched=False, num_proc=1, desc="Add cyclical time features", load_from_cache_file=False)

    def _apply_log_transform(self, ds, log_cols):
        def fn(batch):
            out = {}
            for col in log_cols:
                arrs = []
                for arr in batch[col]:
                    arr_np = np.asarray(arr, dtype=np.float32)
                    if np.any(arr_np < 0):
                        raise ValueError(f"Column {col} contains negative values; cannot apply log transform.")
                    arrs.append(np.log1p(arr_np).astype(np.float32))
                out[col] = arrs
            return out

        return ds.map(
            fn,
            batched=True,
            num_proc=1,
            desc=f"Apply log transform to {log_cols}",
            load_from_cache_file=False,
        )
        

class FevDataset(Dataset):
    def __init__(
        self,
        dataset,
        history_length: int,     
        forecast_length: int, 
        stride: int,
        target_col: str,
        id_col: str,
        past_cols: list | None = None,
        future_cols: list | None = None,
        static_cols: list | None = None,
    ) -> None:
        
        # Speed up row access by returning numpy arrays directly instead of Python lists.
        # Limits the formatted columns to those we actually use.
        columns_for_numpy = ["timestamp", target_col, id_col] + (past_cols or []) + (future_cols or []) + (static_cols or [])
        for extra in ("starts", "target_arr", "past_mat", "future_mat", "static_vec"):
            if extra in dataset.column_names:
                columns_for_numpy.append(extra)
        columns_for_numpy = list(dict.fromkeys(columns_for_numpy))  # preserve order, drop dups
        self.data = dataset.with_format("numpy", columns=columns_for_numpy)
        self.x_size = history_length
        self.y_size = forecast_length
        self.window_size = self.x_size + self.y_size
        self.stride = stride
        
        self.target_col = target_col
        self.id_col = id_col
        # Pull ids once (column-level) to avoid per-row dict materialization.
        self.id_list = [str(i) for i in self.data[id_col]]
        self.past_cols = past_cols or []
        self.future_cols = future_cols or []
        self.static_cols = static_cols or []
        
        # Pre-compute per-series metadata; drop series that are too short.
        self.timestamps = []
        self.starts_per_series = []
        self.valid_series_indices = []
        self._targets = []
        self._past_feats = []    # list of [T, num_past_features] or None
        self._future_feats = []  # list of [T, num_future_features] or None
        self._static = []        # list of [num_static_features] or []
        
        precomputed_cols = {"starts", "target_arr", "past_mat", "future_mat", "static_vec"}
        has_precomputed = precomputed_cols.issubset(set(self.data.column_names))
        
        if has_precomputed:
            self.timestamps = list(self.data["timestamp"])
            self.starts_per_series = list(self.data["starts"])
            self.valid_series_indices = list(range(len(self.data)))
            self._targets = list(self.data["target_arr"])
            self._past_feats = list(self.data["past_mat"])
            self._future_feats = list(self.data["future_mat"])
            self._static = list(self.data["static_vec"])
            
        else:
            raise ValueError("FevDataset requires precomputed columns")

        if not self.starts_per_series:
            raise ValueError("No series have sufficient length for the requested window size.")

        # Keep ids for valid series only
        self.id_list = [self.id_list[i] for i in self.valid_series_indices]
        self.n_series = len(self.starts_per_series)
        self.n_windows_per_series = [len(starts) for starts in self.starts_per_series]
        self.cum_windows = np.cumsum(self.n_windows_per_series)
        self.total_windows = int(self.cum_windows[-1])

        logger.info(f"Total candidate windows (all series): {self.total_windows}")
        logger.info(f"Valid windows after NaN check: {self.total_windows}")

    def __len__(self) -> int:
        return self.total_windows

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
                  • past_target     — Tensor of shape (x_size, num_targets)
                  • past_features   — Tensor of shape (feature_lookback, num_past_features),
                                      or empty Tensor if no past features
                  • future_features — Tensor of shape (feature_lookback + feature_lookahead,
                                      num_future_features), or empty Tensor if no future features
                  • static          — Tensor of shape (num_static_features),
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
        # Map flat idx -> (series_idx, window_idx)
        series_idx = int(np.searchsorted(self.cum_windows, idx, side="right"))
        prev_cum = self.cum_windows[series_idx - 1] if series_idx > 0 else 0
        win_idx = idx - prev_cum
        start = int(self.starts_per_series[series_idx][win_idx])

        ctx_start = start
        ctx_end   = start + self.x_size
        y_start   = ctx_end
        y_end     = ctx_end + self.y_size

        # ---------- target ----------
        target_arr = self._targets[series_idx]  # [T]
        past_target = torch.from_numpy(target_arr[ctx_start:ctx_end]).unsqueeze(-1)  # [x_size, 1]
        out         = torch.from_numpy(target_arr[y_start:y_end]).unsqueeze(-1)      # [y_size, 1]

        # ---------- past dynamic features ----------
        past_mat = self._past_feats[series_idx]  # [T, num_past_features] or None
        if past_mat is not None:
            past_features = torch.from_numpy(past_mat[ctx_start:ctx_end])  # [x_size, num_past_features]
        else:
            past_features = torch.empty(self.x_size, 0, dtype=torch.float32)

        # ---------- future dynamic features ----------
        fut_mat = self._future_feats[series_idx]  # [T, num_future_features] or None
        if fut_mat is not None:
            future_features = torch.from_numpy(fut_mat[ctx_start:y_end])  # [x_size+y_size, num_future_features]
        else:
            future_features = torch.empty(self.window_size, 0, dtype=torch.float32)

        # ---------- static features ----------
        static_vec = self._static[series_idx]  # [num_static_features] or []
        static = torch.from_numpy(static_vec).long() if static_vec.size > 0 else torch.empty(0, dtype=torch.long)

        inp = {
            "past_target": past_target,          # [x_size, 1]
            "past_features": past_features,      # [x_size, num_past_features]
            "future_features": future_features,  # [x_size+y_size, num_future_features]
            "static": static,                   # [num_static_features]
            "id": self.id_list[series_idx],
        }

        return inp, out
