import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class RasterRecord:
    timestep: int
    filename: str
    path: Path
    date: Optional[datetime] = None


@dataclass
class RasterMetadata:
    width: int
    height: int
    crs: object
    transform: Affine
    profile: dict
    nodata: Optional[float]
    dtype: str
    tags: dict


@dataclass
class RasterNormalizer:
    data_min: float
    data_max: float
    data_mean: float
    data_std: float

    @classmethod
    def fit(cls, reader: "RasterSequenceReader", sample_limit: Optional[int] = None) -> "RasterNormalizer":
        total = len(reader.catalog)
        if total == 0:
            raise ValueError("Cannot fit normalizer on an empty catalog.")

        if sample_limit is None or sample_limit >= total:
            indexes = range(total)
        else:
            indexes = np.linspace(0, total - 1, sample_limit, dtype=np.int64).tolist()

        count = 0
        data_min = math.inf
        data_max = -math.inf
        running_sum = 0.0
        running_sq_sum = 0.0

        for index in indexes:
            frame = reader.read_frame(index).numpy()
            finite = frame[np.isfinite(frame)]
            if finite.size == 0:
                continue
            data_min = min(data_min, float(finite.min()))
            data_max = max(data_max, float(finite.max()))
            running_sum += float(finite.sum())
            running_sq_sum += float(np.square(finite).sum())
            count += int(finite.size)

        if count == 0 or not math.isfinite(data_min) or not math.isfinite(data_max):
            raise ValueError("No finite raster values were found when fitting normalizer.")

        data_mean = running_sum / count
        variance = max((running_sq_sum / count) - data_mean**2, 0.0)
        data_std = math.sqrt(variance)
        return cls(data_min=data_min, data_max=data_max, data_mean=data_mean, data_std=data_std)

    @property
    def data_scale(self) -> float:
        return max((self.data_max - self.data_min) / 2.0, 1e-8)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        denom = max(self.data_max - self.data_min, 1e-8)
        return 2.0 * (tensor - self.data_min) / denom - 1.0

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor + 1.0) / 2.0) * (self.data_max - self.data_min) + self.data_min


class RasterCatalog:
    def __init__(self, records: Sequence[RasterRecord]) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("Raster catalog is empty.")

    @classmethod
    def from_directory(
        cls,
        data_path: Path,
        file_format: str = "tif",
        validate_files: bool = False,
    ) -> "RasterCatalog":
        data_path = Path(data_path)
        suffixes = {".tif", ".tiff"} if file_format.lower() in {"tif", "tiff"} else {f".{file_format.lower()}"}
        dated_records: List[Tuple[int, datetime, Path]] = []
        for path in data_path.iterdir():
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if not re.fullmatch(r"\d{8}", path.stem):
                continue
            date = datetime.strptime(path.stem, "%Y%m%d")
            dated_records.append((int(path.stem), date, path))

        dated_records.sort(key=lambda item: item[0])
        records = [
            RasterRecord(timestep=index + 1, filename=path.name, path=path, date=date)
            for index, (_date_int, date, path) in enumerate(dated_records)
        ]
        if validate_files:
            missing = [str(record.path) for record in records if not record.path.exists()]
            if missing:
                raise FileNotFoundError("\n".join(missing[:10]))
        return cls(records)

    @classmethod
    def from_csv(
        cls,
        csv_file: Path,
        data_path: Path,
        validate_files: bool = False,
    ) -> "RasterCatalog":
        csv_file = Path(csv_file)
        data_path = Path(data_path)
        records: List[RasterRecord] = []

        with csv_file.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                if not row or not row[0].strip() or row[0].strip().startswith("#"):
                    continue
                try:
                    timestep = int(row[0].strip())
                except ValueError:
                    if not records:
                        continue
                    raise ValueError(f"Invalid timestep at row {row_number}: {row[0]!r}") from None
                if len(row) < 2 or not row[1].strip():
                    raise ValueError(f"Missing filename at row {row_number}.")
                filename = row[1].strip()
                date = None
                stem = Path(filename).stem
                if re.fullmatch(r"\d{8}", stem):
                    date = datetime.strptime(stem, "%Y%m%d")
                records.append(RasterRecord(timestep=timestep, filename=filename, path=data_path / filename, date=date))

        records.sort(key=lambda item: item.timestep)
        expected = list(range(1, len(records) + 1))
        actual = [record.timestep for record in records]
        if actual != expected:
            raise ValueError("CSV timesteps must be continuous and start from 1.")

        if validate_files:
            missing = [str(record.path) for record in records if not record.path.exists()]
            if missing:
                preview = "\n".join(missing[:10])
                raise FileNotFoundError(f"{len(missing)} raster files are missing. First entries:\n{preview}")
        elif not records[0].path.exists():
            raise FileNotFoundError(f"First raster file does not exist: {records[0].path}")

        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RasterRecord:
        return self.records[index]

    @property
    def total_timesteps(self) -> int:
        return len(self.records)

    def fixed_horizon_available(self, seq_length: int, pred_length: int) -> bool:
        return len(self.records) >= seq_length + pred_length

    def target_time_deltas(self, seq_length: int, pred_length: int, fallback: float = 1.0) -> torch.Tensor:
        return self.window_time_deltas(seq_length, pred_length, fallback=fallback)

    def window_time_deltas(self, input_end: int, pred_length: int, fallback: float = 1.0) -> torch.Tensor:
        deltas = []
        for step in range(pred_length):
            previous = self.records[input_end + step - 1]
            current = self.records[input_end + step]
            if previous.date is not None and current.date is not None:
                delta = max((current.date - previous.date).days, 1)
            else:
                delta = fallback
            deltas.append(float(delta))
        return torch.tensor(deltas, dtype=torch.float32)


class RasterSequenceReader:
    def __init__(
        self,
        catalog: RasterCatalog,
        band: int = 1,
        variable_name: Optional[str] = None,
        resize_shape: Optional[Tuple[int, int]] = None,
        apply_scale_offset: bool = True,
        invalid_fill_value: float = 0.0,
        unreadable_policy: str = "nearest",
        max_fallback_search: Optional[int] = None,
        unreadable_report_limit: int = 20,
        cache_size: int = 0,
    ) -> None:
        self.catalog = catalog
        self.band = band
        self.variable_name = variable_name
        self.resize_shape = resize_shape
        self.apply_scale_offset = apply_scale_offset
        self.invalid_fill_value = invalid_fill_value
        self.unreadable_policy = unreadable_policy
        self.max_fallback_search = max_fallback_search
        self.unreadable_report_limit = unreadable_report_limit
        self._unreadable_report_count = 0
        self._reported_unreadable = set()
        self._readable_indexes = set()
        self._unreadable_indexes = set()
        self.metadata = self._read_metadata()
        if cache_size > 0:
            self._cached_read = lru_cache(maxsize=cache_size)(self._read_frame_uncached)
        else:
            self._cached_read = self._read_frame_uncached

    def _source_path(self, path: Path) -> str:
        if self.variable_name and path.suffix.lower() == ".nc":
            return f'NETCDF:"{path}":{self.variable_name}'
        return str(path)

    def _read_metadata(self) -> RasterMetadata:
        last_error = None
        metadata_scan_limit = len(self.catalog) if self.max_fallback_search is None else self.max_fallback_search + 1
        for record in self.catalog.records[: min(len(self.catalog), metadata_scan_limit)]:
            try:
                with rasterio.open(self._source_path(record.path)) as src:
                    if self.resize_shape:
                        height, width = self.resize_shape
                        transform = src.transform * Affine.scale(src.width / width, src.height / height)
                    else:
                        height, width = src.height, src.width
                        transform = src.transform
                    profile = src.profile.copy()
                    profile.update(height=height, width=width, transform=transform)
                    return RasterMetadata(
                        width=width,
                        height=height,
                        crs=src.crs,
                        transform=transform,
                        profile=profile,
                        nodata=src.nodata,
                        dtype=src.dtypes[self.band - 1],
                        tags={**src.tags(), **src.tags(self.band)},
                    )
            except RasterioIOError as exc:
                last_error = exc
        raise RasterioIOError(f"No readable raster was found for metadata. Last error: {last_error}")

    def read_frame(self, index: int) -> torch.Tensor:
        return self._cached_read(index)

    def clear_cache(self) -> None:
        cache_clear = getattr(self._cached_read, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    def _read_frame_uncached(self, index: int) -> torch.Tensor:
        record = self.catalog[index]
        try:
            return self._read_record(record)
        except RasterioIOError as exc:
            self._unreadable_indexes.add(index)
            return self._handle_unreadable_frame(index, exc)

    def _read_record(self, record: RasterRecord) -> torch.Tensor:
        with rasterio.open(self._source_path(record.path)) as src:
            read_kwargs = {"masked": True}
            if self.resize_shape:
                read_kwargs["out_shape"] = self.resize_shape
                read_kwargs["resampling"] = Resampling.bilinear
            raw = src.read(self.band, **read_kwargs)
            tags = {**src.tags(), **src.tags(self.band)}

        masked = np.ma.asarray(raw)
        data = masked.astype(np.float32).filled(np.nan)
        mask = np.ma.getmaskarray(masked)

        valid_range = self._parse_pair_tag(tags, "valid_range")
        if valid_range is not None:
            low, high = valid_range
            mask = np.logical_or(mask, np.logical_or(data < low, data > high))

        flag_values = self._parse_list_tag(tags, "flag_values")
        if flag_values:
            mask = np.logical_or(mask, np.isin(data, flag_values))

        data = data.astype(np.float32, copy=False)
        data[mask] = np.nan

        if self.apply_scale_offset:
            scale = self._parse_float_tag(tags, "scale_factor", default=1.0)
            offset = self._parse_float_tag(tags, "add_offset", default=0.0)
            data = data * scale + offset

        data = np.nan_to_num(
            data,
            nan=self.invalid_fill_value,
            posinf=self.invalid_fill_value,
            neginf=self.invalid_fill_value,
        )
        return torch.from_numpy(data.astype(np.float32, copy=False)).unsqueeze(0)

    def _read_record_by_index(self, index: int) -> torch.Tensor:
        if index in self._unreadable_indexes:
            raise RasterioIOError(f"Known unreadable index: {index}")
        try:
            tensor = self._read_record(self.catalog[index])
            self._readable_indexes.add(index)
            return tensor
        except RasterioIOError:
            self._unreadable_indexes.add(index)
            raise

    def _handle_unreadable_frame(self, index: int, exc: RasterioIOError) -> torch.Tensor:
        record = self.catalog[index]
        should_report = (
            record.path not in self._reported_unreadable
            and self._unreadable_report_count < self.unreadable_report_limit
        )
        if should_report:
            print(
                f"Warning: unreadable raster at timestep={record.timestep}, "
                f"file={record.filename}. policy={self.unreadable_policy}."
            )
            self._reported_unreadable.add(record.path)
            self._unreadable_report_count += 1
            if self._unreadable_report_count == self.unreadable_report_limit:
                print("Warning: unreadable raster report limit reached; further warnings are suppressed.")

        if self.unreadable_policy == "raise":
            raise exc
        if self.unreadable_policy == "zeros":
            return torch.full(
                (1, self.metadata.height, self.metadata.width),
                fill_value=float(self.invalid_fill_value),
                dtype=torch.float32,
            )

        max_distance = self.max_fallback_search
        if max_distance is None:
            max_distance = max(index, len(self.catalog) - index - 1)

        for distance in range(1, max_distance + 1):
            for candidate_index in (index - distance, index + distance):
                if candidate_index < 0 or candidate_index >= len(self.catalog):
                    continue
                try:
                    fallback = self._read_record_by_index(candidate_index)
                    fallback_record = self.catalog[candidate_index]
                    if should_report:
                        print(
                            f"  Filled timestep={record.timestep} from nearest readable "
                            f"timestep={fallback_record.timestep}."
                        )
                    return fallback
                except RasterioIOError:
                    continue

        raise RasterioIOError(
            f"Raster at timestep={record.timestep} is unreadable and no fallback was found "
            f"within {self.max_fallback_search or 'all'} timesteps."
        )

    def output_profile(self, count: int = 1, dtype: str = "float32") -> dict:
        profile = self.metadata.profile.copy()
        profile.update(
            driver="GTiff",
            height=self.metadata.height,
            width=self.metadata.width,
            count=count,
            dtype=dtype,
            crs=self.metadata.crs,
            transform=self.metadata.transform,
            nodata=None,
        )
        return profile

    @staticmethod
    def _tag_value(tags: dict, suffix: str) -> Optional[str]:
        for key, value in tags.items():
            if key == suffix or key.endswith(f"#{suffix}"):
                return value
        return None

    @classmethod
    def _parse_float_tag(cls, tags: dict, suffix: str, default: float) -> float:
        value = cls._tag_value(tags, suffix)
        if value is None:
            return default
        try:
            return float(str(value).strip("{} "))
        except ValueError:
            return default

    @classmethod
    def _parse_pair_tag(cls, tags: dict, suffix: str) -> Optional[Tuple[float, float]]:
        values = cls._parse_list_tag(tags, suffix)
        if len(values) >= 2:
            return float(values[0]), float(values[1])
        return None

    @classmethod
    def _parse_list_tag(cls, tags: dict, suffix: str) -> List[float]:
        value = cls._tag_value(tags, suffix)
        if value is None:
            return []
        cleaned = str(value).strip("{}[]() ")
        if not cleaned:
            return []
        result = []
        for part in cleaned.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                result.append(float(part))
            except ValueError:
                continue
        return result


class ParameterRasterReader:
    """Loads PINN parameter rasters.

    The legacy project stored layer-wise rasters in Parameter/1..6. The 1987
    soft-soil-3 setup stores one raster per parameter directly in Parameter/.
    Both layouts are supported: flat rasters are returned as [1, H, W], while
    legacy layer folders are stacked as [L, H, W].
    """

    def __init__(
        self,
        parameter_dir: Path,
        reference_metadata: RasterMetadata,
        required_names: Sequence[str] = (),
        nodata_fill: float = 0.0,
    ) -> None:
        self.parameter_dir = Path(parameter_dir)
        self.reference_metadata = reference_metadata
        self.required_names = tuple(required_names)
        self.nodata_fill = float(nodata_fill)
        self.layer_dirs: List[Path] = []
        self.flat_parameter_paths: List[Path] = []
        self._parameters: Optional[Dict[str, torch.Tensor]] = None

    def load(self) -> "ParameterRasterReader":
        if not self.parameter_dir.exists():
            raise FileNotFoundError(f"Parameter raster directory does not exist: {self.parameter_dir}")

        self.layer_dirs = self._discover_layer_dirs()
        self.flat_parameter_paths = sorted(self.parameter_dir.glob("*.tif"))
        if not self.layer_dirs and not self.flat_parameter_paths:
            raise FileNotFoundError(f"No parameter rasters were found in: {self.parameter_dir}")

        names = sorted(
            {path.stem for layer_dir in self.layer_dirs for path in layer_dir.glob("*.tif")}
            | {path.stem for path in self.flat_parameter_paths}
        )
        missing = [name for name in self.required_names if name not in names]
        if missing:
            raise FileNotFoundError(
                "Missing required parameter rasters: "
                + ", ".join(missing)
                + f". Directory: {self.parameter_dir}"
            )

        parameters: Dict[str, torch.Tensor] = {}
        for name in names:
            flat_path = self.parameter_dir / f"{name}.tif"
            if flat_path.exists():
                layers = [self._read_parameter(flat_path)]
            else:
                layers = []
                for layer_dir in self.layer_dirs:
                    path = layer_dir / f"{name}.tif"
                    if not path.exists():
                        layers.append(
                            np.full(
                                (self.reference_metadata.height, self.reference_metadata.width),
                                self.nodata_fill,
                                dtype=np.float32,
                            )
                        )
                        continue
                    layers.append(self._read_parameter(path))
            parameters[name] = torch.from_numpy(np.stack(layers).astype(np.float32))

        self._parameters = parameters
        return self

    def _discover_layer_dirs(self) -> List[Path]:
        dirs = [path for path in self.parameter_dir.iterdir() if path.is_dir() and path.name.isdigit()]
        return sorted(dirs, key=lambda path: int(path.name))

    def _read_parameter(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            if src.width == self.reference_metadata.width and src.height == self.reference_metadata.height:
                data = src.read(1, masked=True).astype(np.float32)
                filled = np.ma.asarray(data).filled(np.nan).astype(np.float32)
            else:
                filled = src.read(
                    1,
                    out_shape=(self.reference_metadata.height, self.reference_metadata.width),
                    resampling=Resampling.bilinear,
                    masked=True,
                ).filled(np.nan).astype(np.float32)
        filled[~np.isfinite(filled)] = self.nodata_fill
        filled[filled == -9999.0] = self.nodata_fill
        return filled

    def parameters(self) -> Dict[str, torch.Tensor]:
        if self._parameters is None:
            self.load()
        assert self._parameters is not None
        return {name: tensor.clone() for name, tensor in self._parameters.items()}

    @property
    def layer_count(self) -> int:
        return len(self.layer_dirs) if self.layer_dirs else (1 if self.flat_parameter_paths else 0)


class FixedHorizonRasterDataset(Dataset):
    def __init__(
        self,
        reader: RasterSequenceReader,
        seq_length: int,
        pred_length: int,
        normalizer: Optional[RasterNormalizer],
        split_mask: torch.Tensor,
        split_name: str,
        parameter_reader: Optional[ParameterRasterReader] = None,
        time_step_fallback: float = 1.0,
        input_start: int = 0,
    ) -> None:
        self.reader = reader
        self.catalog = reader.catalog
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.normalizer = normalizer
        self.split_mask = split_mask.to(dtype=torch.float32)
        self.split_name = split_name
        self.parameter_reader = parameter_reader
        self.input_start = int(input_start)
        self.input_end = self.input_start + seq_length
        self.target_start = self.input_end
        self.target_end = self.target_start + pred_length
        self.target_indexes = torch.arange(self.target_start, self.target_end, dtype=torch.long)
        self.time_deltas = self.catalog.window_time_deltas(self.input_end, pred_length, fallback=time_step_fallback)
        if self.input_start < 0 or self.target_end > len(self.catalog):
            raise ValueError(
                f"Invalid {split_name} window: input_start={self.input_start}, "
                f"seq_length={seq_length}, pred_length={pred_length}, timesteps={len(self.catalog)}."
            )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        inputs = torch.stack([self.reader.read_frame(i) for i in range(self.input_start, self.input_end)], dim=0)
        targets = torch.stack(
            [self.reader.read_frame(i) for i in range(self.target_start, self.target_end)],
            dim=0,
        )
        if self.normalizer is not None:
            inputs = self.normalizer.normalize(inputs)
            targets = self.normalizer.normalize(targets)
        parameters = self.parameter_reader.parameters() if self.parameter_reader is not None else {}
        return inputs, targets, self.target_indexes.clone(), parameters, self.split_mask.clone(), self.time_deltas.clone()


class TargetIndexedRasterDataset(Dataset):
    def __init__(
        self,
        reader: RasterSequenceReader,
        seq_length: int,
        pred_length: int,
        normalizer: Optional[RasterNormalizer],
        split_mask: torch.Tensor,
        split_name: str,
        parameter_reader: Optional[ParameterRasterReader] = None,
        time_step_fallback: float = 1.0,
        target_start_index: int = 0,
        target_end_index: Optional[int] = None,
        require_full_history: bool = True,
    ) -> None:
        self.reader = reader
        self.catalog = reader.catalog
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.normalizer = normalizer
        self.split_mask = split_mask.to(dtype=torch.float32)
        self.split_name = split_name
        self.parameter_reader = parameter_reader
        self.target_start_index = int(target_start_index)
        self.target_end_index = len(self.catalog) if target_end_index is None else int(target_end_index)
        self.time_step_fallback = float(time_step_fallback)
        self.require_full_history = bool(require_full_history)
        self.num_samples = self.target_end_index - self.target_start_index - self.pred_length + 1
        if self.target_start_index < 0 or self.target_end_index > len(self.catalog) or self.num_samples <= 0:
            raise ValueError(
                f"Invalid {split_name} target-indexed range: target_start={self.target_start_index}, "
                f"target_end={self.target_end_index}, seq_length={seq_length}, pred_length={pred_length}, "
                f"timesteps={len(self.catalog)}."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        target_start = self.target_start_index + index
        target_end = target_start + self.pred_length
        input_start = target_start - self.seq_length
        if self.require_full_history and input_start < 0:
            raise ValueError(
                f"{self.split_name} sample target_start={target_start} does not have "
                f"{self.seq_length} previous frames."
            )
        input_indexes = [max(0, input_start + offset) for offset in range(self.seq_length)]
        inputs = torch.stack([self.reader.read_frame(i) for i in input_indexes], dim=0)
        targets = torch.stack([self.reader.read_frame(i) for i in range(target_start, target_end)], dim=0)
        if self.normalizer is not None:
            inputs = self.normalizer.normalize(inputs)
            targets = self.normalizer.normalize(targets)
        target_indexes = torch.arange(target_start, target_end, dtype=torch.long)
        time_deltas = self._time_deltas(target_start)
        parameters = self.parameter_reader.parameters() if self.parameter_reader is not None else {}
        return inputs, targets, target_indexes, parameters, self.split_mask.clone(), time_deltas

    def _time_deltas(self, target_start: int) -> torch.Tensor:
        deltas = []
        for step in range(self.pred_length):
            current_index = target_start + step
            previous_index = current_index - 1
            if previous_index >= 0:
                previous = self.catalog[previous_index]
                current = self.catalog[current_index]
                if previous.date is not None and current.date is not None:
                    deltas.append(float(max((current.date - previous.date).days, 1)))
                    continue
            deltas.append(self.time_step_fallback)
        return torch.tensor(deltas, dtype=torch.float32)


def build_spatial_masks(height: int, width: int, seed: int, train_ratio: float, test_ratio: float):
    total = height * width
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count

    masks = {}
    for name, indexes in (
        ("train", permutation[:train_count]),
        ("test", permutation[train_count : train_count + test_count]),
        ("val", permutation[train_count + test_count :]),
    ):
        flat = np.zeros(total, dtype=np.float32)
        flat[indexes] = 1.0
        masks[name] = torch.from_numpy(flat.reshape(1, height, width))
    return masks, {"train": train_count, "test": test_count, "val": val_count, "total": total}


class LGTMDataModule:
    def __init__(self, config) -> None:
        self.config = config
        self.catalog: Optional[RasterCatalog] = None
        self.reader: Optional[RasterSequenceReader] = None
        self.normalizer: Optional[RasterNormalizer] = None
        self.parameter_reader: Optional[ParameterRasterReader] = None
        self.train_dataset: Optional[TargetIndexedRasterDataset] = None
        self.test_dataset: Optional[TargetIndexedRasterDataset] = None
        self.val_dataset: Optional[TargetIndexedRasterDataset] = None
        self.predict_dataset: Optional[TargetIndexedRasterDataset] = None
        self.split_counts: Dict[str, int] = {}

    def setup(self) -> "LGTMDataModule":
        if self.config.use_csv_input:
            self.catalog = RasterCatalog.from_csv(
                csv_file=self.config.csv_file,
                data_path=self.config.data_path,
                validate_files=self.config.validate_files,
            )
        else:
            self.catalog = RasterCatalog.from_directory(
                data_path=self.config.data_path,
                file_format=self.config.file_format or "tif",
                validate_files=self.config.validate_files,
            )

        self.reader = RasterSequenceReader(
            catalog=self.catalog,
            band=self.config.raster_band,
            variable_name=self.config.raster_variable,
            resize_shape=self.config.resize_shape,
            apply_scale_offset=self.config.apply_scale_offset,
            invalid_fill_value=self.config.invalid_fill_value,
            unreadable_policy=self.config.unreadable_policy,
            max_fallback_search=self.config.max_fallback_search,
            unreadable_report_limit=self.config.unreadable_report_limit,
            cache_size=self.config.cache_size,
        )
        self.normalizer = RasterNormalizer.fit(self.reader, sample_limit=self.config.stats_sample_limit)

        self.parameter_reader = None
        if getattr(self.config, "load_pinn_parameters", False):
            self.parameter_reader = ParameterRasterReader(
                parameter_dir=self.config.parameter_raster_dir,
                reference_metadata=self.reader.metadata,
                required_names=self.config.required_parameter_names,
                nodata_fill=self.config.parameter_nodata_fill,
            ).load()

        self._validate_temporal_split()
        height = self.reader.metadata.height
        width = self.reader.metadata.width
        all_mask = torch.ones((1, height, width), dtype=torch.float32)
        total_pixels = height * width
        self.split_counts = {
            "train": total_pixels,
            "test": total_pixels,
            "val": total_pixels,
            "total": total_pixels,
        }
        self.train_dataset = self._train_dataset(all_mask)
        self.test_dataset = self._target_indexed_dataset(
            split_name="test",
            mask=all_mask,
            target_start_index=self.config.train_timesteps,
            target_end_index=self.config.train_timesteps + self.config.test_timesteps,
        )
        self.val_dataset = self._target_indexed_dataset(
            split_name="val",
            mask=all_mask,
            target_start_index=self.config.train_timesteps + self.config.test_timesteps,
            target_end_index=self.config.train_timesteps + self.config.test_timesteps + self.config.val_timesteps,
        )
        self.predict_dataset = self._target_indexed_dataset(
            split_name="predict",
            mask=all_mask,
            target_start_index=self.config.seq_length,
            target_end_index=(
                self.config.train_timesteps
                + self.config.test_timesteps
                + self.config.val_timesteps
            ),
        )
        return self

    def _validate_temporal_split(self) -> None:
        assert self.catalog is not None
        expected = self.config.train_timesteps + self.config.test_timesteps + self.config.val_timesteps
        if len(self.catalog) < expected:
            raise ValueError(
                f"Not enough timesteps ({len(self.catalog)}) for temporal split "
                f"{self.config.train_timesteps}/{self.config.test_timesteps}/{self.config.val_timesteps}."
            )
        if self.config.pred_length != 1:
            raise ValueError("pred_length must be 1 for sliding-window one-step training.")
        if self.config.train_timesteps < self.config.seq_length + self.config.pred_length:
            raise ValueError("train_timesteps must allow at least one complete training window.")

    def _train_dataset(self, mask: torch.Tensor) -> TargetIndexedRasterDataset:
        return self._target_indexed_dataset(
            split_name="train",
            mask=mask,
            target_start_index=self.config.seq_length,
            target_end_index=self.config.train_timesteps,
        )

    def _target_indexed_dataset(
        self,
        split_name: str,
        mask: torch.Tensor,
        target_start_index: int,
        target_end_index: int,
    ) -> TargetIndexedRasterDataset:
        assert self.reader is not None and self.normalizer is not None
        return TargetIndexedRasterDataset(
            reader=self.reader,
            seq_length=self.config.seq_length,
            pred_length=self.config.pred_length,
            normalizer=self.normalizer,
            split_mask=mask,
            split_name=split_name,
            parameter_reader=self.parameter_reader,
            time_step_fallback=self.config.time_step,
            target_start_index=target_start_index,
            target_end_index=target_end_index,
            require_full_history=True,
        )

    def _horizon_dataset(
        self,
        split_name: str,
        mask: torch.Tensor,
        input_start: int,
        horizon: int,
    ) -> FixedHorizonRasterDataset:
        assert self.reader is not None and self.normalizer is not None
        return FixedHorizonRasterDataset(
            reader=self.reader,
            seq_length=self.config.seq_length,
            pred_length=horizon,
            normalizer=self.normalizer,
            split_mask=mask,
            split_name=split_name,
            parameter_reader=self.parameter_reader,
            time_step_fallback=self.config.time_step,
            input_start=input_start,
        )

    def train_loader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def test_loader(self) -> Optional[DataLoader]:
        return self._loader(self.test_dataset, shuffle=False)

    def val_loader(self) -> Optional[DataLoader]:
        return self._loader(self.val_dataset, shuffle=False)

    def full_loader(self, batch_size: int = 1) -> DataLoader:
        return self._build_loader(self.predict_dataset, batch_size=batch_size, shuffle=False)

    def _loader(self, dataset: Optional[Dataset], shuffle: bool) -> DataLoader:
        if dataset is None:
            raise ValueError("Dataset split has not been initialized.")
        return self._build_loader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _build_loader(self, dataset: Optional[Dataset], batch_size: int, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise ValueError("Dataset split has not been initialized.")
        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "drop_last": False,
            "persistent_workers": self.config.persistent_workers,
        }
        if self.config.prefetch_factor is not None and self.config.num_workers > 0:
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(**kwargs)

    def clear_runtime_caches(self) -> None:
        if self.reader is not None:
            self.reader.clear_cache()

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return tensor
        return self.normalizer.denormalize(tensor)

    def summary(self) -> dict:
        assert self.catalog is not None and self.reader is not None and self.normalizer is not None
        return {
            "timesteps": self.catalog.total_timesteps,
            "input_timesteps": self.config.seq_length,
            "prediction_timesteps": self.config.pred_length,
            "train_timesteps": self.config.train_timesteps,
            "test_timesteps": self.config.test_timesteps,
            "val_timesteps": self.config.val_timesteps,
            "height": self.reader.metadata.height,
            "width": self.reader.metadata.width,
            "train_pixels": self.split_counts.get("train", 0),
            "test_pixels": self.split_counts.get("test", 0),
            "val_pixels": self.split_counts.get("val", 0),
            "total_pixels": self.split_counts.get("total", 0),
            "random_seed": self.config.random_seed,
            "normalizer": self.normalizer,
            "parameter_names": sorted(self.parameter_reader.parameters().keys()) if self.parameter_reader else [],
            "parameter_layers": self.parameter_reader.layer_count if self.parameter_reader else 0,
            "train_windows": len(self.train_dataset) if self.train_dataset is not None else 0,
            "test_windows": len(self.test_dataset) if self.test_dataset is not None else 0,
            "val_windows": len(self.val_dataset) if self.val_dataset is not None else 0,
            "first_file": self.catalog[0].filename,
            "last_file": self.catalog[-1].filename,
            "train_first_input_file": self.catalog[0].filename,
            "train_last_input_file": self.catalog[self.config.train_timesteps - self.config.pred_length - 1].filename,
            "train_first_target_file": self.catalog[self.train_dataset.target_start_index].filename,
            "train_last_target_file": self.catalog[self.config.train_timesteps - 1].filename,
            "test_first_target_file": self.catalog[self.test_dataset.target_start_index].filename,
            "test_last_target_file": self.catalog[self.test_dataset.target_end_index - 1].filename,
            "val_first_target_file": self.catalog[self.val_dataset.target_start_index].filename,
            "val_last_target_file": self.catalog[self.val_dataset.target_end_index - 1].filename,
            "first_target_file": self.catalog[self.train_dataset.target_start_index].filename,
            "last_target_file": self.catalog[self.predict_dataset.target_end_index - 1].filename,
        }
