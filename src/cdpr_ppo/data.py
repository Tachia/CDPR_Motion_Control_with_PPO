from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ConfigurationData:
    ee_pose: np.ndarray
    cable_lengths: np.ndarray


class CDPRDataset:
    """Load CDPR data and split into 4/3/2-cable configurations.

    Supports CSV and Excel sources.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.raw_data: pd.DataFrame | None = None
        self.configurations: Dict[str, ConfigurationData | None] = {
            "4-cable": None,
            "3-cable": None,
            "2-cable": None,
        }
        self._load()
        self._process_all_configurations()

    def _load(self) -> None:
        suffix = self.file_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            self.raw_data = pd.read_excel(self.file_path)
        else:
            self.raw_data = pd.read_csv(
                self.file_path,
                delimiter=";",
                encoding="latin1",
                decimal=",",
                na_values=["", " ", "NA", "NaN"],
                keep_default_na=True,
            )

    def _process_all_configurations(self) -> None:
        if self.raw_data is None:
            raise ValueError("Dataset not loaded")

        indices = {
            "4-cable": {"start": None, "end": None},
            "3-cable": {"start": None, "end": None},
            "2-cable": {"start": None, "end": None},
        }

        for idx, row in self.raw_data.iterrows():
            first = row.iloc[0]
            if isinstance(first, str):
                low = first.lower()
                if "4-cable" in low:
                    indices["4-cable"]["start"] = idx + 1
                elif "3-cable" in low:
                    indices["4-cable"]["end"] = idx
                    indices["3-cable"]["start"] = idx + 1
                elif "2-cable" in low:
                    indices["3-cable"]["end"] = idx
                    indices["2-cable"]["start"] = idx + 1

        indices["2-cable"]["end"] = len(self.raw_data)

        for config in ("4-cable", "3-cable", "2-cable"):
            start_idx = indices[config]["start"]
            end_idx = indices[config]["end"]
            if start_idx is None or end_idx is None:
                continue

            data_section = self.raw_data.iloc[start_idx:end_idx].copy()
            numeric = data_section.apply(pd.to_numeric, errors="coerce")

            ee_pose = numeric.iloc[:, 3:9].to_numpy()
            cable_lengths = numeric.iloc[:, 10:14].to_numpy()
            mask = ~(np.isnan(ee_pose).any(axis=1) | np.isnan(cable_lengths).any(axis=1))

            if mask.any():
                self.configurations[config] = ConfigurationData(
                    ee_pose=ee_pose[mask],
                    cable_lengths=cable_lengths[mask],
                )

    def get_configuration(self, config: str) -> ConfigurationData:
        data = self.configurations.get(config)
        if data is None:
            raise ValueError(f"No valid data found for {config} in {self.file_path}")
        return data
