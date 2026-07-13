from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

APPROVED_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CMU-MOCAP", "CC-BY-4.0+DUA"}


@dataclass(frozen=True)
class DatasetEntry:
    dataset_id: str
    title: str
    task: str
    license_id: str
    license_url: str
    approved_for_training: bool
    attribution_required: bool
    requires_acceptance: bool
    downloader: dict[str, Any]


class DatasetCatalog:
    def __init__(self, path: Path):
        self.path = path.resolve()
        with self.path.open(encoding="utf-8") as handle:
            self.raw = yaml.safe_load(handle)
        if self.raw.get("schema_version") != 1:
            raise ValueError("Unsupported dataset catalog schema")
        self.datasets = {
            dataset_id: DatasetEntry(
                dataset_id=dataset_id,
                title=value["title"],
                task=value["task"],
                license_id=value["license_id"],
                license_url=value["license_url"],
                approved_for_training=bool(value.get("approved_for_training", False)),
                attribution_required=bool(value.get("attribution_required", False)),
                requires_acceptance=bool(value.get("requires_acceptance", False)),
                downloader=dict(value["downloader"]),
            )
            for dataset_id, value in self.raw["datasets"].items()
        }

    def profile(self, name: str) -> list[DatasetEntry]:
        try:
            identifiers = self.raw["profiles"][name]["datasets"]
        except KeyError as error:
            choices = ", ".join(sorted(self.raw["profiles"]))
            raise ValueError(f"Unknown profile {name!r}; choose one of: {choices}") from error
        return [self.require_approved(identifier) for identifier in identifiers]

    def require_approved(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id!r} is not present in the allowlist")
        entry = self.datasets[dataset_id]
        if not entry.approved_for_training or entry.license_id not in APPROVED_LICENSES:
            raise ValueError(f"Dataset {dataset_id!r} is not approved for training")
        return entry

    def audit(self) -> list[str]:
        errors: list[str] = []
        for dataset_id, entry in self.datasets.items():
            if entry.approved_for_training and entry.license_id not in APPROVED_LICENSES:
                errors.append(f"{dataset_id}: unrecognized approved license {entry.license_id}")
            if not entry.license_url.startswith("https://"):
                errors.append(f"{dataset_id}: license URL must use HTTPS")
            if entry.task not in {"motion", "paired_video", "weak_video"}:
                errors.append(f"{dataset_id}: unsupported task {entry.task}")
        for profile_name in self.raw["profiles"]:
            try:
                self.profile(profile_name)
            except ValueError as error:
                errors.append(f"profile {profile_name}: {error}")
        return errors
