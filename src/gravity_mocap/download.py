from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import requests

from .catalog import DatasetEntry

LARGE_DOWNLOAD_BYTES = 20 * 1024**3


def digest(path: Path, algorithm: str) -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_file(path: Path, spec: dict) -> None:
    if expected := spec.get("expected_bytes"):
        if path.stat().st_size != int(expected):
            raise ValueError(f"Wrong size for {path}: {path.stat().st_size}, expected {expected}")
    for algorithm in ("sha256", "md5"):
        if expected := spec.get(algorithm):
            actual = digest(path, algorithm)
            if actual.lower() != str(expected).lower():
                raise ValueError(f"Wrong {algorithm} for {path}: {actual}")


def _check_space(destination: Path, expected_bytes: int, allow_large: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if expected_bytes >= LARGE_DOWNLOAD_BYTES and not allow_large:
        gib = expected_bytes / 1024**3
        raise RuntimeError(f"Refusing a {gib:.1f} GiB download without --allow-large")
    free = shutil.disk_usage(destination).free
    if expected_bytes and free < expected_bytes * 1.1:
        raise RuntimeError(f"Not enough free space: need 10% headroom above {expected_bytes} bytes")


def _download_http(spec: dict, destination: Path, allow_large: bool) -> list[Path]:
    output = destination / spec["filename"]
    if output.exists():
        verify_file(output, spec)
        return [output]
    expected = int(spec.get("expected_bytes", 0))
    _check_space(destination, expected, allow_large)
    partial = output.with_suffix(output.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(spec["url"], headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(partial, output)
    verify_file(output, spec)
    return [output]


def _download_huggingface(spec: dict, destination: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    output = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        repo_type="dataset",
        local_dir=destination,
    )
    return [Path(output)]


def _download_gdrive_folder(spec: dict, destination: Path) -> list[Path]:
    import gdown

    outputs = gdown.download_folder(url=spec["folder_url"], output=str(destination), quiet=False)
    if not outputs:
        raise RuntimeError("Google Drive returned no files")
    return [Path(path) for path in outputs]


def _download_remote_zip(spec: dict, destination: Path, profile: str) -> list[Path]:
    from remotezip import RemoteZip

    with RemoteZip(spec["url"]) as archive:
        if profile == "smoke":
            members = list(spec["smoke_members"])
        else:
            pattern = re.compile(spec["member_regex"])
            members = [
                item.filename
                for item in archive.infolist()
                if item.file_size and pattern.search(item.filename)
            ]
            members = members[: int(spec.get("max_core_members", len(members)))]
        if not members:
            raise RuntimeError("No remote ZIP members matched the configured selection")
        if any(Path(member).is_absolute() or ".." in Path(member).parts for member in members):
            raise RuntimeError("Remote ZIP contains an unsafe member path")
        total = sum(archive.getinfo(name).file_size for name in members)
        _check_space(destination, total, allow_large=True)
        archive.extractall(destination, members)
    return [destination / member for member in members]


def describe(entry: DatasetEntry, profile: str) -> str:
    spec = entry.downloader
    kind = spec["type"]
    if kind == "remote_zip_members":
        selection = (
            len(spec["smoke_members"])
            if profile == "smoke"
            else spec.get("max_core_members", "all")
        )
        archive_gib = spec["archive_bytes"] / 1024**3
        return f"select {selection} member(s) remotely from a {archive_gib:.1f} GiB ZIP"
    if kind == "dryad_manual":
        files = spec.get("files", [spec])
        expected_gib = sum(int(file.get("expected_bytes", 0)) for file in files) / 1024**3
        return (
            f"{len(files)} manual public browser download(s) "
            f"({expected_gib:.1f} GiB total), then checksum verification"
        )
    if expected := spec.get("expected_bytes"):
        return f"{kind}, approximately {expected / 1024**3:.1f} GiB"
    return kind


def download_dataset(
    entry: DatasetEntry,
    root: Path,
    profile: str,
    *,
    allow_large: bool,
    accepted_data_use: bool,
) -> list[Path]:
    if entry.requires_acceptance and not accepted_data_use:
        raise RuntimeError(
            f"{entry.dataset_id} requires --accept-data-use after reading {entry.license_url}"
        )
    destination = root / entry.dataset_id
    destination.mkdir(parents=True, exist_ok=True)
    spec = entry.downloader
    kind = spec["type"]
    if kind == "http":
        outputs = _download_http(spec, destination, allow_large)
    elif kind == "huggingface":
        _check_space(destination, int(spec.get("expected_bytes", 0)), allow_large)
        outputs = _download_huggingface(spec, destination)
    elif kind == "gdrive_folder":
        if not allow_large:
            raise RuntimeError(
                "Google Drive folder size is not declared; inspect it and pass --allow-large"
            )
        outputs = _download_gdrive_folder(spec, destination)
    elif kind == "remote_zip_members":
        outputs = _download_remote_zip(spec, destination, profile)
    elif kind == "dryad_manual":
        outputs = []
        for file_spec in spec.get("files", [spec]):
            output = destination / file_spec["filename"]
            if not output.exists():
                raise RuntimeError(
                    "Dryad blocks anonymous scripted downloads. "
                    f"Download {file_spec['filename']} in a browser from "
                    f"{spec['browser_url']} and place it at {output}; no account is required."
                )
            verify_file(output, file_spec)
            outputs.append(output)
    else:
        raise ValueError(f"Unsupported downloader type: {kind}")
    metadata = {
        "source_id": entry.dataset_id,
        "title": entry.title,
        "license_id": entry.license_id,
        "license_url": entry.license_url,
        "attribution_required": entry.attribution_required,
        "requires_acceptance": entry.requires_acceptance,
        "files": [str(path.relative_to(destination)) for path in outputs],
    }
    (destination / "SOURCE.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return outputs
