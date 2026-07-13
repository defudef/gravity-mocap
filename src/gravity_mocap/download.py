from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .catalog import DatasetEntry

LARGE_DOWNLOAD_BYTES = 20 * 1024**3
PLAYWRIGHT_CLI_PACKAGE = "@playwright/cli@0.1.17"
DRYAD_DOWNLOAD_BASE = "https://datadryad.org/downloads/file_stream"
DRYAD_ASSET_HOST_PREFIX = "dryad-assetstore-"


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
    if partial.exists() and expected and partial.stat().st_size == expected:
        verify_file(partial, spec)
        os.replace(partial, output)
        return [output]
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    action = "Resuming" if offset else "Downloading"
    total_gib = f"{expected / 1024**3:.2f} GiB" if expected else "unknown size"
    print(f"{action} {output.name} at {offset / 1024**2:.1f} MiB / {total_gib}", flush=True)
    started = time.monotonic()
    last_report = started
    downloaded = offset
    with requests.get(spec["url"], headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
            downloaded = 0
            started = time.monotonic()
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 5:
                        elapsed = max(now - started, 0.001)
                        speed = max(downloaded - offset, 0) / elapsed / 1024**2
                        percent = f"{downloaded / expected * 100:.1f}%" if expected else "?"
                        print(
                            f"[download] {output.name}: {downloaded / 1024**3:.2f} GiB "
                            f"({percent}) at {speed:.1f} MiB/s",
                            flush=True,
                        )
                        last_report = now
    os.replace(partial, output)
    print(f"[download] verifying {output.name}", flush=True)
    verify_file(output, spec)
    return [output]


def _sanitized_process_error(result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "unknown error").strip().splitlines()[-10:]
    message = " | ".join(detail) if detail else "unknown error"
    return re.sub(r"https://[^\s]+", "<url>", message)[:500]


def _run_playwright_cli(
    arguments: list[str],
    *,
    cwd: Path,
    timeout: float = 120,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError(
            "Automated Dryad downloads require Node.js 18+ with npx. "
            "Install Node.js, then retry the same command."
        )
    command = [
        npx,
        "--yes",
        "--package",
        PLAYWRIGHT_CLI_PACKAGE,
        "playwright-cli",
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Playwright CLI timed out after {int(timeout)} seconds") from error
    if result.returncode and not allow_failure:
        raise RuntimeError(f"Playwright CLI failed: {_sanitized_process_error(result)}")
    return result


def _extract_dryad_signed_url(output: str, expected_filename: str | None = None) -> str | None:
    try:
        decoded = json.loads(output)
        text = str(decoded.get("result", "")) if isinstance(decoded, dict) else output
    except json.JSONDecodeError:
        text = output
    for candidate in re.findall(r"\[GET\]\s+(https://\S+)\s+=>", text):
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").lower()
        disposition = unquote(parse_qs(parsed.query).get("response-content-disposition", [""])[0])
        if (
            parsed.scheme == "https"
            and hostname.startswith(DRYAD_ASSET_HOST_PREFIX)
            and hostname.endswith(".amazonaws.com")
            and "X-Amz-Signature=" in candidate
            and (expected_filename is None or f"filename={expected_filename}" in disposition)
        ):
            return candidate
    return None


class _DryadBrowser:
    def __init__(self, browser_url: str) -> None:
        if not browser_url.startswith("https://datadryad.org/"):
            raise ValueError("Dryad browser URL must use https://datadryad.org/")
        self.browser_url = browser_url
        self.session = f"gravity-mocap-dryad-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self.cwd: Path | None = None

    def __enter__(self) -> _DryadBrowser:
        self.temporary = tempfile.TemporaryDirectory(prefix="gravity-mocap-dryad-")
        self.cwd = Path(self.temporary.name)
        try:
            try:
                self._run(["open", self.browser_url, "--headed"])
            except RuntimeError as error:
                if (
                    "install-browser" not in str(error).lower()
                    and "executable" not in str(error).lower()
                ):
                    raise
                _run_playwright_cli(["install-browser", "chromium"], cwd=self.cwd, timeout=600)
                self._run(["open", self.browser_url, "--headed"])
            self._run(
                [
                    "run-code",
                    "(page) => { page.on('download', download => { "
                    "download.cancel().catch(() => {}); }); }",
                ]
            )
        except BaseException:
            self.close()
            raise
        return self

    def _run(
        self,
        arguments: list[str],
        *,
        timeout: float = 120,
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if self.cwd is None:
            raise RuntimeError("Dryad browser session is not open")
        return _run_playwright_cli(
            ["--session", self.session, *arguments],
            cwd=self.cwd,
            timeout=timeout,
            allow_failure=allow_failure,
        )

    def signed_url(self, file_id: int, expected_filename: str, *, timeout: float = 120) -> str:
        if file_id <= 0:
            raise ValueError("Dryad file ID must be positive")
        self._run(["requests", "--clear"], allow_failure=True)
        self._run(
            ["goto", f"{DRYAD_DOWNLOAD_BASE}/{file_id}"],
            allow_failure=True,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._run(
                ["--json", "requests", "--static", "--filter", "amazonaws.com"],
                timeout=30,
                allow_failure=True,
            )
            if signed_url := _extract_dryad_signed_url(result.stdout, expected_filename):
                return signed_url
            time.sleep(1)
        raise RuntimeError(
            f"Dryad browser challenge did not produce a download URL within {int(timeout)} seconds"
        )

    def close(self) -> None:
        try:
            if self.cwd is not None:
                self._run(["close"], timeout=30, allow_failure=True)
        finally:
            self.cwd = None
            if self.temporary is not None:
                self.temporary.cleanup()
                self.temporary = None

    def __exit__(self, *_args: object) -> None:
        self.close()


def _download_dryad(spec: dict, destination: Path, allow_large: bool) -> list[Path]:
    file_specs = list(spec.get("files", [spec]))
    outputs = [destination / file_spec["filename"] for file_spec in file_specs]
    missing: list[dict] = []
    for file_spec, output in zip(file_specs, outputs, strict=True):
        if output.exists():
            verify_file(output, file_spec)
        else:
            missing.append(file_spec)
    if not missing:
        return outputs

    print(
        "Opening a temporary Chrome window to pass Dryad's public browser challenge; "
        "no account or interaction is required.",
        flush=True,
    )
    with _DryadBrowser(str(spec["browser_url"])) as browser:
        for file_spec in missing:
            signed_url = browser.signed_url(int(file_spec["file_id"]), str(file_spec["filename"]))
            print(f"Dryad authorized {file_spec['filename']}; starting resumable transfer.")
            _download_http({**file_spec, "url": signed_url}, destination, allow_large)
    return outputs


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
    if kind == "dryad_browser":
        files = spec.get("files", [spec])
        expected_gib = sum(int(file.get("expected_bytes", 0)) for file in files) / 1024**3
        return (
            f"{len(files)} public browser-assisted resumable download(s) "
            f"({expected_gib:.1f} GiB total), then checksum verification; no account"
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
    elif kind == "dryad_browser":
        outputs = _download_dryad(spec, destination, allow_large)
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
