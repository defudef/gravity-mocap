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
import zlib
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


def _check_space(
    destination: Path,
    expected_bytes: int,
    allow_large: bool,
    *,
    existing_bytes: int = 0,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if expected_bytes >= LARGE_DOWNLOAD_BYTES and not allow_large:
        gib = expected_bytes / 1024**3
        raise RuntimeError(f"Refusing a {gib:.1f} GiB download without --allow-large")
    free = shutil.disk_usage(destination).free
    remaining = max(expected_bytes - max(existing_bytes, 0), 0)
    required = remaining + int(expected_bytes * 0.1)
    if expected_bytes and free < required:
        raise RuntimeError(
            f"Not enough free space: need {required} bytes for the remaining download "
            "and 10% headroom"
        )


def _verified_or_remove(path: Path, spec: dict) -> bool:
    if not path.exists():
        return False
    try:
        verify_file(path, spec)
    except ValueError as error:
        print(
            f"[download] discarding invalid {path.name} ({error}); fetching a clean copy",
            flush=True,
        )
        path.unlink()
        return False
    return True


def _streaming_get(spec: dict, headers: dict[str, str]) -> requests.Response:
    request_options = {
        "headers": headers,
        "stream": True,
        "timeout": (30, 120),
    }
    try:
        return requests.get(spec["url"], **request_options)
    except requests.exceptions.SSLError as error:
        fallback_url = spec.get("tls_fallback_url")
        if not fallback_url:
            raise
        primary = urlparse(str(spec["url"]))
        fallback = urlparse(str(fallback_url))
        if (
            primary.scheme != "https"
            or fallback.scheme != "http"
            or primary.hostname != fallback.hostname
            or primary.path != fallback.path
            or primary.query != fallback.query
            or not spec.get("sha256")
        ):
            raise RuntimeError(
                "TLS fallback requires the same host/path and a pinned SHA-256"
            ) from error
        print(
            f"[download] TLS chain failed for {primary.hostname}; using the same-host "
            "checksum-pinned HTTP endpoint",
            flush=True,
        )
        return requests.get(str(fallback_url), **request_options)


def _download_http(spec: dict, destination: Path, allow_large: bool) -> list[Path]:
    output = destination / spec["filename"]
    if _verified_or_remove(output, spec):
        return [output]
    expected = int(spec.get("expected_bytes", 0))
    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists() and expected and partial.stat().st_size > expected:
        print(f"[download] discarding oversized {partial.name}", flush=True)
        partial.unlink()
    if partial.exists() and expected and partial.stat().st_size == expected:
        if _verified_or_remove(partial, spec):
            os.replace(partial, output)
            return [output]
    existing_bytes = partial.stat().st_size if partial.exists() else 0
    _check_space(destination, expected, allow_large, existing_bytes=existing_bytes)

    for attempt in range(2):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        action = "Resuming" if offset else "Downloading"
        total_gib = f"{expected / 1024**3:.2f} GiB" if expected else "unknown size"
        print(
            f"{action} {output.name} at {offset / 1024**2:.1f} MiB / {total_gib}",
            flush=True,
        )
        started = time.monotonic()
        last_report = started
        downloaded = offset
        with _streaming_get(spec, headers) as response:
            response.raise_for_status()
            if offset and response.status_code != 206:
                offset = 0
                downloaded = 0
                started = time.monotonic()
            elif offset:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(
                        f"Server returned an invalid Content-Range for {output.name}: "
                        f"{content_range or 'missing'}"
                    )
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
        print(f"[download] verifying {output.name}", flush=True)
        try:
            verify_file(partial, spec)
        except ValueError:
            partial.unlink(missing_ok=True)
            if attempt == 0:
                print(
                    f"[download] {output.name} failed verification; retrying once from byte 0",
                    flush=True,
                )
                continue
            raise
        os.replace(partial, output)
        return [output]
    raise AssertionError("unreachable download retry state")


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
        if not _verified_or_remove(output, file_spec):
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


def _verified_zip_member(path: Path, expected_bytes: int, expected_crc: int) -> bool:
    if not path.exists() or path.stat().st_size != expected_bytes:
        return False
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF == expected_crc


def _select_remote_zip_members(member_infos: list[object], spec: dict) -> list[object]:
    maximum = int(spec.get("max_core_members", len(member_infos)))
    group_regex = spec.get("selection_group_regex")
    if not group_regex:
        return sorted(
            member_infos,
            key=lambda item: (item.file_size, item.filename),
        )[:maximum]
    pattern = re.compile(str(group_regex))
    groups: dict[str, list[object]] = {}
    for item in member_infos:
        match = pattern.search(item.filename)
        if match is None:
            raise ValueError(
                f"Remote ZIP member does not match selection_group_regex: {item.filename}"
            )
        group = match.group(1) if match.groups() else match.group(0)
        groups.setdefault(group, []).append(item)
    for items in groups.values():
        items.sort(key=lambda item: (item.file_size, item.filename))
    budget = int(spec.get("max_core_bytes", 0))
    selected: list[object] = []
    selected_bytes = 0
    positions = {group: 0 for group in groups}
    while len(selected) < maximum:
        progressed = False
        for group in sorted(groups):
            position = positions[group]
            items = groups[group]
            if position >= len(items):
                continue
            candidate = items[position]
            positions[group] += 1
            if budget and selected_bytes + candidate.file_size > budget:
                continue
            selected.append(candidate)
            selected_bytes += candidate.file_size
            progressed = True
            if len(selected) >= maximum:
                break
        if not progressed:
            break
    return selected


def _download_remote_zip(spec: dict, destination: Path, profile: str) -> list[Path]:
    from remotezip import RemoteZip

    with RemoteZip(spec["url"]) as archive:
        if profile == "smoke":
            member_infos = [archive.getinfo(name) for name in spec["smoke_members"]]
        else:
            pattern = re.compile(spec["member_regex"])
            matching = [
                item
                for item in archive.infolist()
                if item.file_size and pattern.search(item.filename)
            ]
            member_infos = _select_remote_zip_members(matching, spec)
        if not member_infos:
            raise RuntimeError("No remote ZIP members matched the configured selection")
        if any(
            Path(item.filename).is_absolute() or ".." in Path(item.filename).parts
            for item in member_infos
        ):
            raise RuntimeError("Remote ZIP contains an unsafe member path")
        outputs = [destination / item.filename for item in member_infos]
        missing: list[tuple[object, Path]] = []
        for item, output in zip(member_infos, outputs, strict=True):
            if _verified_zip_member(output, item.file_size, item.CRC):
                print(f"[download] verified existing ZIP member {item.filename}", flush=True)
                continue
            if output.exists():
                print(f"[download] discarding incomplete ZIP member {item.filename}", flush=True)
                output.unlink()
            partial = output.with_suffix(output.suffix + ".part")
            if partial.exists():
                print(f"[download] restarting partial ZIP member {item.filename}", flush=True)
                partial.unlink()
            missing.append((item, output))

        _check_space(
            destination,
            sum(item.file_size for item, _output in missing),
            allow_large=True,
        )
        for index, (item, output) in enumerate(missing, start=1):
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_suffix(output.suffix + ".part")
            print(
                f"[download] ZIP member {index}/{len(missing)}: {item.filename} "
                f"({item.file_size / 1024**3:.2f} GiB)",
                flush=True,
            )
            started = time.monotonic()
            last_report = started
            downloaded = 0
            checksum = 0
            with archive.open(item) as source, partial.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
                    downloaded += len(chunk)
                    checksum = zlib.crc32(chunk, checksum)
                    now = time.monotonic()
                    if now - last_report >= 5:
                        elapsed = max(now - started, 0.001)
                        print(
                            f"[download] {item.filename}: "
                            f"{downloaded / item.file_size * 100:.1f}% at "
                            f"{downloaded / elapsed / 1024**2:.1f} MiB/s",
                            flush=True,
                        )
                        last_report = now
            if downloaded != item.file_size or checksum & 0xFFFFFFFF != item.CRC:
                partial.unlink(missing_ok=True)
                raise ValueError(f"Remote ZIP member failed verification: {item.filename}")
            os.replace(partial, output)
            print(f"[download] verified {item.filename}", flush=True)
    return outputs


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
        qualifier = (
            "configured"
            if profile == "smoke"
            else "study-stratified" if spec.get("selection_group_regex") else "smallest matching"
        )
        budget = ""
        if profile != "smoke" and spec.get("max_core_bytes"):
            budget = f" under {int(spec['max_core_bytes']) / 1024**3:.1f} GiB"
        return (
            f"select up to {selection} {qualifier} member(s){budget} remotely from a "
            f"{archive_gib:.1f} GiB ZIP"
        )
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
