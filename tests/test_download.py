import hashlib
import io
import json
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from gravity_mocap import download


def test_space_check_counts_only_missing_bytes_for_resumed_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _path: SimpleNamespace(free=300))

    download._check_space(tmp_path, 1000, allow_large=False, existing_bytes=800)
    with pytest.raises(RuntimeError, match="remaining download"):
        download._check_space(tmp_path, 1000, allow_large=False, existing_bytes=799)


def test_extract_dryad_signed_url_accepts_only_expected_asset_host() -> None:
    signed = (
        "https://dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com/object.zip"
        "?response-content-disposition=attachment%3B%20filename%3Dobject.zip"
        "&X-Amz-Signature=test"
    )
    output = json.dumps({"result": f"2. [GET] {signed} => [200] OK"})

    assert download._extract_dryad_signed_url(output, "object.zip") == signed
    assert download._extract_dryad_signed_url(output, "different.zip") is None
    assert (
        download._extract_dryad_signed_url(
            json.dumps(
                {
                    "result": "1. [GET] https://example.com/object.zip"
                    "?X-Amz-Signature=test => [200] OK"
                }
            )
        )
        is None
    )


def test_http_download_resumes_partial_file_and_verifies_checksum(tmp_path: Path) -> None:
    payload = b"gravity-mocap-dryad-resume" * 4096
    requests_seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            requests_seen.append(range_header)
            start = (
                int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            )
            body = payload[start:]
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(body)))
            if range_header:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "downloads"
    destination.mkdir()
    partial = destination / "fixture.zip.part"
    partial.write_bytes(payload[:1234])
    spec = {
        "url": f"http://127.0.0.1:{server.server_port}/fixture.zip",
        "filename": "fixture.zip",
        "expected_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        outputs = download._download_http(spec, destination, allow_large=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert requests_seen == ["bytes=1234-"]
    assert outputs == [destination / "fixture.zip"]
    assert outputs[0].read_bytes() == payload
    assert not partial.exists()


def test_http_download_uses_same_host_checksum_pinned_fallback_on_tls_error(
    tmp_path: Path,
) -> None:
    payload = b"checksum-pinned-fallback" * 256

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "downloads"
    destination.mkdir()
    path = f"127.0.0.1:{server.server_port}/fixture.zip"
    spec = {
        "url": f"https://{path}",
        "tls_fallback_url": f"http://{path}",
        "filename": "fixture.zip",
        "expected_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        outputs = download._download_http(spec, destination, allow_large=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert outputs[0].read_bytes() == payload


def test_http_download_retries_from_zero_when_resumed_partial_is_corrupt(tmp_path: Path) -> None:
    payload = b"clean-download" * 4096
    requests_seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            requests_seen.append(range_header)
            start = (
                int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            )
            body = payload[start:]
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(body)))
            if range_header:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "downloads"
    destination.mkdir()
    partial = destination / "fixture.zip.part"
    partial.write_bytes(b"x" * 1234)
    spec = {
        "url": f"http://127.0.0.1:{server.server_port}/fixture.zip",
        "filename": "fixture.zip",
        "expected_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        outputs = download._download_http(spec, destination, allow_large=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert requests_seen == ["bytes=1234-", None]
    assert outputs[0].read_bytes() == payload
    assert not partial.exists()


def test_dryad_browser_cancels_browser_download_and_returns_matching_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signed = (
        "https://dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com/file.zip"
        "?response-content-disposition=attachment%3B%20filename%3Dfile.zip"
        "&X-Amz-Signature=test"
    )
    calls: list[list[str]] = []

    def fake_cli(
        arguments: list[str],
        *,
        cwd: Path,
        timeout: float = 120,
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout, allow_failure
        calls.append(arguments)
        stdout = (
            json.dumps({"result": f"2. [GET] {signed} => [200] OK"})
            if "--json" in arguments
            else ""
        )
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    monkeypatch.setattr(download, "_run_playwright_cli", fake_cli)

    with download._DryadBrowser("https://datadryad.org/dataset/example") as browser:
        assert browser.signed_url(123, "file.zip", timeout=1) == signed

    flattened = [argument for call in calls for argument in call]
    assert "--headed" in flattened
    assert "run-code" in flattened
    assert "requests" in flattened
    assert "close" in flattened


def test_dryad_download_skips_browser_when_verified_files_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"already downloaded"
    output = tmp_path / "dataset_release.zip"
    output.write_bytes(payload)
    spec = {
        "browser_url": "https://datadryad.org/dataset/example",
        "files": [
            {
                "file_id": 1,
                "filename": output.name,
                "expected_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }

    class BrowserMustNotStart:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("verified downloads must not open a browser")

    monkeypatch.setattr(download, "_DryadBrowser", BrowserMustNotStart)

    assert download._download_dryad(spec, tmp_path, allow_large=False) == [output]


def test_dryad_download_replaces_corrupt_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"replacement"
    output = tmp_path / "dataset_release.zip"
    output.write_bytes(b"corrupt")
    spec = {
        "browser_url": "https://datadryad.org/dataset/example",
        "files": [
            {
                "file_id": 1,
                "filename": output.name,
                "expected_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    browser_calls: list[tuple[int, str]] = []

    class FakeBrowser:
        def __init__(self, *_args: object) -> None:
            return

        def __enter__(self) -> "FakeBrowser":
            return self

        def signed_url(self, file_id: int, filename: str) -> str:
            browser_calls.append((file_id, filename))
            return "https://example.invalid/replacement"

        def __exit__(self, *_args: object) -> None:
            return

    def fake_download_http(file_spec: dict, destination: Path, allow_large: bool) -> list[Path]:
        del allow_large
        replacement = destination / file_spec["filename"]
        replacement.write_bytes(payload)
        download.verify_file(replacement, file_spec)
        return [replacement]

    monkeypatch.setattr(download, "_DryadBrowser", FakeBrowser)
    monkeypatch.setattr(download, "_download_http", fake_download_http)

    assert download._download_dryad(spec, tmp_path, allow_large=False) == [output]
    assert output.read_bytes() == payload
    assert browser_calls == [(1, output.name)]


def test_remote_zip_members_replace_partial_outputs_atomically(tmp_path: Path) -> None:
    members = {
        "data/first.b3d": b"first-member" * 1024,
        "data/second.b3d": b"second-member" * 2048,
        "data/too-large.b3d": b"large-member" * 4096,
    }
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    archive_payload = archive_buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(archive_payload)))
            self.end_headers()

        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            if range_header:
                start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else len(archive_payload) - 1
                else:
                    suffix_length = int(end_text)
                    start = max(0, len(archive_payload) - suffix_length)
                    end = len(archive_payload) - 1
                body = archive_payload[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(archive_payload)}")
            else:
                body = archive_payload
                self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "downloads"
    incomplete = destination / "data/first.b3d"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"not-a-complete-member")
    stale_partial = destination / "data/second.b3d.part"
    stale_partial.write_bytes(b"partial")
    spec = {
        "url": f"http://127.0.0.1:{server.server_port}/archive.zip",
        "archive_bytes": len(archive_payload),
        "member_regex": r"^data/.+\.b3d$",
        "smoke_members": ["data/first.b3d"],
        "max_core_members": 2,
    }
    try:
        outputs = download._download_remote_zip(spec, destination, "core")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    selected = {"data/first.b3d", "data/second.b3d"}
    assert {path.relative_to(destination).as_posix() for path in outputs} == selected
    for name in selected:
        payload = members[name]
        assert (destination / name).read_bytes() == payload
        assert not (destination / f"{name}.part").exists()
    assert not (destination / "data/too-large.b3d").exists()


def test_remote_zip_core_selection_is_group_stratified() -> None:
    members = [
        SimpleNamespace(filename="train/A/a-small.b3d", file_size=10),
        SimpleNamespace(filename="train/A/a-large.b3d", file_size=20),
        SimpleNamespace(filename="train/B/b-small.b3d", file_size=11),
        SimpleNamespace(filename="train/B/b-large.b3d", file_size=21),
        SimpleNamespace(filename="train/C/c-small.b3d", file_size=12),
    ]
    selected = download._select_remote_zip_members(
        members,
        {
            "selection_group_regex": r"^train/([^/]+)/",
            "max_core_members": 4,
            "max_core_bytes": 60,
        },
    )

    assert [item.filename for item in selected] == [
        "train/A/a-small.b3d",
        "train/B/b-small.b3d",
        "train/C/c-small.b3d",
        "train/A/a-large.b3d",
    ]
