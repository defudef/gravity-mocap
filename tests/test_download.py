import hashlib
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gravity_mocap import download


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
