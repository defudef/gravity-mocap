from pathlib import Path

from gravity_mocap.catalog import DatasetCatalog
from gravity_mocap.data import audit_training_data
from gravity_mocap.fixture import create_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_is_closed_and_auditable() -> None:
    catalog = DatasetCatalog(ROOT / "configs/datasets.yaml")
    assert catalog.audit() == []
    assert {entry.dataset_id for entry in catalog.profile("smoke")} == {
        "cmu_mocap",
        "addbiomechanics",
    }
    assert catalog.datasets["sam"].task == "motion"
    assert catalog.datasets["tum_preha"].task == "motion"
    assert catalog.datasets["mri"].task == "paired_video"
    assert catalog.datasets["mri"].downloader["type"] == "dryad_browser"
    assert {item["filename"] for item in catalog.datasets["mri"].downloader["files"]} == {
        "dataset_release.zip",
        "blurred_videos.zip",
    }


def test_unknown_dataset_fails_closed() -> None:
    catalog = DatasetCatalog(ROOT / "configs/datasets.yaml")
    try:
        catalog.require_approved("mystery_motion_dump")
    except ValueError as error:
        assert "not present in the allowlist" in str(error)
    else:
        raise AssertionError("Unknown source should fail closed")


def test_synthetic_data_is_rejected_by_production_gate(tmp_path: Path) -> None:
    create_fixture(tmp_path / "synthetic.npz", frames=8, image_feature_dim=32)
    catalog = DatasetCatalog(ROOT / "configs/datasets.yaml")
    errors, _ = audit_training_data(tmp_path, catalog, allow_synthetic=False)
    assert errors
    assert "not present in the allowlist" in errors[0]
