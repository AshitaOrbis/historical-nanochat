"""The README must state the runnable boundary and the temporal controls honestly."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_readme_names_external_prerequisites_and_offline_boundary():
    readme = " ".join((REPO / "README.md").read_text().split())
    for required_text in (
        "Network and external-data prerequisites",
        "python3 -m venv .venv",
        "manu/project_gutenberg",
        "Library of Congress API",
        "CLARIN-D",
        "case.law",
        "cannot acquire a corpus offline",
        "does not include the historical corpus",
    ):
        assert required_text in readme


def test_readme_project_tree_does_not_advertise_absent_root_packages():
    readme = (REPO / "README.md").read_text()
    assert "├── notebooks/" not in readme
    assert "└── scripts/" not in readme


def test_readme_names_controls_and_residual_contamination_paths():
    readme = " ".join((REPO / "README.md").read_text().lower().split())
    for control in (
        "publication/issue metadata",
        "zero-record acquisitions",
        "post-cutoff years",
        "checker version and sha-256",
        "unchecked",
    ):
        assert control in readme

    for residual_path in (
        "later editorial annotations",
        "reprints",
        "ocr metadata",
        "semantic anachronisms",
        "memorized overlap",
        "incorrect or missing source metadata",
    ):
        assert residual_path in readme
