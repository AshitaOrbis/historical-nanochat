"""Public research claims must not overstate temporal filtering as a guarantee."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_readme_frames_temporal_ignorance_as_measured_objective():
    readme = (REPO / "README.md").read_text().lower()
    for absolute_claim in (
        "trained exclusively on texts",
        "genuinely don't know",
        "ensuring genuine temporal ignorance",
        "what it doesn't know",
        "authentic historical worldview",
    ):
        assert absolute_claim not in readme

    assert "not to guarantee ignorance" in readme
    assert "residual contamination" in readme
    assert "must be measured" in readme
