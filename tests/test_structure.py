"""Sprint S0 checks: the repository matches the Module 1 structure and the
data dictionary keeps outcome columns out of the feature set (FR2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "CHANGELOG.md", "requirements.txt",
    "docs/model_cards", "docs/data_dictionary.md", "docs/RAID_LOG.md",
    "docs/technical_architecture.md", "docs/backlog.csv",
    "data/raw", "data/processed", "notebooks",
    "src/pipeline", "src/features", "src/models", "src/fairness", "src/actions",
    "api", "dashboard", "tests",
]

OUTCOME_COLUMNS = {"order_delivered_customer_date", "order_status", "review_score"}


def test_module1_structure_exists():
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    assert not missing, f"Missing from Module 1 structure: {missing}"


def test_outcome_columns_tagged_as_outcomes():
    text = (ROOT / "docs" / "data_dictionary.md").read_text(encoding="utf-8")
    for col in OUTCOME_COLUMNS:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {col} |"))
        assert "| O |" in line, f"{col} must be tagged O (outcome only)"
