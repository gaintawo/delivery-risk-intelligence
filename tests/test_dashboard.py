"""Smoke test: every dashboard page renders without an exception."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "dashboard" / "data" / "test_scored.csv.gz").exists(),
                                reason="dashboard data not exported")


def test_app_entry_point_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", ["page_overview", "page_drivers", "page_explorer", "page_fairness",
                                  "page_transparency"])
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    def script(root: str, name: str):
        import sys
        sys.path.insert(0, root)
        from dashboard import views
        getattr(views, name)()

    at = AppTest.from_function(script, args=(str(ROOT), page), default_timeout=120).run()
    assert not at.exception, at.exception
