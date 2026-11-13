# Dashboard

Streamlit stakeholder dashboard (Sprint 5, Module 5). Entry point: `dashboard/app.py`; pages in `dashboard/views.py`.

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

**Deploy on Streamlit Community Cloud:** share.streamlit.io → Create app → repository `gaintawo/delivery-risk-intelligence`,
branch `main`, main file path `dashboard/app.py` → Advanced settings: Python 3.13 → Deploy. No secrets are needed.

The model is in shadow mode: the dashboard is for review by stakeholders and the Responsible AI Review Board, not for daily operations.
