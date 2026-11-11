# Delivery Risk Intelligence (DRI)
## RAID Log – Module 2

Technical RAID log for Sprint 1 (Project Setup). The same 24 entries, with editable scores and filters, are in [`raid_log.xlsx`](raid_log.xlsx). Risk score = likelihood × impact on a 1–3 scale (6–9 High, 3–5 Medium, 1–2 Low).

### 1. Risks

| ID | Risk | L | I | Score | Mitigation | Owner | Sprint | Links to Module 1 |
|---|---|---|---|---|---|---|---|---|
| R01 | Missing, inconsistent or out-of-sequence delivery dates distort delay calculations | 3 | 2 | 6 High | Validate records with Great Expectations; quarantine and document exclusions | Lead analyst | Sprints 2-3 | Module 1 risk register |
| R02 | Data leakage during model training (post-purchase fields used as features) | 2 | 3 | 6 High | Use only features available at prediction time; P/H/O tags in data dictionary; automated test fails the build | Lead analyst | Sprints 2-3 | Module 1 risk register; FR2 |
| R03 | Historical Olist data may not reflect current logistics conditions | 3 | 2 | 6 High | Time-based validation; PSI drift monitoring (alert above 0.2); document limitations and avoid unsupported generalisation | Lead analyst | Sprint 4 | Module 1 risk register |
| R04 | Model performance may vary across regions (recall gap above 10 points) | 2 | 3 | 6 High | Evaluate regional fairness using Fairlearn for every model version; release blocked if gap exceeds 10 points | Review Board | Sprint 4 | ER1; G5 |
| R05 | Limited project time and computing resources | 2 | 2 | 4 Medium | Prioritise essential deliverables; minimum working version first; local Docker fallback for cloud quotas | Product owner | All | Module 1 constraints |
| R06 | Class imbalance: late orders are only about 7-8% of data | 3 | 1 | 3 Medium | Class weighting; judge on PR-AUC and recall at daily capacity, not accuracy | Lead analyst | Sprint 4 | G1 |
| R07 | Postcode features act as a proxy for income | 2 | 3 | 6 High | Train with and without postcode features; record trade-off in model card | Review Board | Sprint 4 | ER2 |
| R08 | Geolocation quality: several coordinates per postcode prefix, some prefixes missing | 2 | 2 | 4 Medium | Median point per prefix; fall back to state centroid; range checks | Lead analyst | Sprint 3 | FR1 |

### 2. Assumptions

| ID | Assumption | Validation | Owner | Sprint | Status |
|---|---|---|---|---|---|
| A01 | Olist data is sufficient for an academic prototype | Confirm required fields during data exploration | Lead analyst | Sprints 2-3 | To validate |
| A02 | Historical order information can support late-delivery prediction | Evaluate predictive performance against the logistic-regression baseline | Lead analyst | Sprint 4 | To validate |
| A03 | Open-source Python tools are suitable for development | Test the environment and dependencies; CI installs on every pull request | Lead analyst | Sprint 1 | Met |
| A04 | Operational recommendations can be derived from model outputs | Review recommendations against available evidence in persona review | Product owner | Sprint 5 | To validate |
| A05 | Distance and freight value adequately stand in for missing courier and GPS data | Test feature importance in Sprint 4; state as a limitation in the model card | Lead analyst | Sprint 4 | To validate |
| A06 | Operations team can act on 50-100 flagged orders per day | Recall measured at this capacity; confirm in persona review | Product owner | Sprint 5 | To validate |

### 3. Issues

| ID | Issue | Action | Owner | Sprint | Status |
|---|---|---|---|---|---|
| I01 | No live HESU operational dataset is included | Use Olist for academic research and explain transferability limitations | Product owner | All | Known limitation |
| I02 | Courier GPS and live tracking data are unavailable | Restrict predictions to available historical features (see A05) | Lead analyst | All | Known limitation |
| I03 | About 3% of orders have no delivery date (undelivered or cancelled) | Excluded from training; reviewed separately; cleaning rule documented in pipeline | Lead analyst | Sprints 2-3 | Open |
| I04 | Portuguese review text may contain personal names | Excluded from version 1 features; redaction required before any later use | Data Protection Officer | Sprints 2-3 | Resolved by design |

### 4. Dependencies

| ID | Dependency | Management | Owner | Sprint | Status |
|---|---|---|---|---|---|
| D01 | Olist dataset availability and licence compliance (CC BY-NC-SA 4.0) | Raw copy versioned with DVC; Olist credited | Lead analyst | Sprints 2-3 | Met |
| D02 | Python environment and required libraries | Minimum versions set; CI installs on every pull request | Lead analyst | All | Met |
| D03 | GitHub repository and Project Board | Board has 11 issues (#1-#11); each detailed ticket in docs/backlog.csv links to one | Product owner | Sprint 1 | Met |
| D04 | Completion of data cleaning before model development | Sprint 4 work starts only after DRI-10 and DRI-11 (issue #5) are done | Lead analyst | Sprints 3-4 | Open |
| D05 | Model evaluation before dashboard reporting | Dashboard (DRI-28) waits on fairness report and model card (DRI-22, DRI-24) | Lead analyst | Sprints 4-5 | Open |
| D06 | Responsible AI Review Board sign-off at end of Sprint 3 and Sprint 4 | Blocking tickets DRI-15 (issue #5) and DRI-25 (issue #9) | Data Protection Officer | Sprints 3-4 | Open |

### 5. Monitoring and Review

The RAID Log will be reviewed at each sprint milestone. New risks and issues will be recorded as they arise, while assumptions and dependencies will be updated as evidence becomes available.

The project owner is responsible for maintaining this log and communicating significant risks to relevant stakeholders.
