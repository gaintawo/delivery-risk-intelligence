
# Delivery Risk Intelligence (DRI)
## RAID Log – Module 2

### 1. Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R01 | Missing or inconsistent delivery dates | High | Validate records and document exclusions |
| R02 | Data leakage during model training | High | Use only features available at prediction time |
| R03 | Historical Olist data may not reflect current logistics conditions | High | Document limitations and avoid unsupported generalisation |
| R04 | Model performance may vary across regions | High | Evaluate regional fairness using Fairlearn |
| R05 | Limited project time and computing resources | Medium | Prioritise essential deliverables |

### 2. Assumptions

| ID | Assumption | Validation |
|---|---|---|
| A01 | Olist data is sufficient for an academic prototype | Confirm required fields during data exploration |
| A02 | Historical order information can support late-delivery prediction | Evaluate predictive performance |
| A03 | Open-source Python tools are suitable for development | Test the environment and dependencies |
| A04 | Operational recommendations can be derived from model outputs | Review recommendations against available evidence |

### 3. Issues

| ID | Issue | Status | Action |
|---|---|---|---|
| I01 | No live HESU operational dataset is included | Known limitation | Use Olist for academic research and explain transferability limitations |
| I02 | Courier GPS and live tracking data are unavailable | Known limitation | Restrict predictions to available historical features |

### 4. Dependencies

| ID | Dependency | Importance |
|---|---|---|
| D01 | Olist dataset availability and licence compliance | High |
| D02 | Python environment and required libraries | High |
| D03 | GitHub repository and Project Board | High |
| D04 | Completion of data cleaning before model development | High |
| D05 | Model evaluation before dashboard reporting | High |

### 5. Monitoring and Review

The RAID Log will be reviewed at each sprint milestone. New risks and issues will be recorded as they arise, while assumptions and dependencies will be updated as evidence becomes available.

The project owner is responsible for maintaining this log and communicating significant risks to relevant stakeholders.
