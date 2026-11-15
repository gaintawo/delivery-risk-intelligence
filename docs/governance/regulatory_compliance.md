# Regulatory Compliance Report

Delivery Risk Intelligence · November 2026 · prepared for the Data Protection Officer. This is a project assessment, not legal advice.

## 1. Which laws apply

| Law | Applies? | Why |
| --- | --- | --- |
| **LGPD**, Brazil (Lei 13.709/2018) | **Yes: primary** | Source data describes orders by Brazilian customers and sellers |
| **GDPR**, EU (Regulation 2016/679) | No, used as a benchmark | No EU data subjects and no EU establishment; the LGPD closely follows the GDPR, so GDPR principles are used as the reference standard |
| **CCPA / CPRA**, California | No | No California consumers; the business thresholds are not met |
| **HIPAA**, United States | No | No health data and no covered entity or business associate |
| **Tanzania PDPA 2022**, **Kenya DPA 2019** | Only if piloted at HESU or in Kenya | Module 1 names East Africa as the next context; controls are designed so the same model can be assessed there |

## 2. Assessment against LGPD and GDPR principles

| Principle (LGPD Art. 6 / GDPR Art. 5) | How DRI complies | Evidence |
| --- | --- | --- |
| Purpose limitation | Data used only to improve delivery reliability and customer updates; not marketing, not seller penalties | Charter; governance principle 1 |
| Data minimisation | Allow-list of columns; free text, postcodes, coordinates and raw IDs deleted | `src/pipeline/anonymize.py`; privacy gate |
| Accuracy | 35 Great Expectations checks; the model-ready suite blocks loading if any of 22 fail | `docs/evidence/` GX results |
| Storage limitation | 24-month retention; explanation store and audit log on the same schedule | `docs/data_governance.md` §3 |
| Security | Salted hashing with the salt held as a secret; least-privilege roles; non-root containers; secrets never in code | `.env.example`; Dockerfiles |
| Transparency to data subjects | Public Trust Statement; reasons for every score | `public_trust_statement.md` |
| Accountability | Audit log of every data access; explanation store of every score; Review Board records | `logs/audit.log`; `api/store.py` |
| Lawful basis | Legitimate interest (LGPD Art. 7 IX; GDPR Art. 6(1)(f)) for improving delivery service, with a balancing test in the AIA | `algorithmic_impact_assessment.md` |
| Automated decisions (LGPD Art. 20; GDPR Art. 22) | No decision based solely on automated processing: advisory, shadow mode, human decides. People can ask for a review and receive the stored explanation | `GET /explanations/{id}` |
| Data protection impact assessment (LGPD Art. 38; GDPR Art. 35) | AIA plus privacy plan serve as the impact report | AIA §3 |
| International transfer | Cloud hosting (Render, Streamlit Community Cloud, GitHub) serves only the public, de-identified extract and the model; no restricted data leaves the controlled environment | `dashboard/data/README.md` |

## 3. Data-subject rights and incident duties

| Right or duty | Process | Owner | Timing |
| --- | --- | --- | --- |
| Access or explanation of a score | Retrieve by prediction ID from the explanation store | DPO with model owner | 15 days (LGPD Art. 19) |
| Review of an automated outcome | Human review by operations lead; seller representative informed | Operations lead | 15 days |
| Deletion | Delete from operational stores; pseudonymised training data rebuilt at next retrain | Data owner | Next monthly cycle |
| Security incident | Incident response plan, privacy track | DPO | Notify ANPD and affected people within 3 working days (ANPD Resolution 15/2024) |

## 4. Licence compliance

Olist data is CC BY-NC-SA 4.0: credited in every output, used non-commercially (coursework), and the public extract keeps the same licence.

## 5. Findings

No blocking compliance issue for the current shadow-mode, coursework use. Two conditions before production: (1) the DPO confirms the lawful basis and retention for live order data; (2) before any East African pilot, a fresh assessment under the Tanzania PDPA 2022 (registration with the Personal Data Protection Commission) or the Kenya DPA 2019.
