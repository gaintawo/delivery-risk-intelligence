# Raw Data

The nine original Olist files (Olist, 2018; CC BY-NC-SA 4.0). They are never edited and never committed to Git.

Run `python -m src.pipeline.flow`: the first stage downloads any missing file from Olist's public copy and records each file's SHA-256 fingerprint in `logs/audit.log`.
