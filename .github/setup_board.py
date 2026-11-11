"""Create the DRI GitHub Project board from docs/backlog.csv.

Run once from the repository root, after pushing the repo to GitHub:

    gh auth login                      # if not already signed in
    gh auth refresh -s project         # gives gh permission to manage Projects
    python .github/setup_board.py

What it does:
  1. Creates the labels (data, model, fairness, product, infra, docs).
  2. Creates one milestone per sprint (S0-S4).
  3. Creates one issue per ticket (DRI-01 ... DRI-32).
  4. Creates a user-owned Project "DRI Delivery Board", links it to this repo,
     adds two single-select fields - Stage (the Kanban columns) and Sprint -
     and sets both on every issue.
  5. Closes the S0 tickets already finished in Module 2.

Afterwards, in the Project's Board view, choose "Column by: Stage" and
"Group by: Sprint" (or filter by Sprint), then take the screenshot.
Requires the GitHub CLI (gh) 2.40 or newer. Safe to read before running:
it only calls gh.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs" / "backlog.csv"
PROJECT_TITLE = "DRI Delivery Board"
STAGES = ["Backlog", "Sprint Ready", "In Progress", "In Review", "Done"]
SPRINTS = ["S0", "S1", "S2", "S3", "S4"]
LABELS = {
    "data": ("1D76DB", "Pipeline and data quality"),
    "model": ("5319E7", "Features, training and evaluation"),
    "fairness": ("D93F0B", "Fairness, ethics and Review Board gates"),
    "product": ("0E8A16", "Recommendations and dashboard"),
    "infra": ("6A737D", "Repo, CI, cloud and deployment"),
    "docs": ("FBCA04", "Documentation and governance"),
}


def gh(*args, parse=False):
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out) if parse else out.strip()


def main():
    rows = list(csv.DictReader(BACKLOG.open(encoding="utf-8")))
    repo = gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    owner = repo.split("/")[0]
    print(f"Repository: {repo} ({len(rows)} tickets)")

    for name, (color, desc) in LABELS.items():
        gh("label", "create", name, "--color", color, "--description", desc, "--force")

    existing = {m["title"] for m in gh("api", f"repos/{repo}/milestones?state=all", parse=True)}
    for title in dict.fromkeys(r["milestone"] for r in rows):
        if title not in existing:
            gh("api", f"repos/{repo}/milestones", "-f", f"title={title}")

    proj = gh("project", "create", "--owner", owner, "--title", PROJECT_TITLE, "--format", "json", parse=True)
    num, proj_id = str(proj["number"]), proj["id"]
    gh("project", "link", num, "--owner", owner, "--repo", repo)
    gh("project", "field-create", num, "--owner", owner, "--name", "Stage",
       "--data-type", "SINGLE_SELECT", "--single-select-options", ",".join(STAGES))
    gh("project", "field-create", num, "--owner", owner, "--name", "Sprint",
       "--data-type", "SINGLE_SELECT", "--single-select-options", ",".join(SPRINTS))
    fields = {f["name"]: f for f in gh("project", "field-list", num, "--owner", owner,
                                         "--format", "json", parse=True)["fields"]}
    opt = {name: {o["name"]: o["id"] for o in fields[name]["options"]} for name in ("Stage", "Sprint")}

    for r in rows:
        body = (f"**Deliverable:** {r['deliverable']}\n\n"
                f"**Traces to (Module 1):** {r['traces_to']}\n\n"
                f"**Acceptance criteria:** {r['acceptance_criteria']}\n")
        url = gh("issue", "create", "--title", f"{r['id']} {r['title']}", "--body", body,
                 "--label", r["labels"], "--milestone", r["milestone"])
        item = gh("project", "item-add", num, "--owner", owner, "--url", url, "--format", "json", parse=True)
        for field, value in (("Stage", r["stage"]), ("Sprint", r["sprint"])):
            gh("project", "item-edit", "--id", item["id"], "--project-id", proj_id,
               "--field-id", fields[field]["id"], "--single-select-option-id", opt[field][value])
        if r["stage"] == "Done":
            gh("issue", "close", url, "--reason", "completed")
        print(f"  {r['id']}  {r['stage']:<12} {r['title']}")

    print(f"\nDone. Open the board: gh project view {num} --owner {owner} --web")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"gh failed: {' '.join(e.cmd)}\n{e.stderr}")
