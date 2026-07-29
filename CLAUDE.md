# Sentinel Attendance — CLAUDE.md

## Goal
Open-source, on-prem attendance system with facial recognition: self-hosted server + any IP/USB camera (no special edge hardware required), inference via a cloud AI vision model API, admin dashboard for enrollment/check-in/attendance logs. Phase 2 adds theft detection: object/asset removal detection + suspicious behavior detection.

## Stack
TBD — will be set when first decision lands. Likely direction: Python/FastAPI backend (camera ingest + vision API calls + attendance/theft logic), Postgres for records, a lightweight web dashboard (Next.js or plain React), Docker Compose for the on-prem self-host deploy.

## Folder structure
- `.claude/skills/` — project-scoped skills (load these in addition to ~/.claude/skills/)
- (add as repo grows)

## Key files
(none yet)

## Deploy
On-prem, self-hosted (Docker Compose target). No cloud hosting requirement — the "cloud" part is only the AI vision model API call, not the deployment.

## Project rules
- Honour all global rules in ~/.claude/CLAUDE.md and ~/.claude/rules/enforcement/
- Update this file when structure, deploy, or key files change
- Full stack stays open source (MIT/Apache), license TBD at first commit

## Linked notes
- [[Sentinel Attendance]]
