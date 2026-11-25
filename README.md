---
title: ClaimsSQL-Trust
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.67.0
app_file: app/streamlit_app.py
pinned: false
short_description: Text-to-SQL on Medicare claims that can abstain
---

# ClaimsSQL-Trust demo

Ask questions about CMS DE-SynPUF Sample 1 (synthetic Medicare claims). The system
answers only when the SQL passes a schema check, a dry run and a self-consistency
check; otherwise it says why it will not answer.

- **Benchmark explorer**: what each model did on the 200 benchmark questions.
- **Ask your own question**: the live pipeline on a 5,000-member subset.

Built from commit `d5cbc0e`. Data: CMS 2008-2010 DE-SynPUF Sample 1 (synthetic).
