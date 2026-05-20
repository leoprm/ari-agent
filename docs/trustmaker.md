# Ari Agent — Hermes Agent custom for Trust Maker

**Ari Agent** is a customized fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com), tailored to power **Ari**, the AI assistant in [Trust Maker](https://github.com/leoprm/Trust-Suite).

## What is Trust Maker?

Trust Maker is a chat-first platform where communities self-organize via Telegram + AI. Each community ("tree") has its own sandboxed AI assistant (Ari) that helps with decisions, memory, documents, and coordination.

- **Bot:** [@TrustMakerBot](https://t.me/TrustMakerBot)
- **Repo:** https://github.com/leoprm/Trust-Suite

## Key customizations over upstream Hermes Agent

- **Tree-scoped memory** — each tree gets isolated session persistence and Obsidian vault
- **Session search block** — cross-tree memory isolation
- **Gateway patches** — Telegram topic routing, interaction modes (MAXIMUM/MEDIUM/MINIMUM)
- **Sandbox REST API** — Ari accesses filesystem exclusively via HTTP endpoints (no direct file/terminal tools)
- **7-tool constraint** — Ari runs with exactly 7 tools: `skill_view`, `skill_manage`, `skills_list`, `web_search`, `web_extract`, `session_search`, `execute_code`

## Linux user setup

This agent runs as the `trustmaker` Linux user with strict isolation:

```bash
# 1. Create the trustmaker user
sudo useradd -m -s /bin/bash trustmaker

# 2. Clone the Ari Agent fork
sudo -u trustmaker git clone https://github.com/leoprm/ari-agent.git /home/trustmaker/.hermes/hermes-agent
cd /home/trustmaker/.hermes/hermes-agent
sudo -u trustmaker git checkout ari-agent

# 3. Install Hermes Agent
sudo -u trustmaker bash setup-hermes.sh

# 4. Configure the trustmaker profile
#    Port: 8644
#    Tools: web, skills, session_search, code_execution
#    No terminal, no file, no memory tools
sudo -u trustmaker hermes config edit

# 5. Start the gateway
sudo -u trustmaker hermes gateway start --profile trustmaker
```

### Filesystem isolation

```
/home/trustmaker/           # home directory (0700)
├── .hermes/                # Hermes Agent config & data
│   └── hermes-agent/       # this repo
├── trees/                  # sandbox workspaces (one per tree)
│   └── <treeId>/
│       ├── obsidian/       # Ari's memory vault
│       ├── apps/           # sandboxed applications
│       ├── data/           # tree data
│       └── context/        # tree context files
└── workers/                # worker sandboxes
```

### Security

- **No sudo access** — `trustmaker` user runs without privileges
- **Kernel capabilities all zero** — no `CAP_SYS_ADMIN`, `CAP_DAC_OVERRIDE`, etc.
- **Sandbox isolation via bubblewrap (bwrap)** — exec commands run in PID/IPC namespaces
- **Filesystem access only via REST API** — POST `/api/trees/:treeId/sandbox/exec|read|write|upload`
- **API key per tree** — HMAC-derived key from master `HERMES_API_SERVER_KEY`

## Branch strategy

| Branch | Purpose |
|--------|---------|
| `main` | Upstream tracking (NousResearch/hermes-agent) |
| `ari-agent` | Custom patches for Trust Maker |

## Related

- [Trust Maker](https://github.com/leoprm/Trust-Suite) — the platform Ari runs on
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — upstream project
