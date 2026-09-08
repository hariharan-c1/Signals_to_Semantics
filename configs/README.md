# Configuration

Tracked configuration files contain no credentials or machine-specific paths.

1. Copy `paths.example.yaml` to `paths.yaml` and set the local Argoverse 2 root.
2. Copy `db.env.example` to `db.env` before using S6 or S7.
3. Export Azure variables only when selecting the Azure backend in
   `llm_backends.yaml`.

Generated `paths.yaml`, `db.env`, and any `.env` file are ignored by Git.
