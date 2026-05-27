# Epicure MCP Server

Public, anonymous, read-only Model Context Protocol (MCP) server for the
Epicure ingredient-embedding model.

The server is **stateless** and **deterministic**: every tool call is a
pure function of the request arguments plus the bundled artefacts. There
are no external model calls, no embedding fallback, and no user state.

Designed for [Azure Container Apps](https://learn.microsoft.com/azure/container-apps/)
deployment with a replica cap to bound spend.

## Tools

| Category | Tool | Description |
|----------|------|-------------|
| Ported  | `compare_on_axis` | Project two ingredients onto a named axis and compare. |
| Ported  | `pairing_score`   | Overall cosine affinity (300-d) between two ingredients. |
| Ported  | `find_pairings`   | Cluster + bridge graph computed in-process from the bundled embeddings. |
| Ported  | `flavour_correlations` | Which axes correlate with each other. |
| Ported  | `cultural_profile` | Cosine to each cuisine direction. |
| Novel   | `neighbors`        | Top-k cosine neighbours. |
| Novel   | `morph`            | Unified SLERP toward a direction, mode, or ingredient. |
| Novel   | `list_targets`     | Catalogue of valid `morph` targets + `angle_deg` primer. |
| Novel   | `list_factors`     | Residualised ICA factor catalogue (Claude-labelled poles). |
| Novel   | `ingredient_on_factor` | Signed projection onto an ICA factor. |
| Novel   | `pareto_navigate`  | Pareto frontier on (proximity, pole-projection). |
| Novel   | `closest_mode`     | Which named GMM mode the ingredient lives in. |
| Novel   | `where_on_atlas`   | Precomputed UMAP `(x, y)` + nearest-in-2D peers. |

## Local development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Build the data bundle from a local epicure-data checkout
python scripts/build_data.py --source-repo /path/to/epicure-data --out-dir data
python scripts/verify_data.py --data-dir data

# Run server
python -m epicure_mcp.server

# Smoke-test
curl http://localhost:8080/healthz
```

Endpoints:

| Path | Method | Description |
|------|--------|-------------|
| `/healthz` | GET | Liveness probe (does not load the bundle). |
| `/mcp`     | POST | Streamable HTTP MCP JSON-RPC endpoint. |

## Environment variables

| Var | Default | Description |
|-----|---------|-------------|
| `EPICURE_DATA_DIR` | `<repo>/data` | Bundled-artefact directory. |
| `HOST` | `0.0.0.0` | Bind address. |
| `PORT` | `8080` | Bind port. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Token-bucket refill rate. |
| `RATE_LIMIT_BURST` | `10` | Token-bucket capacity. |
| `MCP_SERVER_NAME` | `epicure` | Reported in the MCP `initialize` response. |

The server is fully self-contained: there is no upstream API call.
`find_pairings` runs the graph algorithm locally against the bundled
embeddings + ingredient metadata.

## Bundled data

The `data/` directory is **committed to this repo** (~13 MB) so the
server is fully self-contained: clone, build, deploy. No external data
checkout required.

| File | Source | Size |
|------|--------|------|
| `embeddings.csv` | epicure-data: `deploy/payload/embeddings.csv` | ~10 MB |
| `ingredient_list.csv` | epicure-data payload | ~75 KB |
| `ingredient_tags.csv`  | epicure-data payload | ~100 KB |
| `consolidated_nodes.csv` | epicure-data payload | ~70 KB |
| `factor_labels_ica_cooc.json` | `application/paper/results/` | ~75 KB |
| `mode_explorer_cooc.json` | `application/exploratory/results/` | ~2 MB |
| `supervised_directions.npz` | computed (38 axes) | ~55 KB |
| `factor_dirs_ica_n20.npy` | computed (20 unit vectors) | ~25 KB |
| `mode_poles_cooc.npy` | computed (150 unit vectors) | ~180 KB |
| `umap_coords.csv` | computed (1,790 x 2) | ~55 KB |

### Refreshing the bundle when the model changes

When a new `epicure-data` training run lands, regenerate the bundle from
a local checkout and commit the diff:

```bash
python scripts/build_data.py --source-repo /path/to/epicure-data --out-dir data
python scripts/verify_data.py --data-dir data
git add data/ && git commit -m "data: refresh bundle from <run-id>"
```

## Azure Container Apps deployment

### One-time setup

You need the Azure CLI installed and an authenticated session:

```bash
# Install az (Ubuntu/Debian)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login
az account set --subscription "<your-sub-id>"

# Provision RG + ACR + ACA env + container app + GitHub OIDC federation
./scripts/azure_setup.sh
```

The script prints the GitHub Actions secrets / variables you must set
on the repo for the deploy workflow to function.

### Continuous deployment

`.github/workflows/deploy.yml` runs on every push to `main`:

1. Checks out the repo (the data bundle is already inside it).
2. Builds & pushes the Docker image to ACR via OIDC.
3. Calls `az containerapp update` and waits for the new revision to
   answer `/healthz`.

### Scaling and rate limit

- `--max-replicas 3` puts a hard cap on burst spend.
- `--min-replicas 0` allows scale-to-zero (cold start ~3-5 s while
  the bundle loads).
- The in-process token bucket limits each client IP to 60 req/min with
  a burst of 10. Limits drift across replicas; precision is bounded by
  the replica cap.

## Connecting clients

Once deployed, the MCP endpoint is `https://<aca-fqdn>/mcp`.

### Claude.ai

Add a custom MCP server in **Settings -> Integrations -> Add custom**:

```
Name: Epicure
URL : https://<aca-fqdn>/mcp
Auth: None
```

### Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "epicure": {
      "transport": "streamable-http",
      "url": "https://<aca-fqdn>/mcp"
    }
  }
}
```

### ChatGPT (custom GPT)

Use Actions with the OpenAPI schema generated from the MCP `tools/list`
response.

## License

[MIT](LICENSE)
