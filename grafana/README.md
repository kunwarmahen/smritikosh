# Grafana observability config

Ready-to-provision Grafana assets for a Smritikosh deployment. Prometheus must
scrape the API's `GET /metrics` (and the standalone worker's
`WORKER_METRICS_PORT`, if enabled — job and LLM series come from whichever
process runs the scheduler). Series definitions: `smritikosh/metrics.py`.

## Contents

| Path | What |
|---|---|
| `dashboards/smritikosh_api_observability.json` | API dashboard — latency, throughput, error rates, in-flight requests |
| `alerting/smritikosh_alerts.yaml` | Alert rules (items B1/G1) — job staleness & failures, LLM error ratio & cost burn, task-queue inline fallback, API 5xx / p95 latency |

## Dashboard

Import via **Dashboards → New → Import** and pick your Prometheus datasource
when prompted (the JSON uses a `DS_PROMETHEUS` input, so no editing needed).

## Alert rules

The rules file uses Grafana's [file provisioning](https://grafana.com/docs/grafana/latest/alerting/set-up/provision-alerting-resources/file-provisioning/)
format. Mount it into the Grafana container:

```yaml
# docker-compose snippet
grafana:
  image: grafana/grafana:11.4.0
  volumes:
    - ./grafana/alerting:/etc/grafana/provisioning/alerting:ro
```

The rules reference the Prometheus datasource by `uid: prometheus`. Either
provision the datasource with that UID:

```yaml
# /etc/grafana/provisioning/datasources/prometheus.yaml
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
```

…or replace `datasourceUid: prometheus` in `alerting/smritikosh_alerts.yaml`
with your existing datasource's UID (**Connections → Data sources → settings →
UID**).

### Thresholds to review per deployment

- **Job staleness** — thresholds assume the production default crons in
  `smritikosh/config.py` (consolidation hourly, pruning daily, clustering 6 h,
  belief mining 12 h, fact decay weekly), alerting at ~2× the interval. If you
  override `SCHEDULER_*_CRON` (the dev `.env` runs everything in minutes),
  adjust the evaluator params.
- **LLM cost burn** — defaults to $5/hour; deliberately low. Raise it for
  large deployments and silence it during benchmark runs
  (`scripts/run_publishable_benchmarks.sh` legitimately burns much more).
- **Inline task fallback** — expected to fire only when Redis is deliberately
  absent (single-process dev). In production it means the ARQ taskworker path
  is broken and LLM write-backs are running inside the API process.
- **API p95 latency** — `/context` and `/memory/event` are LLM-bound; 5 s
  suits a cloud provider, not a single-GPU local model.

Alerts carry `severity` (`warning` | `critical`) and `component`
(`scheduler` | `llm` | `tasks` | `api`) labels for notification-policy routing.
Contact points / notification policies are deployment-specific and not
provisioned here.

## Remaining (deferred)

- Dashboard rows for the job/LLM/task series (`smritikosh_job_*`,
  `smritikosh_llm_*`, `smritikosh_tasks_total`) — the alert rules cover the
  failure modes; graphs are a nice-to-have alongside a real deployment.
- Memory-store size gauges (deferred in G1).
