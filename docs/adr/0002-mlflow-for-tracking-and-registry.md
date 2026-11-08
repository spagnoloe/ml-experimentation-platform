# MLflow for tracking and registry instead of custom tables

Every Run, its Candidates (as child runs), metrics, params and the fitted pipeline are recorded in an MLflow tracking server, and Model Versions will live in MLflow's registry. Our own Postgres tables hold only what MLflow does not model well: Training Jobs (queue state and progress) and Deployments (ADR-0001). We chose MLflow over writing `runs`, `metrics` and `artifacts` tables ourselves because it gives the data scientist a comparison UI, nested runs, versioned model storage with signatures, and an S3-ready artifact proxy for free, and it has managed equivalents in production (SageMaker managed MLflow).

All Runs go to one MLflow Experiment, `california-housing`, so every algorithm and config can be compared in one table. The config name is a run tag, not a separate Experiment.

## Considered Options

- **Custom tables plus pickles on a volume**: full control, but we would rebuild a run browser, artifact storage and a model registry, which is most of the 4-hour budget, and it is exactly what the notebook got wrong (issues 5, 6 and 9).
- **One MLflow Experiment per config name**: matches MLflow tutorials, but splits the Leaderboard across Experiments and makes cross-algorithm comparison awkward in the MLflow UI.

## Consequences

- MLflow owns the `mlflow` database and its migrations. We never write to it with SQL.
- Artifacts go over HTTP through the server's artifact proxy (`--serve-artifacts`), so no client mounts the artifacts volume. Moving to S3 changes one server flag.
- Pipelines are logged with MLflow's sklearn flavour, serialized with skops, with an explicit list of trusted types (`SKOPS_TRUSTED_TYPES`), and a signature built from the schema module.
- The MLflow server runs from our own image, so it shares the Postgres driver and the pinned MLflow version with its clients.
- A job's Run and its row in `training_jobs` must agree on failure. The worker marks both failed, including after a restart.
