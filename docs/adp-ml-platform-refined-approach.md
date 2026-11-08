# ML Platform — Refined Implementation Requirements

Nov 8, 2026 · @Enric

## 1. Context and success criteria

We build a local proof of concept, capped at 4 hours, that takes a model from tracked experiment to served prediction and back to rollback. It must run with one command and map onto a production design without rewrites.

**Constraints**

- Hard cap of 4 hours. A well-executed subset beats a rushed full system.
- Runs locally via `make up` (Docker Compose), with no cloud dependencies.
- Every local component has a named production counterpart (section 14).

**Done means this demo path works end to end:**

1. Launch a training job from the UI, either with fixed hyperparameters or a search (random by default, grid optional), and watch its progress.
2. Compare finished runs on a leaderboard where the best run is highlighted.
3. Register a run's pipeline as a new version of a registered model, and optionally set an alias on it.
4. Deploy that version through `POST /deploy`; the API validates it before switching.
5. Send a prediction from the UI test form or `curl` and see which model version answered.
6. Roll back to the previous version; the deployment history shows both events.

## 2. Notebook issues and the rule that prevents each

The inherited notebook has nine defects. Each becomes a platform rule, and the README repeats this table so reviewers see the reasoning.

| # | Issue | Evidence in notebook | Platform rule |
| --- | --- | --- | --- |
| 1 | Scaling leaks test data into training | `scaler.fit_transform(df[features])` runs on the full dataframe before the split | All preprocessing lives inside a scikit-learn `Pipeline`, fit only on training folds |
| 2 | Target overwritten in place | `y_train = np.log1p(y_train)`; re-running the cell applies log twice, and later cells inherit it | Target transform is a config option via `TransformedTargetRegressor`; data is never mutated |
| 3 | Metrics in log space | MSE and R² computed on `log1p` values | All metrics are reported in original units (MedHouseVal, $100k) |
| 4 | Engineered feature leaks across models | `X_train["BedroomsPerRoom"] = …` mutates shared state | Engineered features are a pipeline step, toggled per experiment |
| 5 | One artifact, always overwritten | Every model dumps to the same `model.pkl` | Each run logs its own artifact; registry versions are immutable |
| 6 | Results lost | `mse` and `r2` overwritten; `gb_model` defined twice | Every run persists params, metrics, artifact and metadata |
| 7 | Hardcoded local path | `/Users/john/.../training_data.csv` | Data path comes from settings; dataset identified by its sha256 |
| 8 | Model selected on the test set | Only a train/test split exists | Selection uses k-fold CV on train; the test set is scored once per finished run |
| 9 | Untracked serialization | Raw `pickle.dump`, no library versions recorded | MLflow sklearn flavour, scikit-learn version recorded, only trusted artifacts loaded |

## 3. Scope

Everything in Must is required for the demo path; Should is built only if time remains; Out of scope is described in the production doc instead.

| Priority | Items |
| --- | --- |
| Must | CV training with fixed hyperparameters or search (random default, grid optional) · background worker with progress · MLflow tracking and registry with versions and aliases · best run highlighted · `/deploy` with validation · `/reload` that reconciles with the DB · deployment history and rollback · `/predict`, `/model`, `/health` · Streamlit UI · CLI to submit jobs · Docker Compose · tests · README · production architecture doc · decision records |
| Should | `DummyRegressor` baseline in the algorithm list · link from Streamlit to the MLflow UI · seed command that trains and registers one model per algorithm |
| Out of scope | Batch predictions · periodic background reconcile · authentication · plausible-range input validation · more than one environment · user-extensible algorithm list · dataset versioning beyond a content hash · concurrent training jobs |

**Accepted risk (no periodic reconcile):** if the DB's active deployment changes without going through `/deploy` (a manual edit, or a second API process), the running API keeps its current model until `/reload` is called or it restarts. With one API process and `/deploy` as the only writer, this does not happen in the demo.

## 4. Architecture

Four containers share two volumes; each piece of state has exactly one writer, which keeps the SQLite setup safe and makes the production split obvious.

&#91;embedded content: Local architecture · 4 containers, 2 stores\]

| State | Store | Writer | Readers |
| --- | --- | --- | --- |
| Training jobs: status, progress, error | app.db `training_jobs` | UI and CLI insert `queued`; worker updates | UI |
| Runs: params, metrics, artifacts | MLflow tracking | Worker | UI leaderboard, MLflow UI |
| Registered models, versions, aliases | MLflow registry | UI (register, set or remove alias) | API, UI |
| Deployment events: active model and history | app.db `deployment_events` | API only, through `/deploy` | API (startup, `/reload`); UI through `GET /deployments` |

- **Two SQLite files on purpose.** MLflow owns its schema and migrations; our tables live in `app.db` behind a SQLAlchemy repository, opened in WAL mode.
- **Volumes:** `appdata` (app.db) is mounted in `ui`, `worker` and `api`; `mlflowdata` (mlflow.db and artifacts) only in `mlflow`.
- **Artifacts go over HTTP.** The MLflow server proxies artifacts (`--serve-artifacts`), so worker, UI and API never read artifact paths directly. Swapping the volume for S3 later changes one server flag.
- **Service names, not localhost:** clients use `http://api:8000` and `http://mlflow:5000` from settings.

## 5. Data and evaluation protocol

Every experiment uses the same data, the same held-out test set and the same CV folds, so any two runs on the leaderboard are directly comparable.

- **Source:** `data/training_data.csv`, committed to the repo; the path comes from settings (`DATA_PATH`).
- **Schema in one module** (`app/ml/schema.py`): the 8 float features `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`; target `MedHouseVal` in units of $100k. Training, the API request model and the UI form all import it. The loader checks columns and dtypes and fails fast.
- **Dataset identity:** sha256 of the CSV, logged on every run as `dataset_sha256`. The leaderboard only compares runs with the current hash.
- **Split:** 80/20 train/test with `random_state=42`, fixed in settings rather than per experiment. Logged as `split_seed` and `test_size`. The test set is never used for selection.
- **Selection:** `KFold(n_splits=5, shuffle=True, random_state=42)` on the training set; every candidate in every job sees the same folds.
- **Metrics, all in original units:** RMSE (primary), MAE and R². CV metrics are logged as mean and std across folds; test metrics once, after refitting the chosen candidate on the full training set.
- **Best run:** lowest mean CV RMSE among completed runs on the current dataset hash. Test RMSE is shown next to it but never used to rank.

**Known data property:** in the scikit-learn version of this dataset the target is capped at about 5.0, so errors on expensive districts are understated. The README notes this as a limitation; we do not correct for it.

## 6. Experiment configuration and training flow

An experiment is a validated config. A fixed-hyperparameter run is treated as a search with exactly one candidate, so both modes share one code path.

**Config** (Pydantic `ExperimentConfig`; the UI builds it, the CLI loads it from YAML):

```yaml
name: gb-log-target
algorithm: gradient_boosting      # linear_regression | random_forest | gradient_boosting | dummy
features:
  bedrooms_per_room: true
target_transform: log1p           # none | log1p
tuning:
  mode: search                    # fixed | search
  strategy: random                # random (default) | grid
  n_iter: 20                      # random only
  params:
    n_estimators: {type: int, low: 50, high: 400}
    learning_rate: {type: float, low: 0.01, high: 0.3, log: true}
    max_depth: [2, 3, 4, 5]
random_state: 42
```

- **Parameter values by mode:** `fixed` takes single values; `grid` takes lists; `random` takes lists (sampled uniformly) or ranges `{type, low, high, log}`, mapped to scipy `randint`, `uniform` or `loguniform`.
- **Algorithms are a fixed list** in `app/ml/algorithms.py`, each with default fixed values and a default search space that prefill the UI. `dummy` takes no parameters and only runs in `fixed` mode.
- **Validation at submit time:** unknown parameter names are rejected (checked against the estimator's `get_params()`), and the candidate count is computed and shown before launch. A grid above `MAX_CANDIDATES` (200) is rejected.
- **Pipeline per run:** optional `BedroomsPerRoom` step (a transformer that appends the ratio with safe division) → `StandardScaler` for linear regression only → estimator. With `target_transform: log1p` the whole pipeline is wrapped in `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`, so predictions and metrics come out in original units.
- **Parameter names stay simple:** configs use estimator-level names; the code adds the pipeline prefixes.

**Job flow in the worker:**

1. Load the CSV, compute its hash, apply the fixed split.
2. Enumerate candidates with `ParameterGrid` or `ParameterSampler(n_iter, random_state)`; `fixed` gives one candidate.
3. Open an MLflow parent run tagged with `job_id`, algorithm, mode and strategy. Log the config YAML, dataset hash, split settings, git SHA, and Python and scikit-learn versions.
4. For each candidate, run `cross_validate` on the fixed folds (RMSE, MAE, R²), log a child run with its params and CV metrics, then update job progress.
5. Choose the candidate with the lowest mean CV RMSE and log its params and CV mean and std on the parent run.
6. Refit it on the full training set, score it once on the test set, and log the test metrics.
7. Log the fitted pipeline with `mlflow.sklearn.log_model`, including a signature built from the schema and an input example of 5 rows.
8. Mark the job `completed` with its `mlflow_run_id`. Any exception marks the job `failed` with the error text, and the parent run `FAILED`.

The leaderboard ranks parent runs only; child runs stay visible in the MLflow UI.

## 7. Training jobs and worker

Jobs are rows in `training_jobs`. One worker process runs them one at a time and writes progress after every candidate.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Primary key |
| `name` | text | From the config |
| `config` | JSON | Validated `ExperimentConfig` |
| `status` | enum | `queued`, `running`, `completed`, `failed` |
| `phase` | text, nullable | `cv`, `refit`, `logging` |
| `progress_done`, `progress_total` | int | Candidates finished and total |
| `mlflow_run_id` | text, nullable | Set when the parent run opens |
| `error` | text, nullable | Exception message on failure |
| `created_at`, `started_at`, `finished_at` | timestamp (UTC) |  |

**Submitting:** the UI and the CLI (`uv run python -m app.cli submit configs/gb.yaml`, wrapped as `make train CONFIG=…`) both call one `jobs.submit(config)` service that validates and inserts a `queued` row. The CLI takes an optional `--wait` that polls until the job finishes.

**Worker loop:**

- Every 2 seconds (`WORKER_POLL_SECONDS`), pick the oldest `queued` job and claim it with `UPDATE … SET status='running' WHERE id=:id AND status='queued'`. Run it only if exactly one row changed.
- On startup, mark any `running` rows as `failed` with error `worker restarted`. With a single worker, any such row is orphaned.
- The work itself is one function, `run_job(job_id)`, which tests call directly without the polling loop.

**Progress in the UI:** while any job is `queued` or `running`, the jobs table refreshes every 2 seconds and shows a progress bar (`progress_done / progress_total`) plus the current phase.

## 8. Model registry and best-model selection

MLflow's registry holds registered models, auto-incremented versions and user aliases. Our code adds only the leaderboard and the rule that defines "best".

- **Register:** from a completed run on the leaderboard, the user picks a registered model name (default `california-housing`, free text allowed). The UI calls `mlflow.register_model("runs:/<run_id>/model", name)` and MLflow assigns the next version.
- **Version tags:** `job_id`, `algorithm`, `cv_rmse_mean`, `cv_rmse_std`, `test_rmse`, `dataset_sha256`, so the registry page needs no extra lookups.
- **One version per run and name:** the Register button is disabled for a run already registered under that name.
- **Aliases:** set, move or remove from the Registry page via `set_registered_model_alias` and `delete_registered_model_alias`. An alias points at one version at a time. Aliases are labels for people; deployments never follow them (section 9).
- **Best run:** the leaderboard highlights the completed parent run with the lowest mean CV RMSE on the current dataset hash. The Registry page puts a "best" badge on any version created from that run.
- **Noise is visible:** the leaderboard shows CV RMSE as mean ± std, and the `dummy` baseline's RMSE when one exists, so a narrow win can be judged by eye. There is no automatic significance test.

## 9. Deployment and model switching

A deployment is an append-only event naming an exact `(model_name, version)`. The latest event per environment is the active model, and only the API writes events, after it has validated the model.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | int, autoincrement | Ordering key; highest id per environment is active |
| `environment` | text | Always `local` for now |
| `model_name`, `model_version` | text, int | Exact version, never an alias |
| `action` | enum | `deploy` or `rollback` |
| `previous_model_name`, `previous_model_version` | nullable | For the history view |
| `created_at` | timestamp (UTC) |  |

Rows are never updated or deleted.

**`POST /deploy {model_name, version, action}`** (`action` defaults to `deploy`):

1. If the requested version is already loaded, return 200 and write nothing.
2. Load `models:/{model_name}/{version}` from MLflow into a local variable.
3. Validate it: signature input columns equal the schema's 8 features; a prediction on the schema's bundled sample rows returns finite values of shape `(n,)`.
4. On failure, return 422 with the reason. Nothing is written and the current model keeps serving.
5. On success, insert the deployment event in a transaction.
6. Swap the in-memory model under a lock and return the new model info.

If the insert fails, return 500 and do not swap. If the process dies between steps 5 and 6, startup loads the already validated model.

**Rollback:** the Deployments page offers "Roll back" (deploy the previous event's version with `action: rollback`) and "Redeploy" on any history row. Both go through `/deploy`, so they are validated and recorded like any deploy.

**`POST /reload`** reconciles with the DB: read the latest event and, if it differs from the loaded model, run steps 2, 3 and 6 without inserting. If validation fails it keeps the current model and returns 409; `/model` then shows `in_sync: false`.

**Startup** runs the same reconcile. With no event, or if loading fails, the API starts without a model: `/predict` returns 503 and `/health` reports `degraded`.

The API runs as a single uvicorn worker; the lock serializes concurrent deploys within it.

## 10. Serving API and ModelManager

Consumers only need `/predict` and `/model`; the HTTP layer is thin, and all loading, validation and swapping live in `ModelManager`.

| Method | Path | Used by | Request | Response |
| --- | --- | --- | --- | --- |
| GET | `/health` | Ops | – | `status` (`ok` or `degraded`), `model_loaded` |
| GET | `/model` | Consumers, UI | – | Loaded model (name, version, run id, algorithm, loaded at), desired model from DB, `in_sync` |
| POST | `/predict` | Consumers, UI test form | The 8 features as named floats | `prediction`, `unit: "100k USD"`, `model_name`, `model_version` |
| POST | `/deploy` | UI (admin) | `model_name`, `version`, optional `action` | New model info; 422 or 500 on failure |
| POST | `/reload` | Admin | – | `changed`, model info; 409 if validation fails |
| GET | `/deployments` | UI (admin) | `limit` | Deployment events, newest first |

- **Request model** is built from the schema module: all 8 fields required floats, `extra="forbid"`. Types only; range checks are left to the production doc. Bad input gets FastAPI's standard 422.
- **Predict** builds a one-row DataFrame in schema column order and calls the pipeline. It reads the current model reference once per request, so a swap mid-request is safe.
- **Admin endpoints** live in a separate router tagged `admin`, so production can put them behind auth or an internal network without touching consumer routes.
- **`ModelManager`** exposes `current()`, `load_and_validate(name, version)`, `swap(loaded)` and `reconcile()`. A `LoadedModel` dataclass holds the pipeline, name, version, run id, algorithm and `loaded_at`.
- **`DeploymentRepository`** exposes `latest(env)`, `append(event)` and `history(env, limit)`.
- **`RegistryClient`** wraps MLflow (`load_model`, `get_version_metadata`), so tests use a fake and the registry backend can change without touching the API.
- **Logging:** one structured stdout line per prediction with model version and latency.

## 11. Streamlit UI

Three pages cover the whole workflow; the UI holds no training code and no SQL, only calls to services and HTTP clients.

| Page | What the data scientist does | Talks to |
| --- | --- | --- |
| Experiments | Fill the new-experiment form (algorithm, feature toggle, target transform, fixed or search, random or grid, `n_iter`, prefilled params, candidate count shown before launch) · watch the jobs table with progress bars · browse the leaderboard (CV RMSE mean ± std, test RMSE, MAE, R², algorithm, key params; best run highlighted) · open a run's details · register a run · open the run in the MLflow UI | Jobs service (app.db), MLflow client |
| Registry | Browse registered models and their versions with tags and aliases, "best" badge · set, move or remove an alias | MLflow client |
| Deployments | See the current model and `in_sync` · deploy a chosen name and version · roll back or redeploy from history · send a test prediction from a form prefilled with a sample row and see which version answered | API only |

Form polish is out of scope. Errors from the API (422, 409, 503) are shown as-is with their reason.

## 12. Repo structure, tooling and Docker Compose

One Python package and one image run all four services; packages are split by responsibility, so the UI, worker and API import the same schema and services.

```text
ml-platform/
├── app/
│   ├── settings.py             # pydantic-settings: paths, URLs, seeds, limits
│   ├── ml/
│   │   ├── schema.py           # features, target, units, sample rows
│   │   ├── data.py             # load, hash, split, folds
│   │   ├── algorithms.py       # fixed list, defaults, search spaces
│   │   ├── pipeline.py         # build_pipeline(config)
│   │   └── training.py         # candidates, CV, refit, test scoring
│   ├── jobs/
│   │   ├── config.py           # ExperimentConfig
│   │   ├── service.py          # submit, list, get
│   │   ├── runner.py           # run_job(job_id): training + MLflow logging
│   │   └── worker.py           # polling loop, orphan recovery
│   ├── db/
│   │   ├── tables.py           # SQLAlchemy models
│   │   ├── session.py          # engine, WAL pragma, create_all
│   │   └── repositories.py     # JobRepository, DeploymentRepository
│   ├── registry/client.py      # RegistryClient (MLflow wrapper)
│   ├── api/
│   │   ├── main.py             # app factory, startup reconcile
│   │   ├── routes_public.py    # /health /model /predict
│   │   ├── routes_admin.py     # /deploy /reload /deployments
│   │   └── model_manager.py
│   ├── ui/
│   │   ├── app.py
│   │   └── pages/              # experiments.py registry.py deployments.py
│   └── cli.py                  # submit, seed
├── configs/                    # example experiment YAMLs
├── data/training_data.csv
├── docs/
│   ├── architecture-production.md
│   └── decisions/              # ADR-001 …
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml, uv.lock
└── README.md
```

**Tooling:** Python 3.12, `uv` with a committed `uv.lock`, `ruff` for lint and format, `pytest`. Tables are created with `create_all` at startup; Alembic migrations are named in the production doc. One image for all services guarantees the worker and the API run the same scikit-learn version, which joblib/pickle artifacts depend on.

**Makefile targets:** `install`, `lint`, `format`, `test`, `up`, `down`, `logs`, `train CONFIG=…`, `seed` (one job per algorithm, then register the best of each).

| Service | Command | Port | Volumes |
| --- | --- | --- | --- |
| `mlflow` | `mlflow server` with SQLite backend, `--artifacts-destination`, `--serve-artifacts` | 5000 | `mlflowdata` |
| `api` | `uvicorn app.api.main:app --workers 1` | 8000 | `appdata` |
| `worker` | `python -m app.jobs.worker` | – | `appdata` |
| `ui` | `streamlit run app/ui/app.py` | 8501 | `appdata` |

`api`, `worker` and `ui` wait for a healthy `mlflow`. Settings come from env vars: `MLFLOW_TRACKING_URI=http://mlflow:5000`, `API_URL=http://api:8000`, `APP_DB_URL=sqlite:////data/app.db`. The training CSV is copied into the image.

## 13. Tests

Tests target the four agreed risks: leakage, failed reloads, the API contract and persistence. They run with `make test` in seconds, on a small sample of the CSV, a temporary SQLite file and a fake `RegistryClient`.

| Area | Test | What it proves |
| --- | --- | --- |
| Pipeline | After fitting on a subset, the scaler's `mean_` equals that subset's mean, not the full data's | Preprocessing is fit only on training data |
| Pipeline | With `log1p`, predictions and metrics are on the original target scale | Notebook bug 3 cannot recur |
| Pipeline | `BedroomsPerRoom` with zero rooms yields finite values | Engineered feature is safe at inference |
| Config | Unknown param rejected; grid above the cap rejected; `fixed` gives 1 candidate; random gives `n_iter` | Bad jobs fail at submit, not in the worker |
| Jobs | A second claim of the same job changes 0 rows; startup marks orphaned `running` jobs `failed`; an exception in `run_job` records `failed` with the error | Worker state machine is sound |
| Jobs (integration) | `run_job` on the sample with MLflow in a temp dir produces a parent run, child runs and a logged model | Training path works end to end |
| ModelManager | A model failing validation leaves the current model in place; a signature mismatch is rejected | Safe switching |
| `/deploy` | Failure writes no event; success writes one event and `/model` reports the new version; redeploying the loaded version writes nothing | DB and memory stay consistent |
| `/reload` | No-op when in sync; loads the DB's version when it differs | Reconcile works |
| `/predict` | Missing, extra or wrongly typed field returns 422; no model returns 503; response carries the model version | Consumer contract |
| Repository | `latest` returns the highest id per environment; `history` is newest first | Deployment pointer is correct |

## 14. Deliverables

The private repo contains the code, a README, a production architecture doc with Mermaid diagrams, and short decision records.

**README.md:** quickstart (`make up`, then the three URLs), the six-step demo walkthrough from section 1, the notebook issues table from section 2, a design summary, known limitations, and what was cut for time.

**docs/architecture-production.md:** a Mermaid component diagram, a Mermaid sequence diagram of a production rollout, and this mapping from each local piece to its production counterpart:

| Concern | Local proof of concept | Production |
| --- | --- | --- |
| App metadata | SQLite + `create_all` | Managed PostgreSQL + Alembic migrations |
| Tracking and registry | MLflow server, SQLite backend | MLflow on PostgreSQL, or a managed registry |
| Artifacts | Volume behind MLflow's artifact proxy | S3 or equivalent, versioned, sha256 checked at load |
| Training | One worker polling `training_jobs` | Job queue or Kubernetes Jobs, autoscaled workers, resource limits |
| Serving | One uvicorn process; reconcile at startup and on `/reload` | N replicas, each reconciling periodically or on a pub/sub event; canary or rolling rollout |
| Deploy control | `/deploy` on the serving API | Separate control-plane service with authorization, approval and audit |
| Auth | None | SSO for the UI; API keys or OAuth for consumers; admin routes internal only |
| Input validation | Types only | Plausible ranges and schema drift checks |
| Observability | Stdout logs | Structured logs, metrics (latency, errors, prediction distribution), tracing, prediction logging for drift |
| Environments | `local` only | dev, staging, prod with promotion gates |
| CI/CD | `make lint test` by hand | CI on every PR, image build, automated gate: candidate must beat the current model's CV RMSE |
| Data | CSV in repo, sha256 | Versioned datasets in object storage with validation |

**docs/decisions/** — one short record each (context, decision, consequences):

1. MLflow for tracking and registry instead of custom tables.
2. A worker polling a jobs table for background training.
3. `/deploy` validates before committing; the API is the only writer of deployment events.
4. The DB is the source of truth; reconcile on startup and `/reload`, with no periodic check in the proof of concept.
5. Fixed test split and k-fold CV for selection; the test set is scored once per run.
6. Persist the full pipeline, including feature steps and target transform, against one shared schema.
7. Deployments pin exact versions; aliases are labels only.
8. SQLite behind a repository layer now, PostgreSQL later.

## 15. Time plan (4 hours)

The plan is tight, so each slot has a concrete exit check and the documentation slot is protected.

| Slot | Min | Work | Exit check |
| --- | --- | --- | --- |
| 0:00–0:20 | 20 | uv project, settings, schema, Makefile, Dockerfile stub | `make test` runs |
| 0:20–1:05 | 45 | `ml/`: data, algorithms, pipeline, CV and search; pipeline and config tests | A search on the sample returns a best candidate |
| 1:05–1:35 | 30 | `db/` and `jobs/`: tables, repositories, submit, `run_job` with MLflow logging, worker, CLI | `make train` produces a parent run with a logged model |
| 1:35–2:15 | 40 | API: `RegistryClient`, `ModelManager`, endpoints, deploy and reload, API tests | `curl` deploy then predict works |
| 2:15–2:55 | 40 | Streamlit: three pages | Demo path works in the browser |
| 2:55–3:20 | 25 | Compose: four services, healthchecks, `seed` | `make up && make seed` works from a clean clone |
| 3:20–4:00 | 40 | README, production doc | Docs committed |

- **Write decision records as you go,** about 5 minutes each, at the moment the decision lands in code. Leaving all eight for the last slot will not fit.
- **Cut order if a slot overruns by more than 10 minutes:** MLflow UI link → `dummy` baseline → `seed` command → CLI `--wait` → run detail view. The docs slot never drops below 30 minutes.

## 16. Assumptions and items to verify

The defaults below were not explicitly decided; change any of them before implementation starts.

**Assumed defaults**

- 5 CV folds, 80/20 split, seed 42 everywhere.
- Default registered model name `california-housing`; free text allowed.
- `MAX_CANDIDATES` = 200 for grids; default `n_iter` = 20 for random search.
- `dummy` baseline is in the algorithm list (Should).
- A run can be registered once per model name.
- One training job runs at a time.
- Python 3.12.

**Verify during implementation**

- [ ] `training_data.csv` has the 8 features plus `MedHouseVal` and matches the scikit-learn dataset (20,640 rows); confirm the target cap before writing it in the README.
- [ ] Pinned MLflow version: argument names of `mlflow.sklearn.log_model` (MLflow 3 changed some), alias API, the model URI returned for registration, and `--serve-artifacts` behaviour.
- [ ] The pinned Streamlit version supports periodic refresh of a page fragment for the jobs table.
- [ ] Random forest artifact size with unlimited depth; if it is very large, cap `max_depth` in the default search space.
- [ ] SQLite in WAL mode on the shared named volume works with three containers on your machine.
