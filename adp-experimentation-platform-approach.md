# ML Platform: Implementation Plan

## 1. Goal

Build a local proof of concept for an ML platform that supports the
model lifecycle:

1.  Run and track training experiments.
2.  Compare model performance and register trained models.
3.  Select a registered model for deployment.
4.  Expose the selected model through an inference API.
5.  Test predictions through a lightweight UI.

The implementation should be simple enough to complete within the
assignment's time constraints, while keeping clear boundaries between
components so the system can evolve towards production.

## 2. Proposed architecture

The application has three main parts:

-   **Streamlit UI:** the interface for experimentation and deployment
    management.
-   **FastAPI service:** the interface for model inference and model
    reloading.
-   **Persistence and artifact storage:** SQLite stores metadata; the
    filesystem stores serialized model artifacts.

The Streamlit UI has two tabs:

-   **Experimentation:** configure, run, inspect and compare
    experiments.
-   **Deployments:** browse registered models, select the model to
    serve, inspect the active deployment and test predictions.

The UI is the control plane. FastAPI is the inference service. Consumers
of the model interact with FastAPI and do not need to know how
experiments were run or which model implementation is currently
selected.

### Component overview

``` text
                 +---------------------------+
                 |       Streamlit UI        |
                 |                           |
                 |  Experimentation | Deploy |
                 +-----------+---------------+
                             |
                 +-----------+---------------+
                 |                           |
         +-------v--------+          +-------v--------+
         | Experiment     |          | Deployment     |
         | workflow       |          | workflow       |
         +-------+--------+          +-------+--------+
                 |                           |
                 +-------------+-------------+
                               |
                       +-------v-------+
                       |    SQLite     |
                       |               |
                       | Experiments   |
                       | Models        |
                       | Deployments   |
                       +---------------+

                 +---------------------------+
                 |         FastAPI           |
                 |                           |
                 | GET /health               |
                 | GET /model                |
                 | POST /predict             |
                 | POST /reload              |
                 +-------------+-------------+
                               |
                       +-------v-------+
                       | Model Manager |
                       +-------+-------+
                               |
                       +-------v-------+
                       | Model artifacts|
                       | Local filesystem|
                       +---------------+
```

The UI and API share access to the SQLite database and artifact
directory in the local proof of concept. Keep database access behind a
persistence/repository layer rather than scattering SQL queries
throughout the application.

## 3. Technology choices

  -----------------------------------------------------------------------
  Area                    Local proof of concept  Possible production
                                                  evolution
  ----------------------- ----------------------- -----------------------
  UI                      Streamlit               Streamlit or a
                                                  dedicated frontend

  ML                      scikit-learn            scikit-learn or other
                                                  frameworks

  API                     FastAPI                 FastAPI

  Database                SQLite                  PostgreSQL

  ORM / persistence       SQLAlchemy              SQLAlchemy

  Model serialization     joblib                  joblib or an
                                                  appropriate
                                                  framework-specific
                                                  format

  Artifact storage        Local filesystem        S3 or equivalent object
                                                  storage

  Runtime                 Docker Compose          Managed containers or
                                                  Kubernetes
  -----------------------------------------------------------------------

These production technologies are possible directions, not requirements
for the local implementation.

## 4. Experimentation workflow

The experimentation tab should allow a data scientist to:

-   Choose an algorithm and configure its hyperparameters.
-   Launch a training experiment.
-   View experiment status and results.
-   Compare completed experiments using evaluation metrics.
-   Inspect an experiment's parameters, metrics and model details.
-   Register a trained model so it can be considered for deployment.

The initial implementation can focus on the algorithms in the supplied
notebook:

-   Linear Regression
-   Random Forest Regressor
-   Gradient Boosting Regressor

Use the California Housing dataset and its `MedHouseVal` target. Keep
the dataset and feature definitions in one place so training and
inference use a consistent schema.

Use a scikit-learn `Pipeline` to bundle preprocessing and the estimator.
Fit preprocessing only on the training data, and persist the complete
pipeline so inference applies the same transformations.

Track at least:

-   Experiment ID and name
-   Creation timestamp and status
-   Algorithm and hyperparameters
-   Dataset identifier or version
-   Evaluation metrics, such as RMSE, MAE and R²
-   Model artifact location, when a model is produced

For the proof of concept, experiment tracking can be implemented with
the application's own database tables. A dedicated experiment-tracking
product is not necessary.

## 5. Model registry and artifact storage

A model is registered separately from an experiment. An experiment
records a training run; a registered model is a persisted artifact that
can be selected for deployment.

Store model artifacts on the local filesystem. Use `joblib` for the
scikit-learn pipeline unless another format is justified.

A possible directory structure:

``` text
artifacts/
├── model_001/
│   ├── model.joblib
│   └── metadata.json
├── model_002/
│   ├── model.joblib
│   └── metadata.json
└── model_003/
    ├── model.joblib
    └── metadata.json
```

The registry stores metadata and the artifact path. The artifact itself
is not stored in SQLite.

For this small assignment, artifacts can be committed to the repository
if their size is manageable. In a production system, store them in
object storage such as S3. Git should not become the long-term model
artifact store.

Persist the full preprocessing and prediction pipeline, not only the
estimator. This avoids requiring the serving API to reproduce
training-time preprocessing separately.

## 6. Database design

Use SQLite for the local proof of concept. SQLite is an embedded
database, so it does not need its own Docker Compose service.

A minimal schema can use three main tables.

### `experiments`

  Field          Description
  -------------- ------------------------------------
  `id`           Unique experiment identifier
  `name`         Human-readable experiment name
  `algorithm`    Model algorithm
  `parameters`   Hyperparameters, stored as JSON
  `metrics`      Evaluation metrics, stored as JSON
  `dataset`      Dataset identifier or version
  `status`       Running, completed or failed
  `created_at`   Creation timestamp

### `models`

  Field             Description
  ----------------- ----------------------------------------
  `id`              Unique model identifier
  `experiment_id`   Experiment that produced the model
  `name`            Model name
  `version`         Model version
  `artifact_path`   Path or URI of the serialized artifact
  `created_at`      Registration timestamp

### `deployments`

  Field           Description
  --------------- -----------------------------------------
  `id`            Deployment record identifier
  `model_id`      Model selected for deployment
  `environment`   Deployment environment, such as `local`
  `updated_at`    Last update timestamp

For the initial implementation, support one active model per
environment. Enforce that rule in the application and, where practical,
in the database.

Use SQLAlchemy and a repository/data-access layer to keep persistence
details separate from training, UI and serving logic. This should make a
later move from SQLite to PostgreSQL easier.

## 7. Model serving API

FastAPI exposes the currently selected model through a stable inference
interface.

Suggested endpoints:

  Method   Endpoint     Purpose
  -------- ------------ ---------------------------------------------
  `GET`    `/health`    Health check
  `GET`    `/model`     Return information about the loaded model
  `POST`   `/predict`   Generate predictions using the loaded model
  `POST`   `/reload`    Load a specified registered model

The prediction endpoint should accept the expected input features and
return the prediction. Consumers should not need to know whether the
underlying model is Linear Regression, Random Forest or Gradient
Boosting.

Keep the HTTP layer thin. Put artifact loading, model validation and
prediction behind a separate `ModelManager` class or equivalent service.

## 8. Switching the deployed model

The deployments tab lets the data scientist select a registered model.
The selected model should become the model used by subsequent inference
requests.

Do not make Streamlit kill and restart the FastAPI process. Instead, use
a model-reload workflow:

1.  The user selects a registered model in Streamlit.
2.  Streamlit calls the API's internal `/reload` endpoint with the model
    ID.
3.  FastAPI looks up the model's artifact path in the registry.
4.  The model manager loads the new artifact into a temporary variable.
5.  The manager validates that the artifact can be loaded and used for
    prediction.
6.  If validation succeeds, the manager replaces the currently loaded
    model.
7.  The deployment record is updated and the API reports the new active
    model.
8.  If loading or validation fails, the existing model remains active
    and the deployment is not changed.

This avoids restarting the API and reduces the risk of interrupting
inference. For the local proof of concept, a single FastAPI worker is
sufficient. A production implementation with multiple workers or
replicas would need a coordinated rollout mechanism so every serving
instance uses the intended model version.

On application startup, FastAPI should read the active deployment from
the database and load that model. This makes the service recover its
selected model after a container restart.

The `/reload` endpoint should be treated as an internal administrative
operation, not exposed as an unrestricted public endpoint in production.

## 9. Docker Compose and local development

Use Docker Compose to launch the Streamlit UI and FastAPI service
together.

SQLite is a file, not a separate service. Both containers need access to
the same database file and artifact directory. Mount shared volumes for
these paths.

A minimal Compose setup will contain:

-   `ui`: Streamlit application, exposed on port `8501`.
-   `api`: FastAPI application, exposed on port `8000`.
-   Shared volume for the SQLite database.
-   Shared volume for model artifacts.

The containers can use the same image initially, with different startup
commands. They should communicate over the Compose network using the API
service name, rather than `localhost`.

For SQLite, ensure both services point to exactly the same mounted
database file. Keep writes relatively simple because SQLite serializes
concurrent writes. This is adequate for the expected local workload.

## 10. Suggested project structure

``` text
ml-platform/
├── app/
│   ├── ui/
│   │   ├── app.py
│   │   └── pages/
│   │       ├── experimentation.py
│   │       └── deployments.py
│   ├── api/
│   │   └── main.py
│   ├── ml/
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── model_manager.py
│   ├── registry/
│   │   ├── models.py
│   │   └── repository.py
│   └── database/
│       ├── models.py
│       └── connection.py
├── artifacts/
├── data/
├── tests/
├── docker-compose.yml
├── Dockerfile
└── README.md
```

The exact structure can change during implementation. The important
separation is between UI, training logic, persistence, registry
operations and model serving.

The UI should invoke training functionality through a service/module
rather than containing the training implementation itself. This leaves
room to move training into a background worker in a future production
version.

## 11. Production architecture considerations

The local proof of concept should be designed so its main concepts can
be retained in production.

Likely changes include:

-   Replace SQLite with PostgreSQL.
-   Replace local artifact storage with S3 or equivalent object storage.
-   Run the UI and API as separately deployed services.
-   Add authentication and authorization, especially around model
    promotion and deployment operations.
-   Add logging, metrics and tracing for training jobs and inference
    requests.
-   Introduce background workers and a job queue if training becomes
    long-running.
-   Use a coordinated deployment or rollout mechanism when serving
    multiple API replicas.
-   Add model and dataset versioning, artifact integrity checks and
    deployment history.
-   Restrict model artifact loading to trusted artifacts. Serialized
    Python objects can execute code during loading, so artifacts must
    come from a trusted source.

The production design should preserve a stable prediction API for
consumers. Changing the selected model should not require consumers to
change their integration.

## 12. Scope and implementation priorities

The assignment is time-constrained, so prioritize a working end-to-end
flow over extensive infrastructure.

Suggested implementation order:

1.  **Database and persistence:** create the SQLite schema and
    repository functions.
2.  **Training:** implement reproducible training and evaluation for the
    selected algorithms.
3.  **Experiment tracking:** persist experiment parameters, status and
    metrics.
4.  **Model registry:** save artifacts and register their metadata.
5.  **Serving:** implement FastAPI prediction and model-information
    endpoints.
6.  **Model switching:** implement safe model reload and
    deployment-record updates.
7.  **Streamlit UI:** connect the experimentation and deployment
    workflows.
8.  **Docker Compose:** launch the UI and API with shared database and
    artifact volumes.
9.  **Tests and documentation:** test the main workflows and document
    how to run the system.

A successful proof of concept should demonstrate this complete path:

``` text
Run experiment
      |
      v
Record parameters and metrics
      |
      v
Register model artifact
      |
      v
Select model for deployment
      |
      v
FastAPI loads selected model
      |
      v
Consumer sends prediction request
      |
      v
FastAPI returns prediction
```

The central design principle is to keep experimentation, model
registration, deployment selection and inference as separate
responsibilities, connected through explicit interfaces and shared
metadata.
