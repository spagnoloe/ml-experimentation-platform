# Our own Streamlit UI and serving API, not MLflow's UI and model serving

The data scientist launches Training Jobs, deploys Model Versions and sends test predictions from our Streamlit UI. Consumers get predictions from our FastAPI serving API. We do not use MLflow's UI or its model serving for either job. MLflow is used for tracking and the registry (ADR-0002), and its UI stays available for inspecting Runs and Candidates in depth. We do not serve with MLflow because open-source MLflow's UI has no action that deploys a Model Version or starts an inference server. MLflow's serving, `mlflow models serve` or the `mlflow deployments` plugins, is driven from the CLI or the SDK. It also lacks what a Deployment needs here: a check before switching, a history of what was served when, and a response that names the Model Version that answered. The MLflow UI also cannot launch a Training Job, which is half of what our UI does.

Checked against MLflow 3 (`mlflow>=3.1`), November 2026.

## Considered Options

- **The MLflow UI as the only UI.** It needs no UI code, and it already compares Runs. But it cannot submit an Experiment Config or show Training Job progress. Deploying would still need a terminal, so the demo path would not run from one place.
- **`mlflow models serve` as the serving API.** It needs no serving code. But each process serves one model URI fixed at startup, so switching Model Versions means restarting it, with downtime and no validation gate. Its `/invocations` route is MLflow's generic contract, not our versioned `/v1/predict`: it has no row limit and does not return `model_version`. Nothing records a Deployment, so it cannot work with ADR-0001.
- **MLflow AI Gateway endpoints.** Since MLflow 3 these can be created from the tracking server's UI. They forward requests to LLM providers, such as OpenAI. They do not serve a Model Version from our registry, so they do not fit this use case.
- **A managed MLflow with a serve button.** Databricks has a "Serve this model" button, but that is Databricks Model Serving, a paid managed service. It breaks the rule that everything runs locally under Docker Compose with no cloud dependencies.

## Consequences

- We own the UI and serving code (`app/ui/`, `app/api/`) and their tests, instead of getting them from MLflow.
- The Streamlit UI links each Run to the MLflow UI instead of rebuilding MLflow's views of runs, child runs and artifacts.
- In production, the serving API can still run on a managed host (EKS or ECS, or a SageMaker endpoint). It keeps our validation, our Deployments table and our `/v1` contract.
