# Deployments pin exact Model Versions; MLflow aliases are not used

The serving API loads whatever the most recent Deployment names: an exact `(registered model, version)` pair stored as an append-only row in our own Postgres table, written only by `/deploy` after the model has been loaded and validated. We deliberately do not use MLflow registry aliases (e.g. `@champion`), even though that is the MLflow-native way to mark "the model in production", because moving an alias has no validation step, leaves no history of what served when, and can be changed by anyone with registry access outside the API.

## Considered Options

- **Serve `models:/<name>@champion`**: least code, but a bad alias move silently changes what consumers get on the next load, and "what was serving at 14:00?" cannot be answered.
- **Aliases as human labels alongside Deployments**: rejected as a second, non-authoritative pointer that readers would mistake for the serving one.

## Consequences

- Registering a Run and deploying a Model Version are separate actions; registering never changes what is served.
- Switching to an older Model Version is an ordinary Deployment; there is no separate rollback mechanism in the proof of concept.
- The production design keeps this shape: a deploy control-plane writes Deployments, serving replicas reconcile to the latest one.
