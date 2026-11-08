# A worker polling a Postgres jobs table instead of a message broker

Submitting an Experiment Config inserts a `queued` row into `training_jobs`. A single worker process polls every `WORKER_POLL_SECONDS` (2s) for the oldest queued row and claims it with a conditional update, `UPDATE … SET status='running' WHERE id=:id AND status='queued'`. It runs the job only if exactly one row changed. The same row carries phase, progress, the MLflow run id and the error, so the UI and the CLI read job state from the one place the worker writes it.

We chose this over Celery/RQ with Redis or RabbitMQ because the proof of concept runs one job at a time on a laptop. Postgres is already there, and a broker would add a service, a second source of truth for job state, and retry semantics we do not need.

## Considered Options

- **Celery or RQ with Redis**: real queues, retries and concurrency, but job status would live in the broker's result backend and still need mirroring into a table for the UI.
- **`SELECT … FOR UPDATE SKIP LOCKED` claim**: the right shape for many workers. With one worker the conditional update is enough, and it is the simplest statement to test.
- **Run training in the API or UI process**: no queue at all, but a long fit blocks requests, and a crash loses the job silently.

## Consequences

- Exactly one worker is supported. On startup it marks every `running` job, and its MLflow Run, failed with `worker restarted`, which is only correct because no other worker can own them.
- Every status update after the claim is guarded on `status = 'running'`, so a job already failed as an orphan cannot be revived by a late write.
- Jobs run in submission order, one at a time. A 2-second poll adds at most 2 seconds of latency.
- In production the table becomes an SQS queue feeding SageMaker Training Jobs or EKS Jobs. The `submit` service and the job row's role as the status record stay the same.
