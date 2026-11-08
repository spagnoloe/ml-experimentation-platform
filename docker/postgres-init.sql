-- Runs once, on the first start of an empty pgdata volume.
-- app: our tables. mlflow: MLflow's backend store. app_test: the test suite.
CREATE DATABASE app;
CREATE DATABASE mlflow;
CREATE DATABASE app_test;
