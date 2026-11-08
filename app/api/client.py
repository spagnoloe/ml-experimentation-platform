"""An HTTP client for the serving API: what the UI's Deployments page and `seed` call.

Responses are parsed into the API's own response models, so the two cannot drift apart.
"""

from typing import Any

import httpx
from pydantic import BaseModel

from app.api.routes_admin import DeploymentOut
from app.api.routes_public import ModelStatusOut, PredictResponse
from app.api.schemas import LoadedModelOut
from app.registry.model_version import ModelVersionRef

TIMEOUT_SECONDS = 60


class ApiError(Exception):
    """The API refused a request (`status_code` set) or could not be reached (`None`).

    The message is the API's own reason (`detail`), unchanged.
    """

    def __init__(self, status_code: int | None, reason: str) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason

    def __str__(self) -> str:
        if self.status_code is None:
            return self.reason
        return f"{self.status_code}: {self.reason}"


class ApiClient:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    @classmethod
    def at(cls, base_url: str) -> "ApiClient":
        return cls(httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS))

    def model_status(self) -> ModelStatusOut:
        return self._call("GET", "/v1/model", ModelStatusOut)

    def deploy(self, ref: ModelVersionRef) -> LoadedModelOut:
        body = {"model_name": ref.model_name, "version": ref.model_version}
        return self._call("POST", "/deploy", LoadedModelOut, json=body)

    def deployments(self, limit: int = 50) -> list[DeploymentOut]:
        found = self._request("GET", "/deployments", params={"limit": limit})
        return [DeploymentOut.model_validate(d) for d in found]

    def predict(self, rows: list[dict[str, float]]) -> PredictResponse:
        return self._call("POST", "/v1/predict", PredictResponse, json={"rows": rows})

    def _call[M: BaseModel](self, method: str, path: str, model: type[M], **kwargs: Any) -> M:
        return model.model_validate(self._request(method, path, **kwargs))

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(None, f"cannot reach the API at {self._http.base_url}: {exc}") from exc
        if response.is_error:
            raise ApiError(response.status_code, _detail(response))
        return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        return response.text
    if isinstance(detail, list):
        # FastAPI's own validation errors: one `{loc, msg}` per problem.
        return "; ".join(_validation_error(error) for error in detail)
    return detail if isinstance(detail, str) else str(detail)


def _validation_error(error: Any) -> str:
    if not isinstance(error, dict):
        return str(error)
    where = ".".join(str(part) for part in error.get("loc", ()))
    return f"{where}: {error.get('msg', '')}" if where else str(error.get("msg", error))
