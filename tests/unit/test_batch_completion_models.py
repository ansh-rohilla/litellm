import time

import litellm
from litellm.batch_completion.main import batch_completion_models


def test_batch_completion_models_returns_first_completed_not_first_listed(monkeypatch):
    """
    Regression test for issue #44918.
    Verifies batch_completion_models returns the first completed response rather than
    waiting for models in listed order or waiting for slower models to finish.
    """

    def _mock_completion(*args, model=None, **kwargs):
        if model == "slow-model":
            time.sleep(0.4)
            return {"model": model, "content": "slow response"}
        elif model == "fast-model":
            time.sleep(0.02)
            return {"model": model, "content": "fast response"}
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    start = time.perf_counter()
    response = batch_completion_models(
        models=["slow-model", "fast-model"],
        messages=[{"role": "user", "content": "hello"}],
    )
    elapsed = time.perf_counter() - start

    assert response is not None
    assert response["model"] == "fast-model"
    # Ensure it did not block waiting for slow-model
    assert elapsed < 0.35


def test_batch_completion_models_recovers_from_failed_model(monkeypatch):
    """
    Verifies that an error/exception from a fast or earlier model does not abort the helper;
    it should continue and return the first successful completed response.
    """

    def _mock_completion(*args, model=None, **kwargs):
        if model == "failing-model":
            time.sleep(0.01)
            raise RuntimeError("500 Internal Server Error")
        elif model == "success-model":
            time.sleep(0.05)
            return {"model": model, "content": "success response"}
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    response = batch_completion_models(
        models=["failing-model", "success-model"],
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response is not None
    assert response["model"] == "success-model"


def test_batch_completion_models_returns_none_when_all_fail(monkeypatch):
    """
    Verifies that if all models fail, None is returned.
    """

    def _mock_completion(*args, model=None, **kwargs):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    response = batch_completion_models(
        models=["failing-1", "failing-2"],
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response is None


def test_batch_completion_models_accepts_single_string_and_empty_list(monkeypatch):
    called_models = []

    def _mock_completion(*args, model=None, **kwargs):
        called_models.append(model)
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    # Empty list returns None without error
    empty_res = batch_completion_models(models=[], messages=[{"role": "user", "content": "hello"}])
    assert empty_res is None
    assert called_models == []

    # Single string model works
    single_res = batch_completion_models(models="single-model", messages=[{"role": "user", "content": "hello"}])
    assert single_res == {"model": "single-model"}
    assert called_models == ["single-model"]


def test_batch_completion_models_deployments_returns_first_completed(monkeypatch):
    """
    Verifies that the deployments branch also returns the first completed response promptly.
    """

    def _mock_completion(*args, **kwargs):
        model = kwargs.get("model")
        if model == "slow-deployment":
            time.sleep(0.4)
            return {"model": model, "content": "slow"}
        elif model == "fast-deployment":
            time.sleep(0.02)
            return {"model": model, "content": "fast"}
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    start = time.perf_counter()
    response = batch_completion_models(
        deployments=[
            {"model": "slow-deployment", "api_base": "http://slow"},
            {"model": "fast-deployment", "api_base": "http://fast"},
        ],
        model_list=[],
        messages=[{"role": "user", "content": "hello"}],
    )
    elapsed = time.perf_counter() - start

    assert response is not None
    assert response["model"] == "fast-deployment"
    assert elapsed < 0.35


def test_batch_completion_models_deployments_recovers_from_error(monkeypatch):
    """
    Verifies that the deployments branch continues on failure and returns the successful result.
    """

    def _mock_completion(*args, **kwargs):
        model = kwargs.get("model")
        if model == "failing-deployment":
            time.sleep(0.01)
            raise RuntimeError("deployment connection failed")
        elif model == "success-deployment":
            time.sleep(0.05)
            return {"model": model, "content": "success"}
        return {"model": model}

    monkeypatch.setattr(litellm, "completion", _mock_completion)

    response = batch_completion_models(
        deployments=[
            {"model": "failing-deployment"},
            {"model": "success-deployment"},
        ],
        model_list=[],
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response is not None
    assert response["model"] == "success-deployment"
