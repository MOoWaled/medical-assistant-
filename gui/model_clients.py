"""Grounded adapters for the baseline, Qwen, and Mistral providers."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


DEFAULT_BASELINE_URL = "http://localhost:5000"
DEFAULT_QWEN_URL = ""
DEFAULT_MISTRAL_URL = ""
REQUEST_TIMEOUT_SECONDS = 90
QWEN_TIMEOUT_SECONDS = 45


class ModelRequestError(RuntimeError):
    """An actionable, safe-to-display error from a model provider."""


def analyse_symptoms(
    *,
    model: str,
    symptoms: str,
    baseline_url: str,
    qwen_url: str,
    mistral_url: str,
    mistral_api_key: str,
    age: int | None = None,
    gender: str | None = None,
) -> dict[str, Any]:
    """Retrieve first, then let a provider select only from the known candidates."""
    if model == "logistic_baseline":
        return _call_baseline(symptoms, baseline_url, age=age, gender=gender)
    if model in {"deep_rnn", "deep_lstm"}:
        return _call_deep_learning(
            symptoms,
            baseline_url,
            model.removeprefix("deep_"),
            age=age,
            gender=gender,
        )

    # These imports are needed only for the optional LLM paths. Keeping them
    # lazy lets the local-only GUI open cleanly before the local API is started.
    from rag_llm_API.retriever import grounded_result, match_candidate

    candidates = _retrieve_candidates(symptoms)
    if model == "qwen":
        selected_name = _call_qwen(symptoms, candidates, qwen_url, age=age, gender=gender)
    elif model == "mistral":
        selected_name = _call_mistral(
            symptoms, candidates, mistral_url, mistral_api_key, age=age, gender=gender
        )
    else:
        raise ModelRequestError(f"Unsupported model: {model}")

    # A provider can select a candidate but can never introduce a new condition or text.
    selected = match_candidate(selected_name, candidates) or candidates[0]
    result = grounded_result(selected, candidates=candidates)
    result["model_selection"] = selected_name or "No valid selection; used the top retrieved condition."
    result["patient_context"] = {"age": age, "gender": gender}
    return result


def _retrieve_candidates(symptoms: str) -> list[dict[str, Any]]:
    try:
        from rag_llm_API.retriever import get_retriever

        candidates = get_retriever().search(symptoms, top_k=3)
    except Exception as error:
        raise ModelRequestError(
            "Could not load the local RAG index. Run `python rag_llm_API/build_faiss_index.py` and try again. "
            f"Details: {error}"
        ) from error
    if not candidates:
        raise ModelRequestError("No matching condition was found in the project knowledge base.")
    return candidates


def _call_baseline(
    symptoms: str, base_url: str, *, age: int | None, gender: str | None
) -> dict[str, Any]:
    endpoint = _require_url(base_url, "Logistic Baseline") + "/predict"
    try:
        response = requests.post(
            endpoint,
            json={"text": symptoms, "age": age, "gender": gender},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise ModelRequestError(f"Could not reach the Logistic Baseline API: {error}") from error
    payload = _json_response(response, "Logistic Baseline")
    if response.status_code >= 400:
        raise ModelRequestError(_error_message(payload, response, "Logistic Baseline"))

    details = payload.get("medical_details") or {}
    return {
        "condition": str(details.get("condition") or payload.get("predicted_condition") or "Not available"),
        "symptoms": str(details.get("symptoms") or symptoms),
        "causes": str(details.get("causes") or "Not available"),
        "warnings": str(details.get("warnings") or "Not available"),
        "recommendations": str(details.get("recommendations") or "Not available"),
        "confidence": payload.get("confidence"),
        "retrieval_candidates": payload.get("retrieval_candidates", []),
        "patient_context": payload.get("patient_context", {"age": age, "gender": gender}),
    }


def _call_deep_learning(
    symptoms: str,
    base_url: str,
    deep_model: str,
    *,
    age: int | None,
    gender: str | None,
) -> dict[str, Any]:
    endpoint = _require_url(base_url, "Deep Learning") + "/predict/deep-learning"
    try:
        response = requests.post(
            endpoint,
            json={"text": symptoms, "deep_model": deep_model, "age": age, "gender": gender},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise ModelRequestError(f"Could not reach the Deep Learning API: {error}") from error
    payload = _json_response(response, "Deep Learning")
    if response.status_code >= 400:
        raise ModelRequestError(_error_message(payload, response, "Deep Learning"))
    details = payload.get("medical_details") or {}
    return {
        "condition": str(details.get("condition") or payload.get("predicted_condition") or "Not available"),
        "symptoms": str(details.get("symptoms") or symptoms),
        "causes": str(details.get("causes") or "Not available"),
        "warnings": str(details.get("warnings") or "Not available"),
        "recommendations": str(details.get("recommendations") or "Not available"),
        "confidence": payload.get("confidence"),
        "retrieval_candidates": payload.get("retrieval_candidates", []),
        "patient_context": payload.get("patient_context", {"age": age, "gender": gender}),
        "training_metrics": payload.get("training_metrics", {}),
        "selection_mode": payload.get("selection_mode", ""),
    }


def _candidate_context(candidates: list[dict[str, Any]]) -> str:
    sections = []
    for number, candidate in enumerate(candidates, start=1):
        sections.append(
            "\n".join(
                [
                    f"Candidate {number}",
                    f"Condition: {candidate['condition']}",
                    f"Symptoms: {candidate['symptoms'][:1600]}",
                    f"Causes: {candidate['causes'][:700]}",
                    f"Warnings: {candidate['warnings'][:700]}",
                    f"Recommendations: {candidate['recommendations'][:700]}",
                ]
            )
        )
    return "\n\n".join(sections)


def _selection_prompt(
    symptoms: str,
    candidates: list[dict[str, Any]],
    *,
    age: int | None,
    gender: str | None,
) -> tuple[str, str]:
    system = (
        "You are a retrieval-grounded medical condition selector. Use only the candidates in the supplied "
        "knowledge-base context. Do not diagnose, invent a condition, paraphrase a condition name, or add facts. "
        "Choose the one candidate whose stored symptoms best match the patient text. "
        "Return only valid JSON in exactly this shape: {\"condition\": \"exact candidate condition name\"}.\n\n"
        "KNOWLEDGE-BASE CANDIDATES:\n"
        f"{_candidate_context(candidates)}"
    )
    context_parts = []
    if age is not None:
        context_parts.append(f"Age: {age}")
    if gender:
        context_parts.append(f"Gender: {gender}")
    context = "\n".join(context_parts) or "Not provided"
    user = (
        f"PATIENT SYMPTOMS:\n{symptoms}\n\n"
        f"OPTIONAL PATIENT CONTEXT:\n{context}\n\n"
        "Use this context only when choosing among the supplied candidates. Do not add demographic facts "
        "or alter retrieved symptoms, causes, warnings, or recommendations."
    )
    return system, user


def _call_qwen(
    symptoms: str,
    candidates: list[dict[str, Any]],
    base_url: str,
    *,
    age: int | None,
    gender: str | None,
) -> str:
    root = _require_url(base_url, "Qwen")
    system, user = _selection_prompt(symptoms, candidates, age=age, gender=gender)
    root = re.sub(r"/api/(generate|chat)?$", "", root.rstrip("/"))
    headers = {"ngrok-skip-browser-warning": "true", "Content-Type": "application/json"}
    model_name = os.getenv("QWEN_MODEL", "qwen3:4b")
    attempts = [
        (
            f"{root}/api/chat",
            {
                "model": model_name,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            },
        ),
        (
            f"{root}/api/generate",
            {
                "model": model_name,
                "prompt": f"{system}\n\n{user}",
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            },
        ),
    ]
    failures: list[str] = []
    for endpoint, body in attempts:
        try:
            response = requests.post(endpoint, json=body, headers=headers, timeout=QWEN_TIMEOUT_SECONDS)
        except requests.RequestException as error:
            failures.append(f"{endpoint}: {error}")
            continue
        try:
            payload = _json_response(response, "Qwen")
        except ModelRequestError as error:
            failures.append(str(error))
            continue
        if response.status_code in (404, 405):
            failures.append(_error_message(payload, response, "Qwen"))
            continue
        if response.status_code >= 400:
            raise ModelRequestError(_error_message(payload, response, "Qwen"))
        return _selected_condition(_response_text(payload), "Qwen", candidates)

    detail = " | ".join(failures)
    raise ModelRequestError(
        "Could not reach a working Qwen/Ollama endpoint. Restart the Qwen server, create a fresh ngrok URL, "
        f"and update the Qwen API URL field. Details: {detail}"
    )


def _call_mistral(
    symptoms: str,
    candidates: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    *,
    age: int | None,
    gender: str | None,
) -> str:
    root = _require_url(base_url, "Mistral")
    system, user = _selection_prompt(symptoms, candidates, age=age, gender=gender)
    official_api = "api.mistral.ai" in root.lower() or root.rstrip("/").endswith("/chat/completions")

    if official_api:
        endpoint = root.rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint += "/chat/completions"
        elif not endpoint.endswith("/chat/completions"):
            endpoint += "/v1/chat/completions"
        if not api_key.strip():
            raise ModelRequestError("A Mistral API key is required for the official Mistral endpoint.")
        headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
        body = {
            "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    else:
        # Custom Kaggle/Flask + ngrok server contract: POST /generate {prompt, max_length}.
        endpoint = root.rstrip("/")
        if not endpoint.endswith("/generate"):
            endpoint += "/generate"
        headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        body = {"prompt": f"{system}\n\n{user}", "max_length": 160, "temperature": 0}

    try:
        response = requests.post(endpoint, json=body, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        raise ModelRequestError(f"Could not reach the Mistral API at {endpoint}: {error}") from error
    payload = _json_response(response, "Mistral")
    if response.status_code >= 400:
        raise ModelRequestError(_error_message(payload, response, "Mistral"))
    return _selected_condition(_response_text(payload), "Mistral", candidates)


def _response_text(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"])
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
    for key in ("response", "generated_text", "text", "output"):
        if payload.get(key):
            return str(payload[key])
    return ""


def _selected_condition(answer: str, model_name: str, candidates: list[dict[str, Any]]) -> str:
    data = _extract_json(answer)
    condition = data.get("condition") if isinstance(data, dict) else None
    if isinstance(condition, str) and condition.strip():
        return condition.strip()

    # Some Ollama deployments ignore JSON-mode instructions. Accept only an exact
    # condition name already retrieved from the local knowledge base.
    normalised_answer = re.sub(r"[^a-z0-9]+", "", str(answer).lower())
    for candidate in candidates:
        normalised_candidate = re.sub(r"[^a-z0-9]+", "", candidate["condition"].lower())
        if normalised_candidate and normalised_candidate in normalised_answer:
            return candidate["condition"]
    raise ModelRequestError(f"{model_name} did not return a valid candidate condition JSON object.")


def _extract_json(answer: Any) -> dict[str, Any] | None:
    if not isinstance(answer, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    for candidate in [*reversed(fenced), cleaned]:
        start = candidate.find("{")
        if start == -1:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _require_url(url: str, model_name: str) -> str:
    if not url or not url.strip():
        raise ModelRequestError(f"Enter a valid {model_name} API URL in the sidebar before submitting.")
    return url.strip().rstrip("/")


def _json_response(response: requests.Response, model_name: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        body = response.text[:400].replace("\n", " ")
        raise ModelRequestError(f"{model_name} returned HTTP {response.status_code} with a non-JSON response: {body}") from error
    if not isinstance(payload, dict):
        raise ModelRequestError(f"{model_name} returned an unexpected JSON response.")
    return payload


def _error_message(payload: dict[str, Any], response: requests.Response, model_name: str) -> str:
    detail = payload.get("error") or payload.get("detail") or payload.get("message") or str(payload)
    return f"{model_name} returned HTTP {response.status_code}: {detail}"
