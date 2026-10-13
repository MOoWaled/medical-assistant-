# Streamlit GUI

This folder contains a standalone Streamlit interface for the three Medical ChatBot models:

- Logistic Baseline — calls `model_API/app.py` at `/predict`.
- Mistral — calls the configured endpoint at `/generate`.
- Qwen — calls the configured Ollama-compatible endpoint at `/api/chat`.

## Run locally

```bash
cd gui
python -m pip install -r requirements.txt
streamlit run app.py
```

Start the Logistic Baseline API first when using that option:

```bash
python model_API/app.py
```

For Mistral and Qwen, enter the public endpoint URL in the Streamlit sidebar. The app sends the symptoms to the selected model and normalises its answer into `condition`, `symptoms`, `causes`, `warnings`, and `recommendations`.

> The interface is for educational purposes and does not provide a medical diagnosis.
