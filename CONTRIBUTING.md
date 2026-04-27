# Contributing

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.api.txt -r requirements.streamlit.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before running the full backend. Local MongoDB
can be started with:

```bash
docker run -d --name chatsolveai-mongo -p 27017:27017 mongo:7
```

## Running Locally

Start the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
streamlit run app.py
```

## Checks Before a PR

```bash
pytest -q
python3 -m compileall api pipeline tests app.py
```

When changing prompts, retrieval, or canonical answers, add or update regression
tests so expected behavior is explicit.

## Pull Request Guidelines

- Keep PRs focused and independently reviewable.
- Do not commit real secrets, `.env`, notebook outputs, or local cache files.
- Update `.env.example` when adding configuration.
- Update `README.md` or `CHANGELOG.md` when user-facing behavior changes.
- Prefer existing project patterns over introducing new frameworks.

## Deployment Notes

Backend deployment depends on Hugging Face Space secrets. Frontend deployment
depends on Streamlit Cloud secrets. Coordinate secret changes before merging PRs
that require new environment variables.
