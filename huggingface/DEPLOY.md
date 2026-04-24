# Deploying the ChatSolveAI backend to Hugging Face Spaces

This replaces the Render deployment. Streamlit Community Cloud (frontend) is
left untouched — only the `API_URL` secret is updated at the end.

## 1. Create the Space

1. Go to https://huggingface.co/new-space
2. **Owner**: your HF username
3. **Space name**: `chatsolveai-api` (example — resulting URL will be
   `https://<username>-chatsolveai-api.hf.space`)
4. **License**: MIT
5. **SDK**: choose **Docker** → **Blank**
6. **Hardware**: CPU basic (free)
7. **Visibility**: Public
8. Click **Create Space**

## 2. Push the code to the Space repo

The Space is a separate git repo at `https://huggingface.co/spaces/<username>/chatsolveai-api`.

```bash
# From the project root on your machine:
git clone https://huggingface.co/spaces/<username>/chatsolveai-api hf-space
cd hf-space

# Copy the files the backend needs (Dockerfile + app code + data).
# Do NOT copy the .env file.
cp ../Dockerfile                  .
cp ../requirements.api.txt        .
cp ../chatbot_responses.json      .
cp ../predefined_responses.json   .
cp ../knowledge_base.csv          .
cp ../processed_queries.csv       .
cp -r ../api                      .
cp -r ../pipeline                 .

# HF reads the YAML frontmatter from README.md — use the prepared template:
cp ../huggingface/README.md       README.md

git add .
git commit -m "Initial deploy: ChatSolveAI FastAPI backend"
git push
```

HF will immediately start building the Docker image. Watch the **Logs** tab
on the Space page. First build ≈ 5–8 minutes (faiss-cpu + langchain).

## 3. Set required secrets

In the Space UI → **Settings** → **Variables and secrets** → **New secret**:

| Key              | Value                                                                          |
|------------------|--------------------------------------------------------------------------------|
| `OPENAI_API_KEY` | `sk-...`                                                                       |
| `MONGO_URL`      | `mongodb+srv://<user>:<password>@cluster1.pr6gog2.mongodb.net/chatsolveai?retryWrites=true&w=majority&appName=Cluster1` |

> Use database name **`chatsolveai`** (matches `DB_NAME` in `api/database.py`).
> After adding secrets, click **Restart this Space** so the container picks
> them up.

## 4. Verify the Space

Once the build shows **Running**, test from your laptop:

```bash
SPACE=https://<username>-chatsolveai-api.hf.space

curl -s $SPACE/health
# → {"status":"ok"}

curl -s -X POST $SPACE/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"test-1","query":"How do I reset my password?"}'
# → {"session_id":"test-1","query":"...","answer":"...","source_documents":[...]}

curl -s $SPACE/analytics
# → {"total_sessions":1,"total_queries":1,...}
```

If `/chat` returns 500, check the Space **Logs** for a Mongo traceback —
usually a typo in `MONGO_URL` or the Atlas user's password.

## 5. Point Streamlit at the new backend

1. https://share.streamlit.io → your ChatSolveAI app → **Settings** → **Secrets**.
2. Replace the old value with:

   ```toml
   API_URL = "https://<username>-chatsolveai-api.hf.space"
   ```
3. **Reboot app**.

The sidebar should now say ✅ *API connected*. First query after a long idle
may take ~20–30s while the Space wakes up — the new health check handles
that gracefully.

## 5b. Re-deploying after new features land

When you pull a new `main` that adds backend modules (e.g. `slowapi`,
`/feedback`, `/suggest`, analytics), push them to the Space:

```bash
cd hf-space

# Copy new / changed files from the main repo
cp ../Dockerfile                  .
cp ../requirements.api.txt        .
cp -r ../api                      .        # picks up middleware.py + new routes
cp -r ../pipeline                 .        # picks up intent_lite.py

git add .
git commit -m "Add advanced features: feedback, suggest, analytics, rate limit"
git push
```

HF rebuilds automatically. No secrets change needed.

## 6. (Optional) Decommission Render

Once the HF deployment is verified:
- Render dashboard → `chatsolve-api` service → **Settings** → **Suspend**
  (keeps config if you ever want it back) or **Delete Service**.

## Troubleshooting

| Symptom                                           | Likely cause                                       |
|---------------------------------------------------|----------------------------------------------------|
| Space build fails on `faiss-cpu`                  | HF build ran out of RAM — retry, or upgrade tier   |
| `/health` 200 but `/chat` 500                     | `MONGO_URL` wrong, or Atlas IP allowlist missing `0.0.0.0/0` |
| Streamlit still says "unreachable"                | Browser cached old `API_URL` — hard reload (⌘⇧R) |
| `/chat/stream` hangs                              | Some HF proxies buffer SSE — test `/chat` first    |
