# MARS Deployment

This project is a Flask web app. It needs a server because the simulation calls
the DeepSeek API from the backend.

## Recommended: Render

1. Push this folder to a GitHub repository.
2. In Render, create a new **Web Service** from that repository.
3. Render can use `render.yaml` automatically. If setting it manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300`
4. Add an environment variable in Render:
   - `DEEPSEEK_API_KEY`
5. Deploy and open the Render URL.

## Railway

1. Push this folder to GitHub.
2. Create a Railway service from the repository.
3. Add the environment variable:
   - `DEEPSEEK_API_KEY`
4. Railway should detect the `Procfile`. If it asks for a start command, use:
   `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300`

## Local Production-Style Test

```bash
DEEPSEEK_API_KEY='your-key' PORT=5021 gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300
```

The app intentionally uses one worker because the current simulation uses an
in-process lock and streams progress to the browser. For a larger public test,
add a real job queue before increasing concurrency.
