# SuperKart Sales Forecaster

Flask inference API + Streamlit UI, each in its own Docker container, connected by a Docker network.

## Run in GitHub Codespaces
```bash
# 1. Shared network so the frontend can reach the backend by container name
docker network create superkart-net

# 2. Backend (Flask API on port 7860)
docker build -t superkart-backend ./backend_files
docker run -d --name backend --network superkart-net -p 7860:7860 superkart-backend

# 3. Frontend (Streamlit on port 8501)
docker build -t superkart-frontend ./frontend_files
docker run -d --name frontend --network superkart-net -p 8501:8501 -e BACKEND_URL=http://backend:7860 superkart-frontend

# 4. Check both containers are up
docker ps
curl http://localhost:7860/
```
Then open the **Ports** tab, set ports **7860** and **8501** to **Public**, and open the forwarded URL for 8501.

## Troubleshooting
- `docker logs backend` / `docker logs frontend` show errors.
- Rebuild after code changes: `docker rm -f backend frontend` then repeat steps 2-3.
- 404 on the forwarded URL: confirm the container is running and the port visibility is Public.

## API
- `GET /` health check
- `POST /v1/predict` JSON body with the 10 features -> `{"Predicted_Sales": ..., "warnings": [...]}`
- `POST /v1/predictbatch` CSV in form field `file` -> `{"0": ..., "1": ...}`
