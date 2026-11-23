FROM python:3.13-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the API code and the serialized model
COPY app.py superkart_model.joblib ./

# The API listens on port 7860
EXPOSE 7860

# 2 gunicorn workers serving the Flask app object `superkart_api` in app.py
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:superkart_api"]
