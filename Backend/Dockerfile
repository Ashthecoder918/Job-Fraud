# Base Image
FROM python:3.10-slim

# workdir
WORKDIR /app

# copying requirements
COPY requirements.txt .

# Upgrade pip for stability
RUN pip install --no-cache-dir --upgrade pip

# 1. Install the lightweight CPU version of Torch directly from the official PyTorch index
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of your app dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# copy app
COPY . .

# port expose
EXPOSE 8000

# command to start fastapi
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
