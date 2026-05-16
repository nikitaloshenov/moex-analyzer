FROM python:3.10-slim

WORKDIR /app

# Ставим только то, что реально нужно и точно есть в репозиториях
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ставим зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Закидываем проект
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
