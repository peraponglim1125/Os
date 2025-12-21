FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    build-essential gcc g++ gfortran \
    pkg-config \
    libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Bangkok

COPY requirements.txt .

ARG TARGETPLATFORM

RUN python -m pip install --upgrade pip setuptools wheel

RUN if [ "$TARGETPLATFORM" = "linux/arm/v7" ]; then \
      pip install --no-cache-dir --prefer-binary \
      --extra-index-url https://www.piwheels.org/simple \
      -r requirements.txt ; \
    else \
      pip install --no-cache-dir --prefer-binary -r requirements.txt ; \
    fi

COPY . .

EXPOSE 5000
CMD ["python", "api.py"]
