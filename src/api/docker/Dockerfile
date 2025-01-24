FROM python:3.10.11

WORKDIR /src

COPY /api/docker/requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

ENV DOCKER_CHANNEL stable
ENV DOCKER_VERSION 27.3.1
ENV DOCKER_API_VERSION 1.40
RUN curl -fsSL "https://download.docker.com/linux/static/${DOCKER_CHANNEL}/x86_64/docker-${DOCKER_VERSION}.tgz" | tar -xzC /usr/local/bin --strip=1 docker/docker

COPY main.py /src
COPY /api/controllers /src/api/controllers

ENV PYTHONPATH "/api"

EXPOSE 8001
#CMD ["uvicorn",  "main:app", "--host", "0.0.0.0", "--port", "5004", "--reload"]
# CMD ["python3", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
CMD ["python3", "main.py"]
