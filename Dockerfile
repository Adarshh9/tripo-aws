# Stage 1: Build the dependencies
FROM python:3.12-bullseye AS builder
# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Set working directory
WORKDIR /app
# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472 \
    && rm -rf ~/.cache/pip
# Copy the application files
COPY . /app
# Stage 2: Final image
FROM public.ecr.aws/lambda/python:3.12

# Copy the application files and installed packages from the builder stage
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /var/lang/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /var/lang/bin
# Set the working directory
WORKDIR /app

# Set the entry point
ENTRYPOINT ["python","lambda_function.py"]