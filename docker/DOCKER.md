# ParaScale Docker

This directory contains the minimal container wrapper for local or remote validation.

## Build

```bash
docker build -f docker/Dockerfile -t parascale:latest .
```

## Run A Plan Check

```bash
docker run --rm --gpus all parascale:latest \
  python -m parascale.cli plan --config configs/vision_synthetic.json
```

## Run Tests

```bash
docker run --rm --gpus all parascale:latest python tests/run_tests.py
```

## Compose

```bash
cd docker
docker compose up -d parascale-single
docker exec -it parascale-single bash
python -m parascale.cli plan --config configs/vision_synthetic.json
python tests/run_tests.py
docker compose down
```

For the remote CUDA test machine, prefer the requested base image when available:
`unified-torch-distributed:cu121-torch24`.
