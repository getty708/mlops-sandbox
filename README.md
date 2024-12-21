# Triton Inference Sever Sandbox

## Triton Inference Server

- [Understand the Overheads of Ensemble Models](./benchmarks/ensemble_model_overhead/)

## PyTriton

- [Quick Start](./pytriton/quick-start/)

## Setup

```bash
# for CPU devce
make init

# for GPU device
make init-gpu
```

## Launch Monitoring Tools

```bash
# Launch Prometheus and Grafana
make up-monitor

# Stop monitoring services
make stop-monitor

# Stop and remove monitoring services
make down-monitor
```
