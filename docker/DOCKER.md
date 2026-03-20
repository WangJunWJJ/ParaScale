# ParaScale Docker 部署指南

本文档介绍如何使用 Docker 一键部署 ParaScale 框架。

## 前置要求

- Docker >= 20.0
- Docker Compose >= 1.29
- NVIDIA Docker 运行时（用于 GPU 支持）

## 安装 NVIDIA Docker 运行时

```bash
# 添加 NVIDIA Docker 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装 nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 重启 Docker
sudo systemctl restart docker
```

## 快速开始

### 1. 构建镜像

```bash
cd docker
docker build -t parascale:latest ..
```

### 2. 运行单节点容器

```bash
cd docker
docker run --rm -it --gpus all \
    -v $(pwd)/../data:/workspace/data \
    -v $(pwd)/../checkpoints:/workspace/checkpoints \
    parascale:latest
```

### 3. 使用 Docker Compose（推荐）

```bash
# 进入 docker 目录
cd docker

# 启动单节点训练环境
docker-compose up -d parascale-single

# 进入容器
docker exec -it parascale-single bash

# 运行测试
python tests/test_refactoring.py

# 运行示例
python examples/data_parallel_test.py
```

## 多节点分布式训练

### 启动多节点环境

```bash
# 进入 docker 目录
cd docker

# 启动主节点和工作节点
docker-compose up -d parascale-master parascale-worker

# 进入主节点
docker exec -it parascale-master bash

# 启动分布式训练
python examples/multi_node_example.py --epochs=5
```

## 常用命令

```bash
# 构建镜像（从 docker 目录）
cd docker
docker build -t parascale:latest ..

# 运行交互式容器
docker run --rm -it --gpus all parascale:latest

# 运行测试
docker run --rm --gpus all parascale:latest python tests/test_refactoring.py

# 查看日志
docker logs parascale-single

# 停止容器
docker-compose down

# 清理所有容器和镜像
docker-compose down --rmi all
```

## 数据持久化

容器中的以下目录已挂载到宿主机：
- `/workspace/data` -> `./data`
- `/workspace/checkpoints` -> `./checkpoints`
- `/workspace/logs` -> `./logs`

## 自定义配置

### 修改 GPU 可见性

在 `docker-compose.yml` 中修改环境变量：
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0,1  # 只使用 GPU 0 和 1
```

### 修改 PyTorch 版本

在 `Dockerfile` 中修改基础镜像：
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

## 故障排除

### GPU 不可见

```bash
# 检查 NVIDIA Docker 运行时
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 内存不足

```bash
# 限制容器内存
docker run --memory=16g --gpus all parascale:latest
```

### 端口冲突

```bash
# 修改 docker-compose.yml 中的端口映射
ports:
  - "29500:29500"
```
