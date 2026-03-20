# ParaScale Docker 镜像
# 基于 PyTorch 官方镜像构建，支持分布式训练

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY LICENSE .
COPY parascale/ ./parascale/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY docu/ ./docu/

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装 ParaScale 包
RUN pip install -e .

# 创建数据目录
RUN mkdir -p /workspace/data /workspace/checkpoints

# 设置默认命令
CMD ["/bin/bash"]
