"""
ParaScale 多节点训练示例

本示例展示如何在多节点环境中使用 ParaScale 进行分布式训练。
支持 torchrun、SLURM、MPI 等多种启动方式。

使用方法:

1. torchrun 启动（推荐）:
   # 单节点多 GPU
   torchrun --nproc_per_node=4 multi_node_example.py --epochs=5
   
   # 多节点（2 节点，每节点 4 GPU）
   # Node 0:
   torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
       --master_addr=<node0_ip> --master_port=29500 \
       multi_node_example.py --epochs=5
   
   # Node 1:
   torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
       --master_addr=<node0_ip> --master_port=29500 \
       multi_node_example.py --epochs=5

2. SLURM 启动:
   sbatch --nodes=2 --gpus-per-node=4 --ntasks-per-node=4 multi_node_job.sh

3. 手动指定参数:
   python multi_node_example.py --rank=0 --world_size=8 --master_addr=192.168.1.100
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parascale import Engine, ParaScaleConfig
from parascale.utils import (
    initialize_distributed,
    cleanup_distributed,
    get_distributed_info
)
from parascale.utils import print_rank_0, setup_logging, is_main_process


class SimpleCNN(nn.Module):
    """简单的 CNN 模型用于测试"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 8x8
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_cifar10_dataloaders(batch_size=64, data_dir='./data'):
    """获取 CIFAR-10 数据加载器"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # 使用 DistributedSampler 进行分布式采样
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testset, shuffle=False
        )
    else:
        train_sampler = None
        test_sampler = None
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, sampler=test_sampler,
        num_workers=2, pin_memory=True
    )
    
    return trainloader, testloader, train_sampler


def main():
    parser = argparse.ArgumentParser(description='ParaScale 多节点训练示例')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='保存目录')
    
    # 分布式参数（可选，通常由 torchrun/srun 自动设置）
    parser.add_argument('--rank', type=int, default=None, help='进程 rank')
    parser.add_argument('--world-size', type=int, default=None, help='进程总数')
    parser.add_argument('--local-rank', type=int, default=None, help='本地 rank')
    parser.add_argument('--master-addr', type=str, default=None, help='主节点地址')
    parser.add_argument('--master-port', type=int, default=None, help='主节点端口')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    print_rank_0("\n" + "=" * 60)
    print_rank_0("ParaScale 多节点训练示例")
    print_rank_0("=" * 60)
    
    # 初始化分布式环境（如果尚未初始化）
    if not torch.distributed.is_initialized():
        # 如果提供了命令行参数，使用它们
        if args.rank is not None and args.world_size is not None:
            rank, world_size, local_rank = initialize_distributed(
                rank=args.rank,
                world_size=args.world_size,
                local_rank=args.local_rank,
                master_addr=args.master_addr,
                master_port=args.master_port
            )
        else:
            # 自动检测环境
            rank, world_size, local_rank = initialize_distributed()
    else:
        info = get_distributed_info()
        rank = info['rank']
        world_size = info['world_size']
        local_rank = info['local_rank']
    
    print_rank_0(f"\n训练配置:")
    print_rank_0(f"  Epochs: {args.epochs}")
    print_rank_0(f"  Batch size: {args.batch_size}")
    print_rank_0(f"  Learning rate: {args.lr}")
    print_rank_0(f"  World size: {world_size}")
    
    # 创建模型并转移到 GPU
    model = SimpleCNN()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 配置 ParaScale
    config = ParaScaleConfig(
        data_parallel_size=world_size,  # 使用所有进程进行数据并行
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_save_path=args.save_dir
    )
    
    # 创建引擎（自动初始化分布式环境）
    engine = Engine(model, optimizer, config, auto_init_distributed=False)
    
    # 获取数据加载器
    trainloader, testloader, train_sampler = get_cifar10_dataloaders(
        batch_size=args.batch_size, 
        data_dir=args.data_dir
    )
    
    # 训练
    print_rank_0("\n开始训练...")
    for epoch in range(args.epochs):
        # 设置 epoch 以便 DistributedSampler 正确采样
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        engine.train(trainloader, epochs=1)
        
        # 评估
        if epoch % 2 == 0:
            loss, accuracy = engine.evaluate(testloader)
            print_rank_0(f"\nEpoch {epoch+1} 评估结果:")
            print_rank_0(f"  Loss: {loss:.4f}")
            print_rank_0(f"  Accuracy: {accuracy:.2f}%")
    
    # 保存最终模型
    if is_main_process():
        engine.save_checkpoint()
        print_rank_0(f"\n模型已保存到 {args.save_dir}")
    
    # 清理分布式环境
    cleanup_distributed()
    
    print_rank_0("\n训练完成！")


if __name__ == '__main__':
    main()
