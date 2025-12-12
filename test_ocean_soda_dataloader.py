import argparse
import os
from types import SimpleNamespace

import torch
import numpy as np

from data_provider.data_loader import ocean_soda


def run_test(data_path, T_in=3, T_out=3, batch_size=1, train_ratio=0.8, valid_ratio=0.1):
    args = SimpleNamespace(
        loader='ocean_soda',
        data_path=data_path,
        T_in=T_in,
        T_out=T_out,
        batch_size=batch_size,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    dataset = ocean_soda(args)
    train_loader, test_loader, shapelist = dataset.get_loader()
    H, W = shapelist
    N = H * W

    print(f"shapelist={shapelist}, N={N}")
    print(f"T_in={T_in}, T_out={T_out}, batch_size={batch_size}")

    frames_total = len(getattr(train_loader.dataset, "all_identifiers", []))
    seq_len = T_in + T_out
    windows_total = max(frames_total - seq_len + 1, 0)
    train_count = len(train_loader.dataset)
    test_count = len(test_loader.dataset)
    valid_count = max(windows_total - train_count - test_count, 0)
    print(f"frames_total={frames_total}, seq_len={seq_len}, windows_total={windows_total}")
    print(f"samples_train={train_count}, samples_valid={valid_count}, samples_test={test_count}")
    print(f"train_batches={len(train_loader)}, test_batches={len(test_loader)}")
    print(f"ratios: train={train_ratio}, valid={valid_ratio}, test={1.0 - train_ratio - valid_ratio}")

    train_samples = getattr(train_loader.dataset, "samples", None)
    if train_samples is None or len(train_samples) < 2:
        print("FAIL: 训练集样本数量不足或无法访问 samples")
        return

    s0 = train_samples[0]
    s1 = train_samples[1]
    stride_ok = (len(s0) == T_in + T_out) and (len(s1) == T_in + T_out) and (s1[0] == s0[0] + 1) and (s1[-1] == s0[-1] + 1)
    print(f"stride=1 check={stride_ok}")

    try:
        pos, fx, y, mask = next(iter(train_loader))
    except StopIteration:
        print("FAIL: 训练集 DataLoader 无可用批次")
        return

    shape_ok = (
        pos.shape[0] == batch_size and pos.shape[1] == N and pos.shape[2] == 2 and
        fx.shape[0] == batch_size and fx.shape[1] == N and fx.shape[2] == T_in and
        y.shape[0] == batch_size and y.shape[1] == N and y.shape[2] == T_out and
        mask.shape[0] == batch_size and mask.shape[1] == N and mask.shape[2] == T_out
    )
    print(f"batch shapes: pos={tuple(pos.shape)}, fx={tuple(fx.shape)}, y={tuple(y.shape)}, mask={tuple(mask.shape)}")
    print(f"shape check={shape_ok}")

    pos_range_ok = (pos.min().item() >= 0.0) and (pos.max().item() <= 1.0)
    norm_fx_ok = (fx.min().item() >= -1.0001) and (fx.max().item() <= 1.0001)
    norm_y_ok = (y.min().item() >= -1.0001) and (y.max().item() <= 1.0001)
    mask_bool_ok = (mask.dtype == torch.bool) and (mask.any().item() or (~mask).any().item())
    print(f"pos in [0,1]={pos_range_ok}, fx in [-1,1]={norm_fx_ok}, y in [-1,1]={norm_y_ok}, mask_bool={mask_bool_ok}")

    global_grid_ok = (pos.shape[1] == H * W)
    print(f"global grid check={global_grid_ok}")

    ok = stride_ok and shape_ok and pos_range_ok and norm_fx_ok and norm_y_ok and mask_bool_ok and global_grid_ok
    if ok:
        print("PASS: ocean_soda dataloader 读取与构造样本正常")
    else:
        print("FAIL: ocean_soda dataloader 检查未通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"/Data1/carbon_data/OceanSODA-ETHZ-v2/fgco2")
    parser.add_argument("--T_in", type=int, default=3)
    parser.add_argument("--T_out", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    if not os.path.isdir(args.data_path):
        print(f"FAIL: 数据路径不存在: {args.data_path}")
    else:
        run_test(args.data_path, T_in=args.T_in, T_out=args.T_out, batch_size=args.batch_size)