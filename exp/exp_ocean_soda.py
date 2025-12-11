import os
import torch
import torch.nn.functional as F
from exp.exp_basic import Exp_Basic
from utils.loss import L2Loss
import numpy as np


class Exp_Ocean_SODA(Exp_Basic):
    """
    专门用于Ocean SODA数据集的实验类
    支持mask的MSE/MAE计算，只在有效数据点上评估
    """
    def __init__(self, args):
        super(Exp_Ocean_SODA, self).__init__(args)  

    def masked_mse(self, pred, target, mask):
        """
        计算mask区域的MSE
        
        Args:
            pred: (B, N, T_out) 预测值
            target: (B, N, T_out) 真实值
            mask: (B, N, T_out) bool mask, True表示有效数据
        
        Returns:
            mse: 标量
            valid_points: 有效点数量
            total_points: 总点数量
        """
        # 只在mask为True的位置计算误差
        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask.float()
        
        # 统计有效点和总点
        valid_points = mask.sum().item()
        total_points = mask.numel()
        
        if valid_points > 0:
            mse = masked_error.sum() / valid_points
        else:
            mse = torch.tensor(0.0).to(pred.device)
        
        return mse, valid_points, total_points

    def masked_mae(self, pred, target, mask):
        """
        计算mask区域的MAE
        
        Args:
            pred: (B, N, T_out) 预测值
            target: (B, N, T_out) 真实值
            mask: (B, N, T_out) bool mask, True表示有效数据
        
        Returns:
            mae: 标量
            valid_points: 有效点数量
            total_points: 总点数量
        """
        abs_error = torch.abs(pred - target)
        masked_error = abs_error * mask.float()
        
        valid_points = mask.sum().item()
        total_points = mask.numel()
        
        if valid_points > 0:
            mae = masked_error.sum() / valid_points
        else:
            mae = torch.tensor(0.0).to(pred.device)
        
        return mae, valid_points, total_points

    def vali(self):
        """
        验证函数，计算测试集上的masked MSE和MAE
        """
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_valid_points = 0
        total_all_points = 0
        num_batches = 0
        
        with torch.no_grad():
            for pos, fx, y, mask in self.test_loader:
                pos, fx, y, mask = pos.cuda(), fx.cuda(), y.cuda(), mask.cuda()
                
                # 获取mask
                # if hasattr(y, 'mask'):
                #     mask = y.mask.cuda()
                # else:
                #     mask = torch.ones_like(y, dtype=torch.bool)
                
                # 模型预测
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(pos, fx)
                
                # 计算masked MSE和MAE
                mse, valid_pts, total_pts = self.masked_mse(out, y, mask)
                mae, _, _ = self.masked_mae(out, y, mask)
                
                total_mse += mse.item()
                total_mae += mae.item()
                total_valid_points += valid_pts
                total_all_points += total_pts
                num_batches += 1
        
        avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
        
        print(f"  验证集统计: 有效点={total_valid_points}, 总点={total_all_points}, "
              f"有效占比={100*total_valid_points/total_all_points:.2f}%")
        
        return avg_mse, avg_mae

    def train(self):
        """
        训练函数
        """
        # 优化器设置
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                                         lr=self.args.lr, 
                                         weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.args.lr, 
                                        weight_decay=self.args.weight_decay)
        else: 
            raise ValueError('Optimizer only AdamW or Adam')
        
        # 学习率调度器设置
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, 
                epochs=self.args.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=self.args.pct_start
            )
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs
            )
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.args.step_size, 
                gamma=self.args.gamma
            )

        # 训练循环
        for ep in range(self.args.epochs):
            self.model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_valid_points = 0
            train_total_points = 0
            num_batches = 0

            for pos, fx, y, mask in self.train_loader:
                pos, fx, y, mask = pos.cuda(), fx.cuda(), y.cuda(), mask.cuda()
                
                # 获取mask
                # if hasattr(y, 'mask'):
                #     mask = y.mask.cuda()
                # else:
                #     mask = torch.ones_like(y, dtype=torch.bool)
                
                # 前向传播
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(pos, fx)
                
                # 计算masked loss
                loss, valid_pts, total_pts = self.masked_mse(out, y, mask)
                
                train_loss += loss.item()
                train_mse += loss.item()
                train_valid_points += valid_pts
                train_total_points += total_pts
                num_batches += 1
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   self.args.max_grad_norm)
                optimizer.step()
                
                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
            
            if self.args.scheduler in ['CosineAnnealingLR', 'StepLR']:
                scheduler.step()

            # 打印训练统计
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
            print(f"\nEpoch {ep+1}/{self.args.epochs}")
            print(f"  训练损失: {avg_train_loss:.6f}")
            print(f"  训练集统计: 有效点={train_valid_points}, 总点={train_total_points}, "
                  f"有效占比={100*train_valid_points/train_total_points:.2f}%")

            # 验证
            val_mse, val_mae = self.vali()
            print(f"  验证 MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")

            # 保存模型
            if (ep + 1) % 100 == 0 or (ep + 1) == self.args.epochs:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print(f'  保存模型到 checkpoints/{self.args.save_name}.pt')
                torch.save(self.model.state_dict(), 
                          os.path.join('./checkpoints', self.args.save_name + '.pt'))

        print('\n训练完成，保存最终模型')
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(self.model.state_dict(), 
                  os.path.join('./checkpoints', self.args.save_name + '.pt'))

    def test(self):
        """
        测试函数，计算详细的评估指标
        """
        print("\n开始测试...")
        self.model.load_state_dict(
            torch.load("./checkpoints/" + self.args.save_name + ".pt")
        )
        self.model.eval()
        
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        total_mse = 0.0
        total_mae = 0.0
        total_valid_points = 0
        total_all_points = 0
        num_batches = 0
        
        # 用于保存每个样本的结果
        sample_results = []

        with torch.no_grad():
            for batch_idx, (pos, fx, y, mask) in enumerate(self.test_loader):
                pos, fx, y, mask = pos.cuda(), fx.cuda(), y.cuda(), mask.cuda()
                
                # 获取mask
                # if hasattr(y, 'mask'):
                #     mask = y.mask.cuda()
                # else:
                #     mask = torch.ones_like(y, dtype=torch.bool)
                
                # 预测
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(pos, fx)
                
                # 计算指标
                mse, valid_pts, total_pts = self.masked_mse(out, y, mask)
                mae, _, _ = self.masked_mae(out, y, mask)
                
                total_mse += mse.item()
                total_mae += mae.item()
                total_valid_points += valid_pts
                total_all_points += total_pts
                num_batches += 1
                
                # 保存样本结果
                sample_results.append({
                    'batch_idx': batch_idx,
                    'mse': mse.item(),
                    'mae': mae.item(),
                    'valid_points': valid_pts,
                    'total_points': total_pts
                })
                
                # 可视化前几个样本
                if batch_idx < self.args.vis_num:
                    self._visualize_sample(batch_idx, pos, y, out, mask)

        # 计算平均指标
        avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
        valid_ratio = 100 * total_valid_points / total_all_points if total_all_points > 0 else 0.0
        
        # 打印测试结果
        print("\n" + "="*60)
        print("测试结果汇总:")
        print("="*60)
        print(f"平均 MSE: {avg_mse:.6f}")
        print(f"平均 MAE: {avg_mae:.6f}")
        print(f"总有效点数: {total_valid_points:,}")
        print(f"总点数: {total_all_points:,}")
        print(f"有效数据占比: {valid_ratio:.2f}%")
        print(f"过滤掉的点数: {total_all_points - total_valid_points:,}")
        print("="*60)
        
        # 保存结果到文件
        result_file = os.path.join('./results', self.args.save_name, 'test_results.txt')
        with open(result_file, 'w') as f:
            f.write("POC Flux 测试结果\n")
            f.write("="*60 + "\n")
            f.write(f"平均 MSE: {avg_mse:.6f}\n")
            f.write(f"平均 MAE: {avg_mae:.6f}\n")
            f.write(f"总有效点数: {total_valid_points:,}\n")
            f.write(f"总点数: {total_all_points:,}\n")
            f.write(f"有效数据占比: {valid_ratio:.2f}%\n")
            f.write(f"过滤掉的点数: {total_all_points - total_valid_points:,}\n")
            f.write("="*60 + "\n\n")
            
            f.write("每个批次的详细结果:\n")
            for res in sample_results:
                f.write(f"Batch {res['batch_idx']}: MSE={res['mse']:.6f}, "
                       f"MAE={res['mae']:.6f}, Valid={res['valid_points']}/{res['total_points']}\n")
        
        print(f"\n结果已保存到: {result_file}")

    def _visualize_sample(self, idx, pos, y_true, y_pred, mask):
        """
        可视化单个样本(可选实现)
        """
        try:
            import matplotlib.pyplot as plt
            
            # 取第一个batch的第一个时间步
            H, W = self.args.shapelist
            
            # 重塑为图像格式
            true_img = y_true[0, :, 0].cpu().reshape(H, W).numpy()
            pred_img = y_pred[0, :, 0].cpu().reshape(H, W).numpy()
            mask_img = mask[0, :, 0].cpu().reshape(H, W).numpy()
            
            # 创建图像
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            im0 = axes[0].imshow(true_img, cmap='RdBu_r')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(pred_img, cmap='RdBu_r')
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            error = np.abs(true_img - pred_img)
            im2 = axes[2].imshow(error, cmap='hot')
            axes[2].set_title('Absolute Error')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2])
            
            im3 = axes[3].imshow(mask_img, cmap='gray')
            axes[3].set_title('Valid Data Mask')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3])
            
            save_path = os.path.join('./results', self.args.save_name, 
                                    f'visualization_{idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  可视化保存到: {save_path}")
        except Exception as e:
            print(f"  可视化失败: {e}")