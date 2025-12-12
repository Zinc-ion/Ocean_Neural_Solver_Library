import os
import torch
import numpy as np
import swanlab
from exp.exp_basic import Exp_Basic
from utils.loss import L2Loss

class Exp_Ocean_Soda_Autoregressive(Exp_Basic):
    def __init__(self, args):
        super(Exp_Ocean_Soda_Autoregressive, self).__init__(args)
        # --- 新增：多卡并行封装 ---
        # 如果 args.gpu 包含多个ID (例如 "0,1,2,3") 且 CUDA 设备数 > 1
        if ',' in args.gpu and torch.cuda.device_count() > 1:
            print(f"检测到多卡配置 ({torch.cuda.device_count()} GPUs), 启用 DataParallel并行训练...")
            self.model = torch.nn.DataParallel(self.model)
        # ------------------------

    def masked_mse(self, pred, target, mask):
        """
        计算mask区域的MSE
        Args:
            pred: (N, 1) 预测值
            target: (N, 1) 真实值
            mask: (N, 1) bool mask, True表示有效数据
        """
        # 只在mask为True的位置计算误差
        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask.float()
        
        # 统计有效点
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
            pred: (N, 1) 预测值
            target: (N, 1) 真实值
            mask: (N, 1) bool mask, True表示有效数据
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
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_valid_points = 0
        total_all_points = 0
        num_batches = 0
        
        with torch.no_grad():
            for pos, fx, y, mask in self.test_loader:
                pos, fx, y, mask = pos.cuda(), fx.cuda(), y.cuda(), mask.cuda()
                
                # 初始化预测列表
                preds = []
                
                # 自回归预测
                curr_fx = fx
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        model_fx = None
                    else:
                        model_fx = curr_fx
                        
                    # 预测当前步
                    im = self.model(pos, fx=model_fx) # (N, out_dim)
                    preds.append(im)
                    
                    # 更新历史输入 (滑动窗口)
                    # fx: (N, T_in * out_dim)
                    # 移除最早的一个时间步，加入最新的预测
                    curr_fx = torch.cat((curr_fx[..., self.args.out_dim:], im), dim=-1)

                # 拼接所有时间步的预测
                # preds: list of T_out tensors, each (N, out_dim)
                # target y: (N, T_out * out_dim) or (N, T_out) if out_dim=1
                
                all_preds = torch.cat(preds, dim=-1)
                mse, valid_pts, total_pts = self.masked_mse(all_preds, y, mask)
                mae, _, _ = self.masked_mae(all_preds, y, mask)
                total_mse += mse.item()
                total_mae += mae.item()
                total_valid_points += valid_pts
                total_all_points += total_pts
                num_batches += 1
        
        avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
        
        print(f"  验证集统计: 有效点={total_valid_points}, 总点={total_all_points}, "
              f"有效占比={100*total_valid_points/total_all_points:.2f}%")
        
        return avg_mse

    def train(self):
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else: 
            raise ValueError('Optimizer only AdamW or Adam')
            
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        for ep in range(self.args.epochs):
            self.model.train()
            train_mse_accum = 0
            train_full_accum = 0
            num_batches = 0
            train_valid_points = 0
            train_total_points = 0

            for pos, fx, y, mask in self.train_loader:
                optimizer.zero_grad()
                pos, fx, y, mask = pos.cuda(), fx.cuda(), y.cuda(), mask.cuda()
                
                curr_fx = fx
                total_loss = 0
                
                preds = []
                
                # Autoregressive loop
                for t in range(self.args.T_out):
                    target_t = y[..., t*self.args.out_dim : (t+1)*self.args.out_dim]
                    mask_t = mask[..., t*self.args.out_dim : (t+1)*self.args.out_dim]
                    
                    if self.args.fun_dim == 0:
                        model_fx = None
                    else:
                        model_fx = curr_fx
                    
                    im = self.model(pos, fx=model_fx)
                    preds.append(im)
                    
                    # Calculate loss for this step
                    loss_t, valid_pts, total_pts = self.masked_mse(im, target_t, mask_t)
                    total_loss += loss_t
                    
                    train_valid_points += valid_pts
                    train_total_points += total_pts

                    # Update history
                    if self.args.teacher_forcing:
                        # Teacher forcing: use ground truth
                        curr_fx = torch.cat((curr_fx[..., self.args.out_dim:], target_t), dim=-1)
                    else:
                        # Free running: use prediction
                        curr_fx = torch.cat((curr_fx[..., self.args.out_dim:], im), dim=-1)
                
                # 拼接所有时间步的预测，计算完整序列的MSE full-loss
                all_preds = torch.cat(preds, dim=-1)
                full_loss, _, _ = self.masked_mse(all_preds, y, mask)
                train_full_accum += full_loss.item()

                # Backprop 这里只使用单步误差的累计来更新权重，full-loss的整个序列误差只用做展示
                total_loss.backward()
                
                # 检查是否进行梯度裁剪
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
                
                # 记录整个序列的平均MSE (total_loss 是 T_out 个步的 MSE 之和)
                # 这里为了展示方便，我们记录平均每个步的MSE
                train_mse_accum += total_loss.item() / self.args.T_out
                num_batches += 1

            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            avg_train_loss_step = train_mse_accum / num_batches if num_batches > 0 else 0.0
            avg_train_loss_full = train_full_accum / num_batches if num_batches > 0 else 0.0
            print("Epoch {} Train loss step MSE : {:.5f} Train loss full MSE: {:.5f} Valid Points Ratio: {:.2f}%".format(
                ep, avg_train_loss_step, avg_train_loss_full,
                100 * train_valid_points / train_total_points if train_total_points > 0 else 0.0
            ))

            valid_loss = self.vali()
            print("Epoch {} Valid MSE: {:.5f}".format(ep, valid_loss))

            swanlab.log({
                "train_loss_step": avg_train_loss_step,
                "train_loss_full": avg_train_loss_full,
                "valid_points_ratio": 100 * train_valid_points / train_total_points if train_total_points > 0 else 0.0,
                "valid_mse": valid_loss,
                "epoch": ep
            })

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save models')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('final save models')
        torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

    def test(self):
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        total_mse = 0.0
        total_mae = 0.0
        total_valid_points = 0
        total_all_points = 0
        num_batches = 0
        
        result_file = os.path.join('./results', self.args.save_name, 'test_results.txt')
        
        with open(result_file, 'w') as f:
            f.write("OceanSoda Autoregressive Test Results\n")
            f.write("="*60 + "\n")
        
            with torch.no_grad():
                for i, (pos, fx, y, mask) in enumerate(self.test_loader):
                    pos, fx, y, mask = pos.cuda(), fx.cuda(), y.cuda(), mask.cuda()
                    
                    preds = []
                    curr_fx = fx
                    
                    for t in range(self.args.T_out):
                        if self.args.fun_dim == 0:
                            model_fx = None
                        else:
                            model_fx = curr_fx
                        
                        im = self.model(pos, fx=model_fx)
                        preds.append(im)
                        curr_fx = torch.cat((curr_fx[..., self.args.out_dim:], im), dim=-1)
                    
                    all_preds = torch.cat(preds, dim=-1)
                    
                    mse, valid_pts, total_pts = self.masked_mse(all_preds, y, mask)
                    mae, _, _ = self.masked_mae(all_preds, y, mask)
                    
                    total_mse += mse.item()
                    total_mae += mae.item()
                    total_valid_points += valid_pts
                    total_all_points += total_pts
                    num_batches += 1
                    
                    if i < 5: # Log first 5 batches
                         f.write(f"Batch {i}: MSE={mse.item():.6f}, MAE={mae.item():.6f}, Valid={valid_pts}/{total_pts}\n")

            avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
            avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
            valid_ratio = 100 * total_valid_points / total_all_points if total_all_points > 0 else 0.0
            
            print(f"Test MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, Valid Ratio: {valid_ratio:.2f}%")
            
            f.write("="*60 + "\n")
            f.write(f"Average MSE: {avg_mse:.6f}\n")
            f.write(f"Average MAE: {avg_mae:.6f}\n")
            f.write(f"Valid Ratio: {valid_ratio:.2f}%\n")
