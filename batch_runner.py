import os
import sys
import subprocess
import time
from datetime import datetime
from multiprocessing import Process

# ==========================================================================================
# ⚙️ 全局通用参数设置 (所有模型共用的参数)
# ==========================================================================================
COMMON_ARGS = {
    # 训练参数
    'epochs': 500,
    # 'batch_size': 8,
    'lr': 1e-3,
    'optimizer': 'AdamW',
    'scheduler': 'OneCycleLR',
    
    # 数据参数
    'data_path': '/data/fcj/cdsd/data/SCO2',
    'task': 'poc_flux',      # 对应你代码中的 task
    'loader': 'poc_flux',
    'train_ratio': 0.8,
    'valid_ratio': 0.1,
    'img_size': 256,         # POC Flux 特定参数
    
    # 物理/模型通用参数
    'modes': 12,             # FNO/LSM basis functions
    'n_hidden': 64,
    'n_layers': 4,
}

# 你的训练脚本路径
SCRIPT_PATH = 'run.py'  # ⚠️ 请确认这里是你的训练脚本文件名，如果是 main.py 请修改

# ==========================================================================================
# 📋 任务分配列表 (根据你的截图手动分配模型)
# ==========================================================================================

# --- 显卡 0 的任务列表 ---
TASKS_GPU_0 = [
    # 格式: {'model': '模型名', 'extra_args': {特有参数字典, 可覆盖通用参数}}
    {'model': 'U_Net',      'extra_args': {'n_layers': 5}}, # U-Net 2015
    {'model': 'Transolver',     'extra_args': {}},          
    {'model': 'FNO',        'extra_args': {'modes': 16}},   # FNO 2020
    {'model': 'U_NO',       'extra_args': {}},              # U-NO 2022
]

# --- 显卡 1 的任务列表 ---
TASKS_GPU_1 = [
    {'model': 'U_FNO',      'extra_args': {}},              # U-FNO 2022
    {'model': 'F_FNO',      'extra_args': {}},              # F-FNO 2023
    {'model': 'LSM',        'extra_args': {}},              # LSM 2023
    {'model': 'MWT',  'extra_args': {}},              
    {'model': 'ONO',       'extra_args': {'exp_note': 'My_Method'}}, # This work
]

# ==========================================================================================
# 🚀 核心执行引擎
# ==========================================================================================

def get_current_time_str():
    return datetime.now().strftime('%m%d_%H%M')

def construct_command(gpu_id, task_config):
    """构建命令行指令"""
    model_name = task_config['model']
    extra_args = task_config.get('extra_args', {})
    
    # 1. 合并参数: 默认参数 < extra_args
    final_args = COMMON_ARGS.copy()
    final_args.update(extra_args)
    
    # 2. 生成唯一的 save_name (模型名_任务_时间)
    # 这样就不会覆盖文件夹了
    exp_note = final_args.pop('exp_note', 'default') # 如果extra_args里有备注就取出来
    save_name = f"{model_name}_{final_args['task']}_{exp_note}_{get_current_time_str()}"
    
    # 3. 构建命令列表
    cmd = [sys.executable, SCRIPT_PATH]
    
    # 添加固定参数
    cmd.append(f'--gpu={gpu_id}')
    cmd.append(f'--model={model_name}')
    cmd.append(f'--save_name={save_name}')
    
    # 添加动态参数
    for key, val in final_args.items():
        if isinstance(val, bool):
            if val: cmd.append(f'--{key}') # bool类型只加key
        else:
            cmd.append(f'--{key}')
            cmd.append(str(val))
            
    return cmd, save_name

def gpu_worker(gpu_id, task_list):
    """
    独立的工作进程，负责在一张显卡上顺序执行任务列表
    """
    prefix = f" [GPU {gpu_id}] "
    total = len(task_list)
    
    print(f"{prefix}🚀 启动工作进程，待处理任务数: {total}")
    
    for idx, task in enumerate(task_list):
        model_name = task['model']
        cmd, save_name = construct_command(gpu_id, task)
        
        print(f"\n{'-'*20} {prefix} 任务 {idx+1}/{total}: {model_name} {'-'*20}")
        print(f"{prefix}📂 Save Name: {save_name}")
        print(f"{prefix}⌨️  Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        # 设置环境变量，确保只看到当前显卡
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        try:
            # 执行命令
            # stdout=None 表示直接输出到终端，但可能会和另一个GPU的输出混杂
            # 建议: 如果不想看刷屏，可以将 stdout=subprocess.DEVNULL
            subprocess.run(cmd, check=True, env=env)
            
            duration = (time.time() - start_time) / 60
            print(f"{prefix}✅ 任务完成: {model_name} (耗时: {duration:.2f} min)")
            
        except subprocess.CalledProcessError as e:
            print(f"{prefix}❌ 任务失败: {model_name}")
            print(f"{prefix}Error: {e}")
        except KeyboardInterrupt:
            print(f"{prefix}🛑 用户中断")
            break

    print(f"{prefix}💤 所有分配的任务已完成。")

if __name__ == "__main__":
    print(f"💎 自动并行实验调度器启动")
    print(f"   - GPU 0 队列: {[t['model'] for t in TASKS_GPU_0]}")
    print(f"   - GPU 1 队列: {[t['model'] for t in TASKS_GPU_1]}")
    print("="*80)

    # 创建两个独立的进程
    p0 = Process(target=gpu_worker, args=(0, TASKS_GPU_0))
    p1 = Process(target=gpu_worker, args=(1, TASKS_GPU_1))

    # 启动进程
    p0.start()
    p1.start()

    # 等待进程结束
    p0.join()
    p1.join()

    print("\n🎉🎉🎉 所有显卡任务均已结束！")