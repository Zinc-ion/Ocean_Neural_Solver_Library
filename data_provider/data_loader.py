import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import scipy.io as scio
from data_provider.shapenet_utils import get_datalist
from data_provider.shapenet_utils import GraphDataset
from torch.utils.data import Dataset
from utils.normalizer import UnitTransformer, UnitGaussianNormalizer


class plas(object):
    def __init__(self, args):
        self.DATA_PATH = args.data_path + '/plas_N987_T20.mat'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_out = args.T_out
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def random_collate_fn(self, batch):
        shuffled_batch = []
        shuffled_u = None
        shuffled_t = None
        shuffled_a = None
        shuffled_pos = None
        for item in batch:
            pos = item[0]
            t = item[1]
            a = item[2]
            u = item[3]

            num_timesteps = t.size(0)
            permuted_indices = torch.randperm(num_timesteps)
            t = t[permuted_indices]
            u = u.reshape(u.shape[0], num_timesteps, -1)[..., permuted_indices, :].reshape(u.shape[0], -1)

            if shuffled_t is None:
                shuffled_pos = pos.unsqueeze(0)
                shuffled_t = t.unsqueeze(0)
                shuffled_u = u.unsqueeze(0)
                shuffled_a = a.unsqueeze(0)
            else:
                shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
                shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
                shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
                shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)

        shuffled_batch.append(shuffled_pos)
        shuffled_batch.append(shuffled_t)
        shuffled_batch.append(shuffled_a)
        shuffled_batch.append(shuffled_u)

        return shuffled_batch  # B N T 4

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((101 - 1) / r1) + 1)
        s2 = int(((31 - 1) / r2) + 1)

        data = scio.loadmat(self.DATA_PATH)
        input = torch.tensor(data['input'], dtype=torch.float)
        output = torch.tensor(data['output'], dtype=torch.float)
        print(input.shape, output.shape)
        x_train = input[:self.ntrain, ::r1][:, :s1].reshape(self.ntrain, s1, 1).repeat(1, 1, s2)
        x_train = x_train.reshape(self.ntrain, -1, 1)
        y_train = output[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, self.T_out * self.out_dim)
        x_test = input[-self.ntest:, ::r1][:, :s1].reshape(self.ntest, s1, 1).repeat(1, 1, s2)
        x_test = x_test.reshape(self.ntest, -1, 1)
        y_test = output[-self.ntest:, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, self.T_out * self.out_dim)
        print(x_train.shape, y_train.shape)
        
        # Use appropriate normalizer based on norm_type
        if self.norm_type == 'UnitTransformer':
            x_normalizer = UnitTransformer(x_train)
        elif self.norm_type == 'UnitGaussianNormalizer':
            x_normalizer = UnitGaussianNormalizer(x_train)
        
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.cuda()

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(y_train)
                
            y_train = self.y_normalizer.encode(y_train)
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        t = np.linspace(0, 1, self.T_out)
        t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
        t_train = t.repeat(self.ntrain, 1)
        t_test = t.repeat(self.ntest, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=True,
                                                   collate_fn=self.random_collate_fn)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
                                                  batch_size=self.batch_size, shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class elas(object):
    def __init__(self, args):
        self.PATH_Sigma = args.data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
        self.PATH_XY = args.data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self):
        input_s = np.load(self.PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0)
        input_xy = np.load(self.PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

        train_s = input_s[:self.ntrain, :, None]
        test_s = input_s[-self.ntest:, :, None]
        train_xy = input_xy[:self.ntrain]
        test_xy = input_xy[-self.ntest:]

        print(input_s.shape, input_xy.shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(train_s)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(train_s)
                
            train_s = self.y_normalizer.encode(train_s)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_xy, train_xy, train_s),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_xy, test_xy, test_s),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [train_s.shape[1]]


class pipe(object):
    def __init__(self, args):
        self.INPUT_X = args.data_path + '/Pipe_X.npy'
        self.INPUT_Y = args.data_path + '/Pipe_Y.npy'
        self.OUTPUT_Sigma = args.data_path + '/Pipe_Q.npy'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((129 - 1) / r1) + 1)
        s2 = int(((129 - 1) / r2) + 1)

        inputX = np.load(self.INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(self.INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(self.OUTPUT_Sigma)[:, 0]
        output = torch.tensor(output, dtype=torch.float)
        print(input.shape, output.shape)

        x_train = input[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train, y_train),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, y_test),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class airfoil(object):
    def __init__(self, args):
        self.INPUT_X = args.data_path + '/NACA_Cylinder_X.npy'
        self.INPUT_Y = args.data_path + '/NACA_Cylinder_Y.npy'
        self.OUTPUT_Sigma = args.data_path + '/NACA_Cylinder_Q.npy'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((221 - 1) / r1) + 1)
        s2 = int(((51 - 1) / r2) + 1)

        inputX = np.load(self.INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(self.INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(self.OUTPUT_Sigma)[:, 4]
        output = torch.tensor(output, dtype=torch.float)
        print(input.shape, output.shape)

        x_train = input[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[self.ntrain:self.ntrain + self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 2)
        x_test = x_test.reshape(self.ntest, -1, 2)
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_test = y_test.reshape(self.ntest, -1, 1)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train, y_train),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, y_test),
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class darcy(object):
    def __init__(self, args):
        self.train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
        self.test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((421 - 1) / r1) + 1)
        s2 = int(((421 - 1) / r2) + 1)

        train_data = scio.loadmat(self.train_path)
        x_train = train_data['coeff'][:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(self.ntrain, -1, 1)
        x_train = torch.from_numpy(x_train).float()
        y_train = train_data['sol'][:self.ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(self.ntrain, -1, 1)
        y_train = torch.from_numpy(y_train)

        test_data = scio.loadmat(self.test_path)
        x_test = test_data['coeff'][:self.ntest, ::r1, ::r2][:, :s1, :s2]
        x_test = x_test.reshape(self.ntest, -1, 1)
        x_test = torch.from_numpy(x_test).float()
        y_test = test_data['sol'][:self.ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(self.ntest, -1, 1)
        y_test = torch.from_numpy(y_test)

        print(train_data['coeff'].shape, train_data['sol'].shape)
        print(test_data['coeff'].shape, test_data['sol'].shape)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.x_normalizer = UnitTransformer(x_train)
                self.y_normalizer = UnitTransformer(y_train)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.x_normalizer = UnitGaussianNormalizer(x_train)
                self.y_normalizer = UnitGaussianNormalizer(y_train)

            x_train = self.x_normalizer.encode(x_train)
            x_test = self.x_normalizer.encode(x_test)
            y_train = self.y_normalizer.encode(y_train)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                                  batch_size=self.batch_size, shuffle=False)
        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class ns(object):
    def __init__(self, args):
        self.data_path = args.data_path + '/NavierStokes_V1e-5_N1200_T20.mat'
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.out_dim = args.out_dim
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        
        # Validate norm_type
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((64 - 1) / r1) + 1)
        s2 = int(((64 - 1) / r2) + 1)

        data = scio.loadmat(self.data_path)
        print(data['u'].shape)
        train_a = data['u'][:self.ntrain, ::r1, ::r2, None, :self.T_in][:, :s1, :s2, :, :]
        train_a = train_a.reshape(train_a.shape[0], -1, self.out_dim * train_a.shape[-1])
        train_a = torch.from_numpy(train_a)
        train_u = data['u'][:self.ntrain, ::r1, ::r2, None, self.T_in:self.T_out + self.T_in][:, :s1, :s2, :, :]
        train_u = train_u.reshape(train_u.shape[0], -1, self.out_dim * train_u.shape[-1])
        train_u = torch.from_numpy(train_u)

        test_a = data['u'][-self.ntest:, ::r1, ::r2, None, :self.T_in][:, :s1, :s2, :, :]
        test_a = test_a.reshape(test_a.shape[0], -1, self.out_dim * test_a.shape[-1])
        test_a = torch.from_numpy(test_a)
        test_u = data['u'][-self.ntest:, ::r1, ::r2, None, self.T_in:self.T_out + self.T_in][:, :s1, :s2, :, :]
        test_u = test_u.reshape(test_u.shape[0], -1, self.out_dim * test_u.shape[-1])
        test_u = torch.from_numpy(test_u)

        if self.normalize:
            # Use appropriate normalizer based on norm_type
            if self.norm_type == 'UnitTransformer':
                self.x_normalizer = UnitTransformer(train_a)
                self.y_normalizer = UnitTransformer(train_u)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.x_normalizer = UnitGaussianNormalizer(train_a)
                self.y_normalizer = UnitGaussianNormalizer(train_u)

            train_a = self.x_normalizer.encode(train_a)
            test_a = self.x_normalizer.encode(test_a)
            train_u = self.y_normalizer.encode(train_u)

            self.x_normalizer.cuda()
            self.y_normalizer.cuda()

        x = np.linspace(0, 1, s2)
        y = np.linspace(0, 1, s1)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
        pos_train = pos.repeat(self.ntrain, 1, 1)
        pos_test = pos.repeat(self.ntest, 1, 1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                                  batch_size=self.batch_size, shuffle=False)

        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]


class pdebench_autoregressive(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.train_ratio = args.train_ratio
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.batch_size = args.batch_size
        self.out_dim = args.out_dim

    def get_loader(self):
        train_dataset = pdebench_dataset_autoregressive(file_path=self.file_path, train_ratio=self.train_ratio,
                                                        test=False,
                                                        T_in=self.T_in, T_out=self.T_out, out_dim=self.out_dim)
        test_dataset = pdebench_dataset_autoregressive(file_path=self.file_path, train_ratio=self.train_ratio,
                                                       test=True,
                                                       T_in=self.T_in, T_out=self.T_out, out_dim=self.out_dim)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, train_dataset.shapelist


class pdebench_dataset_autoregressive(Dataset):
    def __init__(self,
                 file_path: str,
                 train_ratio: int,
                 test: bool,
                 T_in: int,
                 T_out: int,
                 out_dim: int):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())
            self.shapelist = h5_file[data_list[0]]["data"].shape[1:-1]  # obtain shapelist
        self.ntrain = int(len(data_list) * train_ratio)
        self.test = test
        if not self.test:
            self.data_list = data_list[:self.ntrain]
        else:
            self.data_list = data_list[self.ntrain:]
        self.T_in = T_in
        self.T_out = T_out
        self.out_dim = out_dim

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as h5_file:
            data_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(data_group["data"], dtype="f")
            dim = len(data.shape) - 2
            T, *_, V = data.shape
            # change data shape
            data = torch.tensor(data, dtype=torch.float).movedim(0, -2).contiguous().reshape(*self.shapelist, -1)
            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(data_group["grid"]["x"], dtype="f")
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(data_group["grid"]["x"], dtype="f")
                y = np.array(data_group["grid"]["y"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing="ij")
                grid = torch.stack((X, Y), axis=-1)
            elif dim == 3:
                x = np.array(data_group["grid"]["x"], dtype="f")
                y = np.array(data_group["grid"]["y"], dtype="f")
                z = np.array(data_group["grid"]["z"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
                grid = torch.stack((X, Y, Z), axis=-1)

        return grid, data[:, :self.T_in * self.out_dim], \
            data[:, (self.T_in) * self.out_dim:(self.T_in + self.T_out) * self.out_dim]


class pdebench_steady_darcy(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.ntrain = args.ntrain
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.batch_size = args.batch_size

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        s1 = int(((128 - 1) / r1) + 1)
        s2 = int(((128 - 1) / r2) + 1)
        with h5py.File(self.file_path, "r") as h5_file:
            data_nu = np.array(h5_file['nu'], dtype='f')[:, ::r1, ::r2][:, :s1, :s2]
            data_solution = np.array(h5_file['tensor'], dtype='f')[:, :, ::r1, ::r2][:,:, :s1, :s2]
            data_nu = torch.from_numpy(data_nu)
            data_solution = torch.from_numpy(data_solution)
            x = np.array(h5_file['x-coordinate'])
            y = np.array(h5_file['y-coordinate'])
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            grid = torch.stack((X, Y), axis=-1)[None, ::r1, ::r2, :][:, :s1, :s2, :]

        grid = grid.repeat(data_nu.shape[0], 1, 1, 1)

        pos_train = grid[:self.ntrain, :, :, :].reshape(self.ntrain, -1, 2)
        x_train = data_nu[:self.ntrain, :, :].reshape(self.ntrain, -1, 1)
        y_train = data_solution[:self.ntrain, 0, :, :].reshape(self.ntrain, -1, 1)  # solutions only have 1 channel

        pos_test = grid[self.ntrain:, :, :, :].reshape(data_nu.shape[0] - self.ntrain, -1, 2)
        x_test = data_nu[self.ntrain:, :, :].reshape(data_nu.shape[0] - self.ntrain, -1, 1)
        y_test = data_solution[self.ntrain:, 0, :, :].reshape(data_nu.shape[0] - self.ntrain, -1,
                                                              1)  # solutions only have 1 channel

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                                   batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                                  batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, [s1, s2]


class car_design(object):
    def __init__(self, args):
        self.file_path = args.data_path
        self.radius = args.radius
        self.test_fold_id = 0

    def get_samples(self, obj_path):
        folds = [f'param{i}' for i in range(9)]
        samples = []
        for fold in folds:
            fold_samples = []
            files = os.listdir(os.path.join(obj_path, fold))
            for file in files:
                path = os.path.join(obj_path, os.path.join(fold, file))
                if os.path.isdir(path):
                    fold_samples.append(os.path.join(fold, file))
            samples.append(fold_samples)
        return samples  # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples

    def load_train_val_fold(self):
        samples = self.get_samples(os.path.join(self.file_path, 'training_data'))
        trainlst = []
        for i in range(len(samples)):
            if i == self.test_fold_id:
                continue
            trainlst += samples[i]
        vallst = samples[self.test_fold_id] if 0 <= self.test_fold_id < len(samples) else None

        if os.path.exists(os.path.join(self.file_path, 'preprocessed_data')):
            print("use preprocessed data")
            preprocessed = True
        else:
            preprocessed = False
        print("loading data")
        train_dataset, coef_norm = get_datalist(self.file_path, trainlst, norm=True,
                                                savedir=os.path.join(self.file_path, 'preprocessed_data'),
                                                preprocessed=preprocessed)
        val_dataset = get_datalist(self.file_path, vallst, coef_norm=coef_norm,
                                   savedir=os.path.join(self.file_path, 'preprocessed_data'),
                                   preprocessed=preprocessed)
        print("load data finish")
        return train_dataset, val_dataset, coef_norm, vallst

    def get_loader(self):
        train_data, val_data, coef_norm, vallst = self.load_train_val_fold()
        train_loader = GraphDataset(train_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm)
        test_loader = GraphDataset(val_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm, valid_list=vallst)
        return train_loader, test_loader, [train_data[0].x.shape[0]]

class cfd_3d_dataset(Dataset):
    def __init__(self, data_path, downsamplex, downsampley, downsamplez, 
                 T_in, T_out, out_dim, is_train=True, train_ratio=0.8):
        self.data_path = data_path
        self.T_in = T_in
        self.T_out = T_out
        self.out_dim = out_dim
        self.is_train = is_train
        
        # Calculate grid sizes
        self.r1 = downsamplex
        self.r2 = downsampley
        self.r3 = downsamplez
        self.s1 = int(((128 - 1) / self.r1) + 1)
        self.s2 = int(((128 - 1) / self.r2) + 1)
        self.s3 = int(((128 - 1) / self.r3) + 1)
        
        # Create position grid once (reused for all samples)
        with h5py.File(data_path, 'r') as h5_file:
            x_coords = np.array(h5_file['x-coordinate'][::self.r1])[:self.s1]
            y_coords = np.array(h5_file['y-coordinate'][::self.r2])[:self.s2]
            z_coords = np.array(h5_file['z-coordinate'][::self.r3])[:self.s3]
            
            # Create grid
            x = torch.tensor(x_coords, dtype=torch.float)
            y = torch.tensor(y_coords, dtype=torch.float)
            z = torch.tensor(z_coords, dtype=torch.float)
            X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
            self.grid = torch.stack((X, Y, Z), axis=-1)
            self.grid_flat = self.grid.reshape(-1, 3)

            first_field = sorted(h5_file.keys())[0]
            num_samples = h5_file[first_field].shape[0]
            self.ntrain = int(num_samples * train_ratio)
            
            # Set indices based on train or test
            if self.is_train:
                self.indices = np.arange(self.ntrain)
            else:
                self.indices = np.arange(self.ntrain, num_samples)
        
        self.fields = ['Vx', 'Vy', 'Vz', 'pressure', 'density']
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        
        # Initialize data arrays for this sample only (much smaller memory footprint)
        a_data = np.zeros((self.grid_flat.shape[0], self.T_in * self.out_dim))
        u_data = np.zeros((self.grid_flat.shape[0], self.T_out * self.out_dim))
        # import pdb; pdb.set_trace()

        
        with h5py.File(self.data_path, 'r') as h5_file:
            # Load input timesteps
            for t_in in range(self.T_in):
                for f_idx, field in enumerate(self.fields):
                    var_data = h5_file[field][sample_idx, t_in, ::self.r1, ::self.r2, ::self.r3][:self.s1, :self.s2, :self.s3]
                    var_data_flat = var_data.reshape(-1)
                    a_data[:, t_in*self.out_dim + f_idx] = var_data_flat
            
            # Load output timesteps
            for t_out in range(self.T_out):
                for f_idx, field in enumerate(self.fields):
                    var_data = h5_file[field][sample_idx, self.T_in + t_out, ::self.r1, ::self.r2, ::self.r3][:self.s1, :self.s2, :self.s3]
                    var_data_flat = var_data.reshape(-1)
                    u_data[:, t_out*self.out_dim + f_idx] = var_data_flat
        
        # Convert to tensors
        a_data = torch.tensor(a_data, dtype=torch.float)
        u_data = torch.tensor(u_data, dtype=torch.float)
        
        
        return self.grid_flat, a_data, u_data

class cfd3d(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley
        self.downsamplez = args.downsamplez
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio
        self.out_dim = args.out_dim
        self.T_in = args.T_in
        self.T_out = args.T_out
        

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley
        r3 = self.downsamplez
        s1 = int(((128 - 1) / r1) + 1)
        s2 = int(((128 - 1) / r2) + 1)
        s3 = int(((128 - 1) / r3) + 1)
        
        train_dataset = cfd_3d_dataset(
            self.data_path, self.downsamplex, self.downsampley, self.downsamplez,
            self.T_in, self.T_out, self.out_dim, is_train=True, 
            train_ratio=self.train_ratio,
        )
        
        test_dataset = cfd_3d_dataset(
            self.data_path, self.downsamplex, self.downsampley, self.downsamplez,
            self.T_in, self.T_out, self.out_dim, is_train=False, 
            train_ratio=self.train_ratio,
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        return train_loader, test_loader, [s1, s2, s3]
    


import os
import glob
import netCDF4 as nc
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF


class PocFluxDataset(Dataset):
    """
    用于海洋碳通量数据的PyTorch数据集类
    输出格式兼容现有框架: (pos, input_data, output_data)
    内部保存mask用于评估时过滤缺失值
    """
    def __init__(self, mode, data_path, seq_len, img_size, 
                 T_in=1, T_out=1,
                 need_name=False, 
                 train_ratio=0.8, valid_ratio=0.1, 
                 precomputed_norm_params=None,
                 crop_config=None):
        """
        初始化数据集
        
        参数说明:
        - mode: 'train', 'valid', 或 'test'
        - data_path: NetCDF数据根目录
        - seq_len: 总时间序列长度 (T_in + T_out)
        - img_size: 输出正方形图像尺寸
        - T_in: 输入时间步数
        - T_out: 输出时间步数
        - need_name: 是否返回文件名
        - train_ratio/valid_ratio: 数据集划分比例
        - precomputed_norm_params: 预计算的归一化参数 {'min_val': x, 'max_val': y}
        - crop_config: 裁剪配置 {'top': x, 'left': y, 'height': h, 'width': w}
        """
        super().__init__()
        assert mode in ['train', 'valid', 'test'], "Mode must be one of ['train', 'valid', 'test']"
        
        self.mode = mode
        self.data_path = data_path
        self.seq_len = seq_len
        self.img_size = img_size
        self.T_in = T_in
        self.T_out = T_out
        self.need_name = need_name
        self.crop_config = crop_config

        # 加载数据标识
        all_data_identifiers = self._load_keys()
        if not all_data_identifiers:
            raise FileNotFoundError(f"在路径 {self.data_path} 下没有找到任何 NetCDF 数据")
            
        # 创建样本
        all_samples = self._create_all_samples(all_data_identifiers)

        # 数据集划分
        num_samples = len(all_samples)
        train_end = int(num_samples * train_ratio)
        valid_end = int(num_samples * (train_ratio + valid_ratio))

        if self.mode == 'train':
            self.samples = all_samples[:train_end]
        elif self.mode == 'valid':
            self.samples = all_samples[train_end:valid_end]
        else:
            self.samples = all_samples[valid_end:]

        # 归一化参数
        if precomputed_norm_params:
            self.min_val = precomputed_norm_params['min_val']
            self.max_val = precomputed_norm_params['max_val']
            print(f"模式 '{self.mode}' 加载完成，使用预计算的归一化参数。共 {len(self.samples)} 个样本。")
        else:
            print(f"模式 '{self.mode}' 加载中, 需要计算归一化参数...")
            train_samples_for_norm = all_samples[:train_end]
            self.min_val, self.max_val = self._calculate_normalization_params(train_samples_for_norm)
        
        if self.mode == 'train':
            print(f"数据归一化范围 (min, max): ({self.min_val:.4f}, {self.max_val:.4f})")
        
        # 创建位置网格 (只创建一次)
        self._create_position_grid()

    def _create_position_grid(self):
        """创建标准化的位置网格 [0,1] x [0,1]"""
        x = torch.linspace(0, 1, self.img_size)
        y = torch.linspace(0, 1, self.img_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        # shape: (H, W, 2)
        self.pos_grid = torch.stack([grid_x, grid_y], dim=-1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取一个样本
        
        返回格式兼容 Exp_Steady:
        - pos: (N, 2) 位置坐标
        - fx: (N, T_in) 输入观测值
        - y: (N, T_out) 输出观测值
        
        同时内部保存mask信息供评估使用
        """
        sample_identifiers = self.samples[idx]
        frames = []
        masks = []
        
        # 裁剪配置
        if self.crop_config:
            top = self.crop_config['top']
            left = self.crop_config['left']
            height = self.crop_config['height']
            width = self.crop_config['width']
        
        for nc_path, yyyymm_key in sample_identifiers:
            try:
                with nc.Dataset(nc_path, 'r') as ncfile:
                    fgco2_var = ncfile.variables['fgco2']
                    
                    # 读取数据(带裁剪)
                    if self.crop_config:
                        frame_data = fgco2_var[0, top:top+height, left:left+width]
                    else:
                        frame_data = fgco2_var[0, :, :]

                    frame_data = frame_data.astype(np.float32)
                    
                    # 创建掩码(在替换NaN之前)
                    valid_mask = ~(np.isnan(frame_data) | (np.abs(frame_data) > 1e30))
                    
                    # 将无效数据替换为0用于模型输入
                    frame_data = np.where(valid_mask, frame_data, 0.0)
                    
                    frames.append(frame_data)
                    masks.append(valid_mask)
            
            except Exception as e:
                print(f"读取文件 {nc_path} 时出错: {e}")
                if self.crop_config:
                    error_shape = (self.crop_config['height'], self.crop_config['width'])
                else:
                    error_shape = (713, 1440)
                    if len(frames) > 0:
                        error_shape = frames[0].shape
                frames.append(np.zeros(error_shape, dtype=np.float32))
                masks.append(np.zeros(error_shape, dtype=bool))

        # 堆叠: (T, H, W)
        data_array = np.stack(frames, axis=0)
        mask_array = np.stack(masks, axis=0)
        
        # 数据归一化
        epsilon = 1e-8
        data_array = 2 * (data_array - self.min_val) / (self.max_val - self.min_val + epsilon) - 1
        data_array = np.clip(data_array, -1, 1)
        
        # 转换为Tensor并增加通道维度: (T, H, W) -> (T, 1, H, W)
        data_tensor = torch.as_tensor(data_array, dtype=torch.float).unsqueeze(1)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.bool).unsqueeze(1)
        
        # 填充为正方形
        _t, _c, h, w = data_tensor.shape
        if h != w:
            padding_left = (max(h, w) - w) // 2
            padding_right = max(h, w) - w - padding_left
            padding_top = (max(h, w) - h) // 2
            padding_bottom = max(h, w) - h - padding_top
            
            data_tensor = TF.pad(data_tensor, 
                                [padding_left, padding_top, padding_right, padding_bottom], 
                                fill=0)
            mask_tensor = TF.pad(mask_tensor, 
                                [padding_left, padding_top, padding_right, padding_bottom], 
                                fill=False)

        # 缩放到目标尺寸
        data_tensor = TF.resize(data_tensor, [self.img_size, self.img_size], 
                               interpolation=TF.InterpolationMode.BILINEAR, 
                               antialias=True)
        
        mask_tensor = TF.resize(mask_tensor.float(), [self.img_size, self.img_size], 
                               interpolation=TF.InterpolationMode.NEAREST)
        mask_tensor = mask_tensor.bool()
        
        # 分离输入和输出
        # data_tensor shape: (T, 1, H, W)
        input_frames = data_tensor[:self.T_in]  # (T_in, 1, H, W)
        output_frames = data_tensor[self.T_in:self.T_in+self.T_out]  # (T_out, 1, H, W)
        output_mask = mask_tensor[self.T_in:self.T_in+self.T_out]  # (T_out, 1, H, W)
        
        # 转换为 (N, T_in) 和 (N, T_out) 格式
        # N = H * W
        N = self.img_size * self.img_size
        
        # pos: (H, W, 2) -> (N, 2)
        pos = self.pos_grid.reshape(N, 2)
        
        # fx: (T_in, 1, H, W) -> (N, T_in)
        fx = input_frames.squeeze(1).reshape(self.T_in, N).permute(1, 0)  # (N, T_in)
        
        # y: (T_out, 1, H, W) -> (N, T_out)
        y = output_frames.squeeze(1).reshape(self.T_out, N).permute(1, 0)  # (N, T_out)
        
        # mask: (T_out, 1, H, W) -> (N, T_out)
        mask = output_mask.squeeze(1).reshape(self.T_out, N).permute(1, 0)  # (N, T_out)
        
        # 返回格式: (pos, fx, y), 并将mask保存在y中
        # 我们使用一个技巧: 将mask信息编码到tensor的属性中
        # y.mask = mask  # 添加mask属性用于评估
        
        return pos, fx, y, mask

    def _load_keys(self):
        """扫描并加载所有NetCDF文件路径和月份标识"""
        search_pattern = os.path.join(self.data_path, "**", "*.nc")
        nc_files = sorted(glob.glob(search_pattern, recursive=True))
        
        monthly_identifiers = []
        print(f"发现 {len(nc_files)} 个NetCDF文件，正在扫描...")
        
        for nc_path in tqdm(nc_files, desc="扫描NetCDF文件"):
            try:
                filename = os.path.basename(nc_path)
                yyyymm = filename.split('_')[-1].replace('.nc', '')
                
                with nc.Dataset(nc_path, 'r') as ncfile:
                    if 'fgco2' in ncfile.variables:
                        monthly_identifiers.append((nc_path, yyyymm))
                    else:
                        print(f"警告: 文件 {nc_path} 中没有 'fgco2' 变量")
                        
            except Exception as e:
                print(f"警告: 无法读取文件 {nc_path}: {e}")

        monthly_identifiers.sort(key=lambda x: x[1])
        print(f"共找到 {len(monthly_identifiers)} 个有效月度数据。")
        return monthly_identifiers

    def _create_all_samples(self, data_list):
        """根据序列长度创建滑动窗口样本"""
        num_frames = len(data_list)
        if num_frames < self.seq_len:
            raise ValueError(f"数据总帧数({num_frames})小于序列长度({self.seq_len})")
        samples = []
        for i in range(num_frames - self.seq_len + 1):
            samples.append(data_list[i : i + self.seq_len])
        return samples
        
    def _calculate_normalization_params(self, train_samples):
        """基于训练集计算归一化参数(仅在有效数据区域)"""
        print("正在基于训练集计算归一化参数(忽略NaN和填充值)...")
        min_val, max_val = np.inf, -np.inf
        
        unique_train_data = sorted(list(set(item for sample in train_samples for item in sample)))
        
        if self.crop_config:
            top = self.crop_config['top']
            left = self.crop_config['left']
            height = self.crop_config['height']
            width = self.crop_config['width']
            print(f"将在裁剪区域 [T:{top}, L:{left}, H:{height}, W:{width}] 内计算")
        
        for nc_path, yyyymm_key in tqdm(unique_train_data, desc="计算归一化参数"):
            try:
                with nc.Dataset(nc_path, 'r') as ncfile:
                    fgco2_var = ncfile.variables['fgco2']
                    
                    if self.crop_config:
                        frame_data = fgco2_var[0, top:top+height, left:left+width]
                    else:
                        frame_data = fgco2_var[0, :, :]
                    
                    valid_mask = ~(np.isnan(frame_data) | (np.abs(frame_data) > 1e30))
                    
                    if valid_mask.any():
                        valid_data = frame_data[valid_mask]
                        min_val = min(min_val, valid_data.min())
                        max_val = max(max_val, valid_data.max())
                        
            except Exception as e:
                print(f"警告: 跳过 {nc_path} (原因: {e})")

        return min_val, max_val



class poc_flux(object):
    """
    包装类，用于匹配现有的数据加载框架接口
    """
    def __init__(self, args):
        self.data_path = args.data_path
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.seq_len = args.T_in + args.T_out
        self.img_size = args.img_size if hasattr(args, 'img_size') else 128
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio if hasattr(args, 'train_ratio') else 0.8
        self.valid_ratio = getattr(args, 'valid_ratio', 0.1)
        
        # 裁剪配置(如果需要)
        self.crop_config = None
        # 修改：同时检查属性存在 且 值不为None
        if hasattr(args, 'crop_top') and args.crop_top is not None:
            self.crop_config = {
                'top': int(args.crop_top),       # 建议强制转为 int，防止类型错误
                'left': int(args.crop_left),
                'height': int(args.crop_height),
                'width': int(args.crop_width)
            }
        
        # 先创建训练集以计算归一化参数
        print("正在加载训练数据集...")
        train_dataset_temp = PocFluxDataset(
            mode='train',
            data_path=self.data_path,
            seq_len=self.seq_len,
            img_size=self.img_size,
            T_in=self.T_in,
            T_out=self.T_out,
            need_name=False,
            train_ratio=self.train_ratio,
            valid_ratio=self.valid_ratio,
            precomputed_norm_params=None,
            crop_config=self.crop_config
        )
        
        # 保存归一化参数
        self.norm_params = {
            'min_val': train_dataset_temp.min_val,
            'max_val': train_dataset_temp.max_val
        }
        
        # POC flux数据集不需要y_normalizer (数据已经归一化到[-1,1])
        self.y_normalizer = None

    def get_loader(self):
        """
        返回训练和测试数据加载器以及shape信息
        
        返回格式: train_loader, test_loader, shapelist
        """
        # 创建训练数据集
        train_dataset = PocFluxDataset(
            mode='train',
            data_path=self.data_path,
            seq_len=self.seq_len,
            img_size=self.img_size,
            T_in=self.T_in,
            T_out=self.T_out,
            need_name=False,
            train_ratio=self.train_ratio,
            valid_ratio=self.valid_ratio,
            precomputed_norm_params=self.norm_params,
            crop_config=self.crop_config
        )
        
        # 创建测试数据集
        test_dataset = PocFluxDataset(
            mode='test',
            data_path=self.data_path,
            seq_len=self.seq_len,
            img_size=self.img_size,
            T_in=self.T_in,
            T_out=self.T_out,
            need_name=False,
            train_ratio=self.train_ratio,
            valid_ratio=self.valid_ratio,
            precomputed_norm_params=self.norm_params,
            crop_config=self.crop_config
        )
        
        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # shapelist: [img_size, img_size] 用于2D数据
        shapelist = [self.img_size, self.img_size]
        
        print("数据加载完成。")
        print(f"训练样本数: {len(train_dataset)}")
        print(f"测试样本数: {len(test_dataset)}")
        print(f"图像尺寸: {self.img_size}x{self.img_size}")
        print(f"输入时间步: {self.T_in}, 输出时间步: {self.T_out}")
        
        return train_loader, test_loader, shapelist



class OceanSodaDataset(Dataset):
    """
    修改版: 支持非正方形网格 (Lat!=Lon)，移除强制 Padding 和 Resize
    """
    def __init__(self, mode, data_path, seq_len, 
                 T_in=1, T_out=1,
                 need_name=False, 
                 train_ratio=0.8, valid_ratio=0.1, 
                 precomputed_norm_params=None,
                 crop_config=None,
                 cache_data=True,
                 target_shape=None): # 新增 target_shape 参数
        super().__init__()
        self.mode = mode
        self.data_path = data_path
        self.seq_len = seq_len
        # 如果 target_shape 为 None，则使用原始分辨率 (720, 1440)
        self.target_shape = target_shape 
        self.T_in = T_in
        self.T_out = T_out
        self.need_name = need_name
        self.crop_config = crop_config
        self.cache_data = cache_data

        # 加载数据标识
        self.all_identifiers = self._load_keys()
        if not self.all_identifiers:
            raise FileNotFoundError(f"在路径 {self.data_path} 下没有找到任何 NetCDF 数据")

        # 获取原始数据尺寸 (H, W)
        self.original_h, self.original_w = self._get_original_shape()
        
        # 确定最终使用的尺寸
        if self.target_shape is None:
            self.h, self.w = self.original_h, self.original_w
        else:
            self.h, self.w = self.target_shape

        print(f"数据集尺寸配置: 原始({self.original_h}, {self.original_w}) -> 使用({self.h}, {self.w})")

        # 缓存逻辑 (保持不变)
        self.cached_data = None
        self.cached_masks = None
        if self.cache_data:
            self._load_cache()
            source_data = list(range(len(self.all_identifiers)))
        else:
            source_data = self.all_identifiers
            
        all_samples = self._create_all_samples(source_data)

        # 数据集划分 (保持不变)
        num_samples = len(all_samples)
        train_end = int(num_samples * train_ratio)
        valid_end = int(num_samples * (train_ratio + valid_ratio))

        if self.mode == 'train':
            self.samples = all_samples[:train_end]
        elif self.mode == 'valid':
            self.samples = all_samples[train_end:valid_end]
        else:
            self.samples = all_samples[valid_end:]

        # 归一化参数 (保持不变)
        if precomputed_norm_params:
            self.min_val = precomputed_norm_params['min_val']
            self.max_val = precomputed_norm_params['max_val']
        else:
            if self.cache_data and len(self.samples) > 0:
                start_idx = self.samples[0][0]
                end_idx = self.samples[-1][-1] + 1
                train_data_slice = self.cached_data[start_idx:end_idx]
                train_mask_slice = self.cached_masks[start_idx:end_idx]
                valid_data = train_data_slice[train_mask_slice]
                self.min_val = valid_data.min()
                self.max_val = valid_data.max()
            else:
                train_samples_for_norm = all_samples[:train_end]
                self.min_val, self.max_val = self._calculate_normalization_params(train_samples_for_norm)
        
        self._create_position_grid()

    def _get_original_shape(self):
        """读取第一个文件获取原始尺寸"""
        first_path = self.all_identifiers[0][0]
        with nc.Dataset(first_path, 'r') as ncfile:
            if self.crop_config:
                return self.crop_config['height'], self.crop_config['width']
            shape = ncfile.variables['fgco2'].shape
            return shape[-2], shape[-1] # lat, lon

    def _load_cache(self):
        """修改: 使用 self.original_h 和 self.original_w"""
        print("正在将所有数据加载到内存中 (cache_data=True)...")
        from collections import defaultdict
        file_to_indices = defaultdict(list)
        for global_idx, (nc_path, t_idx) in enumerate(self.all_identifiers):
            file_to_indices[nc_path].append((t_idx, global_idx))
            
        total_frames = len(self.all_identifiers)
        # 使用原始尺寸缓存
        self.cached_data = np.zeros((total_frames, self.original_h, self.original_w), dtype=np.float32)
        self.cached_masks = np.zeros((total_frames, self.original_h, self.original_w), dtype=bool)
        
        # ... (读取数据的逻辑保持不变，只需确保读取时不 Resize) ...
        # 注意: 如果 crop_config 存在，self.original_h/w 已经是 crop 后的尺寸，这里逻辑通用
        
        for nc_path in tqdm(sorted(file_to_indices.keys()), desc="加载文件"):
            try:
                indices_map = file_to_indices[nc_path]
                with nc.Dataset(nc_path, 'r') as ncfile:
                    fgco2_var = ncfile.variables['fgco2']
                    if self.crop_config:
                        # 假设 crop_config 存在
                        top, left = self.crop_config['top'], self.crop_config['left']
                        h, w = self.crop_config['height'], self.crop_config['width']
                        full_data = fgco2_var[:, top:top+h, left:left+w]
                    else:
                        full_data = fgco2_var[:] 
                        
                    for t_idx, global_idx in indices_map:
                        if t_idx < full_data.shape[0]:
                            frame_data = full_data[t_idx]
                            valid_mask = ~(np.isnan(frame_data) | (np.abs(frame_data) > 1e30))
                            frame_data = np.where(valid_mask, frame_data, 0.0)
                            self.cached_data[global_idx] = frame_data
                            self.cached_masks[global_idx] = valid_mask
            except Exception as e:
                print(f"Err: {e}")

    def _create_position_grid(self):
        """修改: 支持长方形网格"""
        x = torch.linspace(0, 1, self.w) # lon
        y = torch.linspace(0, 1, self.h) # lat
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.pos_grid = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        if self.cache_data:
            start_idx = sample_info[0]
            end_idx = sample_info[-1] + 1
            data_array = self.cached_data[start_idx:end_idx].copy()
            mask_array = self.cached_masks[start_idx:end_idx].copy()
        else:
            frames, masks = [], []
            for idx_item in sample_info:
                if isinstance(idx_item, int):
                    nc_path, t_idx = self.all_identifiers[idx_item]
                else:
                    nc_path, t_idx = idx_item
                with nc.Dataset(nc_path, 'r') as ncfile:
                    fg = ncfile.variables['fgco2']
                    if self.crop_config:
                        top, left = self.crop_config['top'], self.crop_config['left']
                        h, w = self.crop_config['height'], self.crop_config['width']
                        frame = fg[t_idx, top:top+h, left:left+w]
                    else:
                        frame = fg[t_idx, :, :]
                    frame = frame.astype(np.float32)
                    valid = ~(np.isnan(frame) | (np.abs(frame) > 1e30))
                    frames.append(np.where(valid, frame, 0.0))
                    masks.append(valid)
            data_array = np.stack(frames, axis=0)
            mask_array = np.stack(masks, axis=0)
        
        # 归一化
        epsilon = 1e-8
        data_array = 2 * (data_array - self.min_val) / (self.max_val - self.min_val + epsilon) - 1
        data_array = np.clip(data_array, -1, 1)
        
        data_tensor = torch.as_tensor(data_array, dtype=torch.float).unsqueeze(1) # (T, 1, H, W)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.bool).unsqueeze(1)
        
        # === 关键修改: 移除强制正方形 Padding 和 Resize ===
        # 只有当 self.target_shape 与当前 shape 不一致时才 Resize
        if (self.h, self.w) != (self.original_h, self.original_w):
             data_tensor = TF.resize(data_tensor, [self.h, self.w], antialias=True)
             mask_tensor = TF.resize(mask_tensor.float(), [self.h, self.w], interpolation=TF.InterpolationMode.NEAREST).bool()
        
        # 分离输入输出
        input_frames = data_tensor[:self.T_in]
        output_frames = data_tensor[self.T_in:self.T_in+self.T_out]
        output_mask = mask_tensor[self.T_in:self.T_in+self.T_out]
        
        N = self.h * self.w
        pos = self.pos_grid.reshape(N, 2)
        fx = input_frames.squeeze(1).reshape(self.T_in, N).permute(1, 0)
        y = output_frames.squeeze(1).reshape(self.T_out, N).permute(1, 0)
        mask = output_mask.squeeze(1).reshape(self.T_out, N).permute(1, 0)
        
        return pos, fx, y, mask

    def __len__(self):
        return len(self.samples)

    def _load_keys(self):
        search_pattern = os.path.join(self.data_path, "**", "*.nc")
        nc_files = sorted(glob.glob(search_pattern, recursive=True))
        identifiers = []
        print(f"发现 {len(nc_files)} 个NetCDF文件，正在索引时间帧...")
        for nc_path in nc_files:
            try:
                with nc.Dataset(nc_path, 'r') as ds:
                    if 'fgco2' not in ds.variables:
                        continue
                    t_len = ds.variables['fgco2'].shape[0]
                    for t_idx in range(t_len):
                        identifiers.append((nc_path, t_idx))
            except Exception as e:
                print(f"警告: 无法读取文件 {nc_path}: {e}")
        print(f"共索引到 {len(identifiers)} 个时间帧。")
        return identifiers

    def _create_all_samples(self, data_list):
        num_frames = len(data_list)
        if num_frames < self.seq_len:
            raise ValueError(f"数据总帧数({num_frames})小于序列长度({self.seq_len})")
        if isinstance(data_list[0], int):
            indices = data_list
        else:
            indices = list(range(num_frames))
        return [indices[i:i+self.seq_len] for i in range(num_frames - self.seq_len + 1)]  # 滑动窗口步长固定为1

    def _calculate_normalization_params(self, train_samples):
        min_val, max_val = np.inf, -np.inf
        unique_indices = sorted(set(idx for sample in train_samples for idx in sample))
        for idx in unique_indices:
            nc_path, t_idx = self.all_identifiers[idx]
            try:
                with nc.Dataset(nc_path, 'r') as ds:
                    fg = ds.variables['fgco2']
                    if self.crop_config:
                        top, left = self.crop_config['top'], self.crop_config['left']
                        h, w = self.crop_config['height'], self.crop_config['width']
                        frame = fg[t_idx, top:top+h, left:left+w]
                    else:
                        frame = fg[t_idx, :, :]
                    valid = ~(np.isnan(frame) | (np.abs(frame) > 1e30))
                    if valid.any():
                        vals = frame[valid]
                        min_val = min(min_val, float(vals.min()))
                        max_val = max(max_val, float(vals.max()))
            except Exception as e:
                print(f"警告: 跳过 {nc_path} (原因: {e})")
        return min_val, max_val


class ocean_soda(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.seq_len = args.T_in + args.T_out
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio if hasattr(args, 'train_ratio') else 0.8
        self.valid_ratio = getattr(args, 'valid_ratio', 0.1)
        
        # 设定目标尺寸: 如果 args.img_size 为默认值(通常较小)且你想用原图，
        # 建议在此处强制指定为 None，或者根据 args 逻辑判断
        # 这里设置为 None 以使用原始尺寸 (720, 1440)
        self.target_shape = None 
        # 如果你想通过命令行控制，可以这样写:
        # self.target_shape = [args.img_size, args.img_size] if hasattr(args, 'force_resize') else None

        self.crop_config = None
        if hasattr(args, 'crop_top') and args.crop_top is not None:
            self.crop_config = {
                'top': int(args.crop_top),
                'left': int(args.crop_left),
                'height': int(args.crop_height),
                'width': int(args.crop_width)
            }

        # 1. 预加载以计算归一化
        print("正在初始化 OceanSodaDataset (Train)...")
        train_dataset_temp = OceanSodaDataset(
            mode='train',
            data_path=self.data_path,
            seq_len=self.seq_len,
            T_in=self.T_in, T_out=self.T_out,
            train_ratio=self.train_ratio, valid_ratio=self.valid_ratio,
            target_shape=self.target_shape, # 传入 None 使用原图
            crop_config=self.crop_config,
            cache_data=True
        )
        
        self.norm_params = {
            'min_val': train_dataset_temp.min_val,
            'max_val': train_dataset_temp.max_val
        }
        self.y_normalizer = None
        
        # 保存 shape 信息供 get_loader 返回
        self.h = train_dataset_temp.h
        self.w = train_dataset_temp.w

    def get_loader(self):
        common_args = {
            'data_path': self.data_path,
            'seq_len': self.seq_len,
            'T_in': self.T_in, 'T_out': self.T_out,
            'train_ratio': self.train_ratio, 'valid_ratio': self.valid_ratio,
            'precomputed_norm_params': self.norm_params,
            'crop_config': self.crop_config,
            'cache_data': True,
            'target_shape': self.target_shape # 关键
        }

        train_dataset = OceanSodaDataset(mode='train', **common_args)
        test_dataset = OceanSodaDataset(mode='test', **common_args)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        # 返回准确的 shapelist [H, W] = [720, 1440]
        shapelist = [self.h, self.w]
        
        print(f"数据加载完成: Train={len(train_dataset)}, Test={len(test_dataset)}")
        print(f"空间分辨率: {shapelist}")
        
        return train_loader, test_loader, shapelist