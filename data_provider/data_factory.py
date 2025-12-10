from data_provider.data_loader import airfoil, ns, darcy, pipe, elas, plas, pdebench_autoregressive, \
    pdebench_steady_darcy, car_design, cfd3d, poc_flux  # 添加 poc_flux


def get_data(args):
    data_dict = {
        'car_design': car_design,
        'pdebench_autoregressive': pdebench_autoregressive,
        'pdebench_steady_darcy': pdebench_steady_darcy,
        'elas': elas,
        'pipe': pipe,
        'airfoil': airfoil,
        'darcy': darcy,
        'ns': ns,
        'plas': plas,
        'cfd3d': cfd3d,
        'poc_flux': poc_flux,  # 添加新数据集
        'ocean_soda': ocean_soda,  # 添加新数据集
    }
    dataset = data_dict[args.loader](args) # 初始化数据集对应的loader类，就是通过args.loader这个参数查表，然后调用对应的类的初始化函数
    train_loader, test_loader, shapelist = dataset.get_loader()
    return dataset, train_loader, test_loader, shapelist