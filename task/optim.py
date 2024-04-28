import torch
import yacs.config
import torch.nn.modules as modules




def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if not str(layer.__class__).startswith("<class 'torch.nn.modules"):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def get_param_list(config, model):
    if config.train.no_weight_decay_on_bn:
        params_only_bn, params_wo_bn = separate_bn_paras(model)
        param_list = [{'params': params_wo_bn}, 
           {'params': params_only_bn, 'weight_decay': 0.0}]
        
    else:
        param_list = [{
            'params': list(model.parameters()),
            'weight_decay': config.train.weight_decay,
        }]
    return param_list





def get_param_list_ori(config, model): # original get param 
    if config.train.no_weight_decay_on_bn:
        param_list = []
        for name, params in model.named_parameters():
            if 'conv.weight' in name:
                param_list.append({
                    'params': params,
                    'weight_decay': config.train.weight_decay,
                    })
            else:
                param_list.append({
                    'params': params,
                    'weight_decay': 0,
                    })
    else:
        param_list = [{
            'params': list(model.parameters()),
            'weight_decay': config.train.weight_decay,
        }]
    return param_list

def create_optimizer(config, model):
    params = get_param_list(config, model)

    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr = config.train.base_lr,
                                    momentum = config.train.momentum,
                                    nesterov = config.train.nesterov)

    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr = config.train.base_lr,)
                                     #betas = config.optim.betas)

    elif config.train.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(params,
                                     lr = config.train.base_lr,
                                     betas = config.optim.adam.betas,
                                     amsgrad = True)

    else:
        raise ValueError()

    return optimizer
    

def create_optimizer1(config, model, sigma1, sigma2):
    params = get_param_list(config, model)
    params.append({
        'params': sigma1,
        'weight_decay': 0,
    })
    params.append({
        'params': sigma2,
        'weight_decay': 0,
    })

    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr = config.train.base_lr,
                                    momentum = config.train.momentum,
                                    nesterov = config.train.nesterov)

    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr = config.train.base_lr,)
                                     #betas = config.optim.betas)

    elif config.train.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(params,
                                     lr = config.train.base_lr,
                                     betas = config.optim.adam.betas,
                                     amsgrad = True)

    else:
        raise ValueError()

    return optimizer
