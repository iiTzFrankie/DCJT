import importlib
import torch
import yacs.config

def create_model(config):
    module = importlib.import_module(f'task.models.{config.model.name}')
    model = module.Model()
    return model
