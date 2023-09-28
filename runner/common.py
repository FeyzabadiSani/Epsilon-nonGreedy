import numpy as np
from DSG.r3DSG import get_unobserved_batch
from util.common import *
from torch.utils.data import TensorDataset, DataLoader
import torch
from config import device, TRAIN_ORDER
import logging
import os
from DSG.common import cartesian_product
from copy import deepcopy


def dub_training(student_config: dict, teacher_config: dict, n_epochs: int, teacher_train_data: np.ndarray,
                 student_train_data: np.ndarray, validation_data: np.ndarray, unobserved_data: np.ndarray,
                 path: str, conv_wait_steps: int, dataset_name: DatasetName):
    # Set train hyperparameters
    attempts = 0
    minimum_validation_loss = np.inf
    # Moving model to GPU and initialize optimizer
    student_config['architecture'].to(device)
    teacher_config['architecture'].to(device)
    student_optimizer = torch.optim.Adam(
        student_config['architecture'].parameters(),
        weight_decay=student_config['weight_decay'],
    )
    student_config['architecture'].train()
    teacher_config['architecture'].eval()
    # Constructing training and validating tensors
    y_val = torch.tensor(validation_data[:, -1].reshape(-1, 1), device=device, dtype=torch.float)
    if dataset_name == DatasetName.coat:    
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.float)
        student_dataset = TensorDataset(
            torch.tensor(student_train_data[:, :-1], dtype=torch.float),
            torch.tensor(student_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
        teacher_dataset = TensorDataset(
            torch.tensor(teacher_train_data[:, :-1], dtype=torch.float),
            torch.tensor(teacher_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    elif dataset_name == DatasetName.yahooR3:
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.long)
        student_dataset = TensorDataset(
            torch.tensor(student_train_data[:, :-1], dtype=torch.long),
            torch.tensor(student_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
        teacher_dataset = TensorDataset(
            torch.tensor(teacher_train_data[:, :-1], dtype=torch.long),
            torch.tensor(teacher_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )

    student_dataloader = DataLoader(
        student_dataset,
        student_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    teacher_dataloader = DataLoader(
        teacher_dataset,
        teacher_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    if dataset_name == DatasetName.coat:
        unobserved_dataset = TensorDataset(
            torch.tensor(unobserved_data, dtype=torch.float)
        )
        unobserved_dataloader = DataLoader(
            unobserved_dataset,
            student_config['unobserved_batch_size'],
            shuffle=True,
            drop_last=True
        )
    n = student_config['unobserved_batch_size'] + student_config['batch_size'] + teacher_config['batch_size']
    # Train epochs
    for epoch in range(1, n_epochs + 1):
        # Checking whether training procedure converged
        if attempts > conv_wait_steps:
            return epoch
        student_iterator = iter(student_dataloader)
        teacher_iterator = iter(teacher_dataloader)
        if dataset_name == DatasetName.coat:
            n_batches = min(len(student_dataloader), len(teacher_dataloader), len(unobserved_dataloader))
            unobserved_iterator = iter(unobserved_dataloader)
        elif dataset_name == DatasetName.yahooR3:
            n_batches = min(len(student_dataloader), len(teacher_dataloader))
        for idx in range(n_batches):
            # Moving data to device
            student_x, student_y = next(student_iterator)
            teacher_x, teacher_y = next(teacher_iterator)
            student_x, student_y = student_x.to(device), student_y.to(device)
            teacher_x, teacher_y = teacher_x.to(device), teacher_y.to(device)
            if dataset_name == DatasetName.coat:
                x_unobserved = next(unobserved_iterator)
                x_unobserved = x_unobserved[0].to(device)
            elif dataset_name == DatasetName.yahooR3:
                x_unobserved = get_unobserved_batch(device, student_config['unobserved_batch_size'])
            # Calculating loss and updating model parameters
            teacher_output_teacher_dataset = teacher_config['architecture'](teacher_x)
            teacher_output_student_dataset = teacher_config['architecture'](student_x)
            student_output_teacher_dataset = student_config['architecture'](teacher_x)
            student_output_student_dataset = student_config['architecture'](student_x)
            teacher_output_unobserved_dataset = teacher_config['architecture'](x_unobserved)
            student_output_unobserved_dataset = student_config['architecture'](x_unobserved)
            # Loss implemented from equation 13 of the paper
            loss_a = bce_sum_loss(student_output_teacher_dataset, teacher_y) / n
            loss_c = bce_sum_loss(student_output_student_dataset, student_y) / n
            loss_e2 = bce_loss(student_output_teacher_dataset, teacher_y - teacher_output_teacher_dataset)
            loss_d = bce_sum_loss(student_output_unobserved_dataset, teacher_output_unobserved_dataset) / n * student_config['gamma']
            loss = loss_a + loss_c + loss_e2 + loss_d
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
        # Validating model
        with torch.no_grad():
            student_config['architecture'].eval()
            preds = student_config['architecture'](x_val)
            student_config['architecture'].train()
            # compute current validation loss and keep best model based on it.
            validation_loss = bce_loss(preds, y_val)
            if validation_loss < minimum_validation_loss:
                checkpoint = {
                    'model_state_dict': student_config['architecture'].state_dict(),
                }
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(checkpoint, path + student_config['model_name'] + '.pth')
                minimum_validation_loss = validation_loss
                attempts = 0
            else:
                attempts += 1
    return n_epochs


def kdrec_bridge_training(student_config: dict, teacher_config: dict, n_epochs: int, teacher_train_data: np.ndarray,
                          student_train_data: np.ndarray, validation_data: np.ndarray, unobserved_data: np.ndarray,
                          path: str, conv_wait_steps: int, dataset_name: DatasetName):
    # Set train hyperparameters
    attempts = 0
    minimum_validation_loss = np.inf
    # Moving model to GPU and initialize optimizer
    student_config['architecture'].to(device)
    teacher_config['architecture'].to(device)
    student_optimizer = torch.optim.Adam(
        student_config['architecture'].parameters(),
        weight_decay=student_config['weight_decay']
        )
    teacher_optimizer = torch.optim.Adam(
        teacher_config['architecture'].parameters(),
        weight_decay=teacher_config['weight_decay']
        )
    student_config['architecture'].train()
    teacher_config['architecture'].train()
    # Constructing training and validating tensors
    y_val = torch.tensor(validation_data[:, -1].reshape(-1, 1), device=device, dtype=torch.float)
    if dataset_name == DatasetName.coat:    
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.float)
        student_dataset = TensorDataset(
            torch.tensor(student_train_data[:, :-1], dtype=torch.float),
            torch.tensor(student_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
        teacher_dataset = TensorDataset(
            torch.tensor(teacher_train_data[:, :-1], dtype=torch.float),
            torch.tensor(teacher_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    elif dataset_name == DatasetName.yahooR3:
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.long)
        student_dataset = TensorDataset(
            torch.tensor(student_train_data[:, :-1], dtype=torch.long),
            torch.tensor(student_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
        teacher_dataset = TensorDataset(
            torch.tensor(teacher_train_data[:, :-1], dtype=torch.long),
            torch.tensor(teacher_train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )

    student_dataloader = DataLoader(
        student_dataset,
        student_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    teacher_dataloader = DataLoader(
        teacher_dataset,
        teacher_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    if dataset_name == DatasetName.coat:
        unobserved_dataset = TensorDataset(
            torch.tensor(unobserved_data, dtype=torch.float)
        )
        unobserved_dataloader = DataLoader(
            unobserved_dataset,
            student_config['unobserved_batch_size'],
            shuffle=True,
            drop_last=True
        )
    # Train epochs
    for epoch in range(1, n_epochs + 1):
        # Checking whether training procedure converged
        if attempts > conv_wait_steps:
            return epoch
        student_iterator = iter(student_dataloader)
        teacher_iterator = iter(teacher_dataloader)
        if dataset_name == DatasetName.coat:
            n_batches = min(len(student_dataloader), len(teacher_dataloader), len(unobserved_dataloader))
            unobserved_iterator = iter(unobserved_dataloader)
        elif dataset_name == DatasetName.yahooR3:
            n_batches = min(len(student_dataloader), len(teacher_dataloader))
        for idx in range(n_batches):
            # Moving data to device
            student_x, student_y = next(student_iterator)
            teacher_x, teacher_y = next(teacher_iterator)
            student_x, student_y = student_x.to(device), student_y.to(device)
            teacher_x, teacher_y = teacher_x.to(device), teacher_y.to(device)
            if dataset_name == DatasetName.coat:
                x_unobserved = next(unobserved_iterator)
                x_unobserved = x_unobserved[0].to(device)
            elif dataset_name == DatasetName.yahooR3:
                x_unobserved = get_unobserved_batch(device, student_config['unobserved_batch_size'])
            # Calculating loss and updating model parameters
            student_output = student_config['architecture'](student_x)
            teacher_output = teacher_config['architecture'](teacher_x)
            student_unobserved_output = student_config['architecture'](x_unobserved)
            teacher_unobserved_output = teacher_config['architecture'](x_unobserved)
            loss = l2_loss(student_output, student_y) + l2_loss(teacher_output, teacher_y) + l2_loss(teacher_unobserved_output, student_unobserved_output)
            teacher_optimizer.zero_grad()
            student_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()
            student_optimizer.step()
        # Validating model
        with torch.no_grad():
            student_config['architecture'].eval()
            preds = student_config['architecture'](x_val)
            student_config['architecture'].train()
            # compute current validation loss and keep best model based on it.
            validation_loss = bce_loss(preds, y_val)
            if validation_loss < minimum_validation_loss:
                student_checkpoint = {
                    'model_state_dict': student_config['architecture'].state_dict(),
                }
                teacher_checkpoint = {
                    'model_state_dict': teacher_config['architecture'].state_dict(),
                }
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(student_checkpoint, path + student_config['model_name'] + '.pth')
                torch.save(teacher_checkpoint, path + teacher_config['model_name'] + '.pth')
                minimum_validation_loss = validation_loss
                attempts = 0
            else:
                attempts += 1
    return n_epochs


def kdrec_refine_training(student_config: dict, teacher: torch.nn.Module, n_epochs: int, train_data: np.ndarray,
                          validation_data: np.ndarray, path: str, conv_wait_steps: int, dataset_name: DatasetName):
    # Set train hyperparameters
    attempts = 0
    minimum_validation_loss = np.inf
    # Moving model to GPU and initialize optimizer
    student_config['architecture'].to(device)
    optimizer = torch.optim.Adam(
        student_config['architecture'].parameters(),
        weight_decay=student_config['weight_decay']
        )
    student_config['architecture'].train()
    teacher.eval()
    # Constructing training and validating tensors 
    y_val = torch.tensor(validation_data[:, -1].reshape(-1, 1), device=device, dtype=torch.float)
    if dataset_name == DatasetName.coat:    
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.float)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.float),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    elif dataset_name == DatasetName.yahooR3:
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.long)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.long),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    dataloader = DataLoader(
        dataset,
        student_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    # Train epochs
    for epoch in range(1, n_epochs + 1):
        # Checking wheter trianing procedure converged
        if attempts > conv_wait_steps:
            return epoch
        for idx, (x, y) in enumerate(dataloader):
            # Moving data to device
            x, y = x.to(device), y.to(device)
            # Calculating loss and update model parameters
            teacher_output = teacher(x)
            teacher_output = (teacher_output - teacher_output.min()) / (teacher_output.max() - teacher_output.min())
            labels = y + student_config['alpha'] * teacher_output
            student_output = student_config['architecture'](x)
            loss = l2_loss(student_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validating model
        with torch.no_grad():
            student_config['architecture'].eval()
            preds = student_config['architecture'](x_val)
            student_config['architecture'].train()
            # compute current validation loss and keep best model based on it.
            validation_loss = bce_loss(preds, y_val)
            if validation_loss < minimum_validation_loss:
                checkpoint = {
                    'model_state_dict': student_config['architecture'].state_dict(),
                }
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(checkpoint, path + student_config['model_name'] + '.pth')
                minimum_validation_loss = validation_loss
                attempts = 0
            else:
                attempts += 1
    return n_epochs 

def autodebias_training(model_config: dict, weight1_config:dict , weight2_config: dict, imputation_config:dict, 
                        n_epochs: int, train_data: np.ndarray, train_uniform_data: np.ndarray,
                        validation_data: np.ndarray, path: str, conv_wait_steps: int, dataset_name: DatasetName):
    # Set train hyperparameters
    attempts = 0
    minimum_validation_loss = np.inf
    # Moving model to GPU and initialize optimizer
    model_config['architecture'].to(device)
    weight1_config['architecture'].to(device)
    weight2_config['architecture'].to(device)
    imputation_config['architecture'].to(device)
    optimizer = torch.optim.Adam(
        model_config['architecture'].parameters(),
        weight_decay=model_config['weight_decay']
    )
    weight1_optimizer = torch.optim.Adam(
        weight1_config['architecture'].parameters(),
        weight_decay=weight1_config['weight_decay'],
    )
    weight2_optimizer = torch.optim.Adam(
        weight2_config['architecture'].parameters(),
        weight_decay=weight2_config['weight_decay']
    )
    imputation_optimizer = torch.optim.Adam(
        imputation_config['architecture'].parameters(),
        weight_decay=imputation_config['weight_decay']
    )
    # Constructing training and validating tensors
    y_val = torch.tensor(validation_data[:, -1].reshape(-1, 1), device=device, dtype=torch.float)
    if dataset_name == DatasetName.coat:    
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.float)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.float),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
        uniform_dataset = TensorDataset(
            torch.tensor(train_uniform_data[:, :-1], dtype=torch.float),
            torch.tensor(train_uniform_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    elif dataset_name == DatasetName.yahooR3:
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.long)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.long),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
        uniform_dataset = TensorDataset(
            torch.tensor(train_uniform_data[:, :-1], dtype=torch.long),
            torch.tensor(train_uniform_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    dataloader = DataLoader(
        dataset,
        model_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    uniform_dataloader = DataLoader(
        uniform_dataset,
        model_config['uniform_batch_size'],
        shuffle=True,
        drop_last=True
    )
    # Train epochs
    for epoch in range(1, n_epochs + 1):
        # Checking wheter trainig process converged
        if attempts > conv_wait_steps:
            return epoch
        n_batches = min(len(dataloader), len(uniform_dataloader))
        iterator = iter(dataloader)
        uniform_iterator = iter(uniform_dataloader)
        for idx in range(n_batches):
            x, y = next(iterator)
            uni_x, uni_y = next(uniform_iterator)
            # All pair
            prod_x, observed, imputation_input = cartesian_product(x, dataset_name)
            # Moving data to device
            x, y = x.to(device), y.to(device)
            uni_x, uni_y = uni_x.to(device), uni_y.to(device)
            prod_x, observed, imputation_input = prod_x.to(device), observed.to(device), imputation_input.to(device)
            # Calculate weight 1
            weight1_config['architecture'].train()
            weight1_output  = weight1_config['architecture'](x, y)
            # Calculate weight 2
            weight2_config['architecture'].train()
            weight2_output = weight2_config['architecture'](prod_x, observed)
            # Calcluate Imputation Values
            imputation_config['architecture'].train()
            imputation_output = imputation_config['architecture'](imputation_input)
            # one_step_model: assumed model, just update one step on base model. it is for updating weight parameters
            onestep_model = deepcopy(model_config['architecture'])
            onestep_opimizer = deepcopy(optimizer)
            # formal parameter: Using training set to update parameters
            onestep_model.train()
            # all pair data in this block
            onestep_output_all = onestep_model(prod_x)
            onestep_cost_all = l2_non_loss(onestep_output_all, imputation_output)
            onestep_loss_all = torch.sum(onestep_cost_all * weight2_output)
            # observation data
            onestep_output_observed = onestep_model(x)
            onestep_cost_observed = l2_non_loss(onestep_output_observed, y)
            onestep_loss_observed = torch.sum(onestep_cost_observed * weight1_output)
            onestep_loss = onestep_loss_observed + model_config['imputation_lambda'] * onestep_loss_all 
            # update parameters of one_step_model
            onestep_opimizer.zero_grad()
            onestep_loss.backward()
            onestep_opimizer.step()
            # latter hyper_parameter: Using uniform set to update hyper_parameters
            onestep_output_uniform = onestep_model(uni_x)
            onestep_loss_uniform = l2_sum_loss(onestep_output_uniform, uni_y)
            # update hyper-parameters
            weight1_optimizer.zero_grad()
            weight2_optimizer.zero_grad()
            imputation_optimizer.zero_grad()
            onestep_loss_uniform.backward()
            weight1_optimizer.step()
            weight2_optimizer.step()
            imputation_optimizer.step()
            # use new weights to update parameters (real update) 
            weight1_config['architecture'].train()
            weight1_output2  = weight1_config['architecture'](x, y)
            # Calculate weight 2
            weight2_config['architecture'].train()
            weight2_output2 = weight2_config['architecture'](prod_x, observed)
            # use new imputation to update parameters
            imputation_config['architecture'].train()
            imputation_output2 = imputation_config['architecture'](imputation_input)
            # loss of training set
            model_config['architecture'].train()
            model_output_all = model_config['architecture'](prod_x)
            model_cost_all = l2_non_loss(model_output_all, imputation_output2)
            model_loss_all = torch.sum(model_cost_all * weight2_output2)
            # observation
            model_output_observed = model_config['architecture'](x)
            model_cost_observed = l2_non_loss(model_output_observed, y)
            model_loss_observed = torch.sum(model_cost_observed * weight1_output2)
            model_loss = model_loss_observed + model_config['imputation_lambda'] * model_loss_all 
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
        # Validating model
        with torch.no_grad():
            model_config['architecture'].eval()
            preds = model_config['architecture'](x_val)
            model_config['architecture'].train()
            # compute current validation loss and keep best model based on it.
            validation_loss = bce_loss(preds, y_val)
            if validation_loss < minimum_validation_loss:
                checkpoint = {
                    'model_state_dict': model_config['architecture'].state_dict(),
                }
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(checkpoint, path + model_config['model_name'] + '.pth')
                minimum_validation_loss = validation_loss
                attempts = 0
            else:
                attempts += 1
    return n_epochs





def epsilon_nongreedy_training(student_config: dict, teacher: torch.nn.Module, n_epochs: int, train_data: np.ndarray,
                               validation_data: np.ndarray, unobserved_data: np.ndarray, path: str, conv_wait_steps: int,
                               dataset_name: DatasetName):
    # Set train hyperparameters
    attempts = 0
    minimum_validation_loss = np.inf
    # Moving model to GPU and initialize optimizer
    student_config['architecture'].to(device)
    if student_config['weight_decay'] is not None:
        optimizer = torch.optim.Adam(
            student_config['architecture'].parameters(),
            weight_decay=student_config['weight_decay']
            )
    else:
        optimizer = torch.optim.Adam(student_config['architecture'].parameters())
    student_config['architecture'].train()
    teacher.eval()
    # Constructing training and validating tensors 
    y_val = torch.tensor(validation_data[:, -1].reshape(-1, 1), device=device, dtype=torch.float)
    if dataset_name == DatasetName.coat:    
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.float)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.float),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    elif dataset_name == DatasetName.yahooR3:
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.long)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.long),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    dataloader = DataLoader(
        dataset,
        student_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    if dataset_name == DatasetName.coat:
        unobserved_dataset = TensorDataset(
            torch.tensor(unobserved_data, dtype=torch.float)
        )
        unobserved_dataloader = DataLoader(
            unobserved_dataset,
            student_config['unobserved_batch_size'],
            shuffle=True,
            drop_last=True
        )
    # Train epochs
    for epoch in range(1, n_epochs + 1):
        # Checking wheter trianing procedure converged
        if attempts > conv_wait_steps:
            return epoch
        
        if dataset_name == DatasetName.coat:
            n_batches = min(len(dataloader), len(unobserved_dataloader))
            unobserved_iterator = iter(unobserved_dataloader)
        elif dataset_name == DatasetName.yahooR3:
            n_batches = len(dataloader)
        iterator = iter(dataloader)

        for idx in range(n_batches):
            # Moving data to device
            x, y = next(iterator)
            x, y = x.to(device), y.to(device)
            if dataset_name == DatasetName.coat:
                x_unobserved = next(unobserved_iterator)
                x_unobserved = x_unobserved[0].to(device)
            elif dataset_name == DatasetName.yahooR3:
                x_unobserved = get_unobserved_batch(device, student_config['unobserved_batch_size'])
            # Calculating loss and updating model parameters
            student_output = student_config['architecture'](x)
            student_unobserved_output = student_config['architecture'](x_unobserved)
            teacher_output = teacher(x_unobserved)
            main_term = bce_loss(student_output, y)
            if 'l1' in student_config['model_name']:
                regularization_term = student_config['lambda'] * l1_loss(student_unobserved_output, teacher_output)
            elif 'l2' in student_config['model_name']:
                regularization_term = student_config['lambda'] * l2_loss(student_unobserved_output, teacher_output)
            elif 'kl' in student_config['model_name']:
                regularization_term = student_config['lambda'] * kl_loss_function(student_unobserved_output, teacher_output)
            else:
                regularization_term = student_config['lambda'] * js_loss_function(student_unobserved_output, teacher_output)
            loss = main_term + regularization_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validating model
        with torch.no_grad():
            if 'dropout' in student_config['model_name']:
                preds = torch.zeros((x_val.shape[0], 1), device=device, dtype=torch.float)
                for i in range(student_config['n_samples']):
                    preds += student_config['architecture'](x_val)
                preds /= student_config['n_samples']
            else:
                student_config['architecture'].eval()
                preds = student_config['architecture'](x_val)
                student_config['architecture'].train()
            # compute current validation loss and keep best model based on it.
            validation_loss = bce_loss(preds, y_val)
            if validation_loss < minimum_validation_loss:
                checkpoint = {
                    'model_state_dict': student_config['architecture'].state_dict(),
                }
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(checkpoint, path + student_config['model_name'] + '.pth')
                minimum_validation_loss = validation_loss
                attempts = 0
            else:
                attempts += 1
    return n_epochs


def baseline_training(model_config: dict, n_epochs: int, train_data: np.ndarray,
                      validation_data: np.ndarray, path: str, conv_wait_steps: int,
                      dataset_name: DatasetName):
    # Set train hyperparameters
    attempts = 0
    minimum_validation_loss = np.inf
    # Moving model to GPU and initilizing optimizer
    model_config['architecture'].to(device)
    if model_config['weight_decay'] is not None:
        optimizer = torch.optim.Adam(
            model_config['architecture'].parameters(),
            weight_decay=model_config['weight_decay']
            )
    else:
        optimizer = torch.optim.Adam(model_config['architecture'].parameters())
    model_config['architecture'].train()
    # Constructing training and validating tensors
    y_val = torch.tensor(validation_data[:, -1].reshape(-1, 1), device=device, dtype=torch.float)
    if dataset_name == DatasetName.coat:    
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.float)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.float),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    elif dataset_name == DatasetName.yahooR3:
        x_val = torch.tensor(validation_data[:, :-1], device=device, dtype=torch.long)
        dataset = TensorDataset(
        torch.tensor(train_data[:, :-1], dtype=torch.long),
        torch.tensor(train_data[:, -1].reshape(-1, 1), dtype=torch.float)
        )
    dataloader = DataLoader(
        dataset,
        model_config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    # Train epochs
    for epoch in range(1, n_epochs+1):
        # Checking wheter trainig process converged
        if attempts > conv_wait_steps:
            return epoch
        for idx, (x, y) in enumerate(dataloader):
            # Moving data to device
            x, y = x.to(device), y.to(device)
            # Calculating loss and update model parameters
            output = model_config['architecture'](x)
            loss = bce_loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validating model
        with torch.no_grad():
            if 'dropout' in model_config['model_name']:
                preds = torch.zeros((x_val.shape[0], 1), device=device, dtype=torch.float)
                for i in range(model_config['n_samples']):
                    preds += model_config['architecture'](x_val)
                preds /= model_config['n_samples']
            else:
                model_config['architecture'].eval()
                preds = model_config['architecture'](x_val)
                model_config['architecture'].train()
            # compute current validation loss and keep best model based on it.
            validation_loss = bce_loss(preds, y_val)
            if validation_loss < minimum_validation_loss:
                checkpoint = {
                    'model_state_dict': model_config['architecture'].state_dict(),
                }
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(checkpoint, path + model_config['model_name'] + '.pth')
                minimum_validation_loss = validation_loss
                attempts = 0
            else:
                attempts += 1
    return n_epochs
            

def train(models: dict, datasets: dict, unobserved_data: np.ndarray, validation_data: np.ndarray,
          logger: logging.Logger, n_epochs: int, saving_path: str, conv_wait_steps: int,
          dataset_name: DatasetName):
    
    for model_name, teacher_name, model_type in TRAIN_ORDER:
        if model_name not in models:
            continue
        # Classic Models
        if model_type == 'classic':
            if 'dropout' in model_name:
                model_config = {
                    'architecture': models[model_name]['architecture'],
                    'model_name': model_name,
                    'batch_size': models[model_name]['config']['batch_size'],
                    'weight_decay': models[model_name]['config']['weight_decay'],
                    'n_samples': models[model_name]['config']['n_samples'],
                }
            else:
                model_config = {
                    'architecture': models[model_name]['architecture'],
                    'model_name': model_name,
                    'batch_size': models[model_name]['config']['batch_size'],
                    'weight_decay': models[model_name]['config']['weight_decay'],
                }
            convergence_epoch = baseline_training(
                model_config=model_config,
                n_epochs= n_epochs,
                train_data=datasets[model_name],
                validation_data=validation_data,
                path=saving_path,
                conv_wait_steps=conv_wait_steps,
                dataset_name=dataset_name,
            )
            logger.error(f"         {model_name} converged after {convergence_epoch} epochs.")
        # Proposed Models
        elif model_type == 'proposed':
            if 'dropout' in model_name:
                model_config = {
                    'architecture': models[model_name]['architecture'],
                    'model_name': model_name,
                    'batch_size': models[model_name]['config']['batch_size'],
                    'weight_decay': models[model_name]['config']['weight_decay'],
                    'n_samples': models[model_name]['config']['n_samples'],
                    'unobserved_batch_size': models[model_name]['config']['unobserved_batch_size'],
                    'lambda': models[model_name]['config']['lambda'],
                }
            else:
                model_config = {
                    'architecture': models[model_name]['architecture'],
                    'model_name': model_name,
                    'batch_size': models[model_name]['config']['batch_size'],
                    'weight_decay': models[model_name]['config']['weight_decay'],
                    'unobserved_batch_size': models[model_name]['config']['unobserved_batch_size'],
                    'lambda': models[model_name]['config']['lambda'],
                }
            convergence_epoch = epsilon_nongreedy_training(
                student_config=model_config,
                teacher=models[teacher_name]['architecture'],
                n_epochs=n_epochs,
                train_data=datasets[model_name],
                validation_data=validation_data,
                unobserved_data=unobserved_data,
                path=saving_path,
                conv_wait_steps=conv_wait_steps,
                dataset_name=dataset_name,
            )
            logger.error(f"         {model_name} converged after {convergence_epoch} epochs.")
        elif model_type == 'kdrec_refine':
            model_config = {
                'architecture': models[model_name]['architecture'],
                'model_name': model_name,
                'batch_size': models[model_name]['config']['batch_size'],
                'weight_decay': models[model_name]['config']['weight_decay'],
                'alpha': models[model_name]['config']['alpha'],
            }
            convergence_epoch = kdrec_refine_training(
                student_config=model_config,
                teacher= models[teacher_name]['architecture'],
                n_epochs= n_epochs,
                train_data= datasets[model_name],
                validation_data= validation_data,
                path= saving_path,
                conv_wait_steps=conv_wait_steps,
                dataset_name=dataset_name,
            )
            logger.error(f"         {model_name} converged after {convergence_epoch} epochs.")
        elif model_type == 'kdrec_bridge':
            student_config = {
                'architecture': models[model_name]['architecture'],
                'model_name': model_name,
                'batch_size': models[model_name]['config']['batch_size'],
                'weight_decay': models[model_name]['config']['weight_decay'],
                'unobserved_batch_size': models[model_name]['config']['unobserved_batch_size'],
            }
            teacher_config = {
                'architecture': models[teacher_name]['architecture'],
                'model_name': teacher_name,
                'batch_size': models[teacher_name]['config']['batch_size'],
                'weight_decay': models[teacher_name]['config']['weight_decay'],
            }
            convergence_epoch = kdrec_bridge_training(
                student_config= student_config,
                teacher_config= teacher_config,
                n_epochs= n_epochs,
                teacher_train_data=datasets[teacher_name],
                student_train_data=datasets[model_name],
                validation_data= validation_data,
                unobserved_data= unobserved_data,
                path= saving_path,
                conv_wait_steps= conv_wait_steps,
                dataset_name=dataset_name,
            )
            logger.error(f"         {model_name, teacher_name} converged after {convergence_epoch} epochs.")
        elif model_type == 'autodebias':
            model_config = {
                'architecture': models[model_name]['architecture'],
                'model_name': model_name,
                'batch_size': models[model_name]['config']['batch_size'],
                'uniform_batch_size': models[model_name]['config']['uniform_batch_size'],
                'weight_decay': models[model_name]['config']['weight_decay'],
                'imputation_lambda': models[model_name]['config']['imputation_lambda'],
            }
            weight1_config = {
                'architecture': models[model_name]['weight1_architecture'],
                'weight_decay': models[model_name]['config']['weights_weight_decay'],
            }
            weight2_config = {
                'architecture': models[model_name]['weight2_architecture'],
                'weight_decay': models[model_name]['config']['weights_weight_decay'],
            }
            imputation_config = {
                'architecture': models[model_name]['imputation_architecture'],
                'weight_decay': models[model_name]['config']['imputation_weight_decay']
            }
            convergence_epoch = autodebias_training(
                model_config=model_config,
                weight1_config=weight1_config,
                weight2_config=weight2_config,
                imputation_config=imputation_config,
                n_epochs=n_epochs,
                train_data=datasets[model_name],
                train_uniform_data=datasets['autodebias-uniform'],
                validation_data= validation_data,
                path=saving_path,
                conv_wait_steps=conv_wait_steps,
                dataset_name=dataset_name,
            )
            logger.error(f"         {model_name} converged after {convergence_epoch} epochs.")
        elif model_type  == 'dub':
            student_config = {
                'architecture': models[model_name]['architecture'],
                'model_name': model_name,
                'batch_size': models[model_name]['config']['batch_size'],
                'weight_decay': models[model_name]['config']['weight_decay'],
                'unobserved_batch_size': models[model_name]['config']['unobserved_batch_size'],
                'gamma': models[model_name]['config']['gamma'],
            }
            teacher_config = {
                'architecture': models[teacher_name]['architecture'], 
                'batch_size': models[teacher_name]['config']['batch_size'], 
            }
            convergence_epoch = dub_training(
                student_config=student_config,
                teacher_config=teacher_config,
                n_epochs=n_epochs,
                teacher_train_data=datasets[teacher_name],
                student_train_data=datasets[model_name],
                validation_data=validation_data,
                unobserved_data=unobserved_data,
                path=saving_path,
                conv_wait_steps=conv_wait_steps,
                dataset_name=dataset_name,
            )
            logger.error(f"         {model_name} converged after {convergence_epoch} epochs.")