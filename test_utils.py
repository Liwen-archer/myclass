import torch
import numpy as np
import math
import logging
from matplotlib import pyplot as plt

from utils import random_instance_masking, multitask_train, test

def train_model(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, prop):
    tar_loss_masked_arr, tar_loss_unmasked_arr, tar_loss_arr, task_loss_arr, min_task_loss = [
    ], [], [], [], math.inf

    instance_weights = torch.as_tensor(torch.rand(
        X_train_task.shape[0], prop['seq_len']), device=prop['device'])
    for epoch in range(1, prop['epochs'] + 1):

        X_train_tar, y_train_tar_masked, y_train_tar_unmasked, boolean_indices_masked, boolean_indices_unmasked = \
            random_instance_masking(
                X_train_task, prop['masking_ratio'], prop['ratio_highest_attention'], instance_weights)

        tar_loss_masked, tar_loss_unmasked, task_loss, instance_weights = multitask_train(model, criterion_tar, criterion_task, optimizer, X_train_tar, X_train_task, y_train_tar_masked, y_train_tar_unmasked, y_train_task, boolean_indices_masked, boolean_indices_unmasked, prop)

        tar_loss_masked_arr.append(tar_loss_masked)
        tar_loss_unmasked_arr.append(tar_loss_unmasked)
        tar_loss = tar_loss_masked + tar_loss_unmasked
        tar_loss_arr.append(tar_loss)
        task_loss_arr.append(task_loss)
        print('Epoch: ' + str(epoch) + ', TAR Loss: ' + str(tar_loss), ', TASK Loss: ' + str(task_loss))

        # save model and optimizer for lowest training loss on the end task
        if task_loss < min_task_loss:
            min_task_loss = task_loss
            best_model.load_state_dict(model.state_dict())
            best_optimizer.load_state_dict(optimizer.state_dict())

    plot_loss(task_loss_arr)

    # Saved best model state at the lowest training loss is evaluated on the official test set
    metrics = test(best_model, X_train_task, y_train_task, prop['batch'], prop['nclasses'], criterion_task, prop['task_type'], prop['device'], prop['avg'])
    print('TRAIN ===> Dataset: ' + prop['dataset'] + ', Acc: ' + str(metrics[1]) )

    del model
    torch.cuda.empty_cache()
    

def test_model(model, criterion_task, X_test, y_test, prop):
    test_metrics = test(model, X_test, y_test, prop['batch'], prop['nclasses'], criterion_task, prop['task_type'], prop['device'], prop['avg'])

    loss, acc, prec, rec, f1 = test_metrics
    print('Test: ' + prop['dataset'], ', Loss: ' + str(loss), ', Acc: ' + str(acc), ', Prec: ' + str(prec), ', Rec: ' + str(rec), ', F1: ' + str(f1))
    logging.info(f"Test {prop['dataset']}, {acc}, {prec}, {rec}, {f1}")


    del model
    torch.cuda.empty_cache()
    

def plot_loss(losses):
    length = len(losses)
    x = np.arange(1, length+1)
    y = np.array(losses)
    plt.plot(x, y)
    plt.savefig('loss')