import torch
import torch.nn as nn
import torch.nn.functional as F


def my_loss(scale_activations, same_indexes,  parameter,
            temperature):
    """
    :param scale_activations: list of lists of pipeline conv layer kernel activations
    :param same_indexes: dict of indices of in-class samples in mini-batch
    :param parameter: loss parameter
    :param temperature: temperature hyperparameter for the custom loss

    :return: total loss
    """

    nmodules = len(scale_activations)
    loss = 0
    # loss2 = []
    for i in range(nmodules):
        l_loss = layer_loss(scale_activations[i], same_indexes, temperature)
        l_loss = l_loss * parameter
        # loss2.append(l_loss) #debug
        loss -= l_loss

    return loss


def layer_loss(layer_activations: torch.Tensor,
               same_indexes: torch.Tensor,
               temperature: float,
               ):

    # batch_energy = layer_activations
    A = layer_activations

    A_n = torch.nn.functional.normalize(A)

    all_energy = torch.exp(torch.matmul(A_n, A_n.T))
    # all_energy = all_energy.fill_diagonal_(0)
    denominator = torch.sum(all_energy)

    loss = 0

    for i in range(len(same_indexes)):
        pos_energy = 0
        if len(same_indexes[i]) != 0:
            for pair in same_indexes[i]:
                pos_energy += (torch.multiply(all_energy[(pair[0], pair[1])], 2) / temperature)
            class_loss = torch.log(torch.div(pos_energy, denominator))
            loss += class_loss

    return loss


def find_negative_pairs(same_indexes: dict):
    pairs = []
    key_list = list(same_indexes.keys())                      # range(0, num_classes)
    for key in key_list:                                      # key = 0
        for i in range(len(same_indexes[key][0])):            # i_class = 0, key = 3
            i_val = same_indexes[key][0][i]                   # i_val = same_indexes{3: --> 0)
            for j in range(len(key_list)):
                # if key_list[j] == i_class or key_list[j] == key:
                if key_list[j] == key:
                    continue
                else:
                    for j_val in same_indexes[key_list[j]][0]:
                        if i_val != j_val and (j_val, i_val) not in pairs:
                            pairs.append((i_val, j_val))

    return pairs


def find_distinct_pairs(same_indexes: dict):
    pairs = []
    key_list = list(same_indexes.keys())                      # range(0, num_classes)
    for key in key_list:                                      # key = 0
        for i in range(len(same_indexes[key][0])):            # i_class = 0, key = 3
            i_val = same_indexes[key][0][i]                   # i_val = same_indexes{3: --> 0)
            for j in range(len(key_list)):
                # if key_list[j] == i_class or key_list[j] == key:
                # if key_list[j] == key:
                #     continue
                # else:
                for j_val in same_indexes[key_list[j]][0]:
                    if i_val != j_val and (j_val, i_val) not in pairs:
                        pairs.append((i_val, j_val))

    return pairs