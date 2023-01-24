import numpy as np
import torch
from matplotlib import pyplot
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CIFAR10Pair
from model import Model


# https://arxiv.org/pdf/2002.05709.pdf
def loss(z1_norm, z2_norm, temperature):
    sim1 = (z1_norm @ z2_norm.t())/temperature
    l1 = torch.log(torch.exp(torch.diag(sim1)) / torch.sum(torch.exp(sim1), dim=1))
    sim2 = (z2_norm @ z1_norm.t())/temperature
    l2 = torch.log(torch.exp(torch.diag(sim2)) / torch.sum(torch.exp(sim2), dim=1))
    l = -torch.mean(torch.cat([l1, l2]))
    return l

def train(epochs, model, batch_size=512, temperature=0.5):
    dataset = CIFAR10Pair(train=True, aug_mode=True)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=20, drop_last=True)
    optim = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    model.cuda()

    loss_log = []
    accuracy_log = []
    tqdm_bar = tqdm(range(epochs))
    for i in tqdm_bar:
        epoch = i+1
        model.train()

        l_epoch = 0
        for x1, x2, l in dataloader:
            x1 = x1.cuda()
            x2 = x2.cuda()
            f1_norm, z1_norm = model(x1)
            f2_norm, z2_norm = model(x2)
            optim.zero_grad()
            l = loss(z1_norm, z2_norm, temperature)
            l.backward()
            optim.step()
            l_epoch += (l.item() / x1.shape[0])

        loss_log.append(l_epoch)
        accuracy = test(model)
        accuracy_log.append(accuracy)
        tqdm_bar.set_description(f'loss: {l_epoch: .4f}, accuracy: {accuracy: .4f}')

    return loss_log, accuracy_log

# knn
def test(model, batch_size=512, k = 200, temperature=0.5):
    model.eval()
    train_dataset = CIFAR10Pair(train=True, aug_mode=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)
    encodings = []
    labels = []
    with torch.no_grad():
        for x, l in train_dataloader:
            x = x.cuda()
            f, z = model(x)
            encodings.append(z)
            labels.append(l)

    encodings = torch.cat(encodings)
    labels = torch.cat(labels)

    test_dataset = CIFAR10Pair(train=False, aug_mode=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)

    correct = 0
    incorrect = 0
    with torch.no_grad():
        for x, l in test_dataloader:
            x = x.cuda()
            l = l.cuda()
            f, z = model(x)

            dist = torch.exp(z @ encodings.t() / temperature)
            top_k_dist_weight, top_k_dist_index = torch.topk(dist, k, dim=-1)
            top_k_labels = labels.cuda()[top_k_dist_index]
            top_k_labels = one_hot(top_k_labels) * top_k_dist_weight[:, :, None]
            pred = torch.argmax(torch.squeeze(torch.sum(top_k_labels, dim=1)), dim=1)

            correct += torch.sum(pred == l).item()
            incorrect += torch.sum(pred != l).item()

    accuracy = (correct) / (correct + incorrect)
    return accuracy


if __name__ == '__main__':
    model = Model()
    loss_log, accuracy_log = train(500, model)
    # print(f'loss: {loss_log}, accuracy: {accuracy_log}')