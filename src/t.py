import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    inputs = torch.randn(24,256)
    inputs = torch.sigmoid(inputs)
    targets = torch.zeros(24,256)
    BCE = torch.zeros(24)
    for i in range(inputs.shape[0]):
        BCE[i] = F.binary_cross_entropy(inputs[i,:], targets[i,:], reduction='mean')
    print(torch.mean(BCE))
    loss = nn.BCELoss(reduction='none')
    BCE = loss(inputs, targets)
    # print(torch.mean(BCE,1))
    print(torch.mean(torch.mean(BCE)))



if __name__ == "__main__":
    main()