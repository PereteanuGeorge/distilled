import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import json
from attack_model import Attack
from attack_model2 import Attack2
from client import model1
from utils import device, trainset
from utils import testset
from server2 import model3
import numpy as np

criterion = nn.CrossEntropyLoss()
my_criterin = nn.MSELoss()

def reproducibilitySeed(seed=73):
    """
    Ensure reproducibility of results; Seeds to 0
    """
    torch_init_seed = seed
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = seed
    np.random.seed(numpy_init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


reproducibilitySeed(0)


def train_attack2(epoch, optimizer2, attack2, validloader):
    print('\nEpoch: %d' % epoch)
    attack2.train()
    total_loss = 0
    for batch_idx, (inputs, _) in enumerate(validloader):
        inputs = inputs.to(device)
        preds1 = attack2(inputs)
        targets = model3(inputs)
        loss = my_criterin(preds1, targets)
        total_loss += loss.item()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    return total_loss


def train_attack(epoch, optimizer, attack, attack2, validloader):
    print('\nEpoch: %d' % epoch)
    attack.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            preds1 = attack2(inputs)
            #preds1 = model3(inputs)
            preds = model1(preds1)
        #print(f'preds shape {preds.shape}')
        outputs = attack(preds)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss


def test(attack, attack2, testloader):
    attack.eval()
    attack2.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds1 = attack2(inputs)
            #print(f'preds1 shape {preds1.shape}')
            with torch.no_grad():
                preds = model1(preds1)
            #print(f'preds shape {preds.shape}')
            outputs = attack(preds)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # print(f' TEEST accuracy e ceva {100.*correct/total} si loss e {test_loss/(batch_idx+1)}')
    return 100. * correct / total, test_loss


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

f = open("attacker-knows-archi.txt", "a+")

lr = [0.05]
batch_size = [128]
num_samples = [500, 2500, 5000]
big_json = {}
for samples in num_samples:
    for learning_rate in lr:
        for batch in batch_size:
            best_acc = 0
            best_batch = 0
            best_lr = 0
            attack2_list_loss = []
            attack_list_loss = []
            test_list_loss = []
            attack2 = Attack2().to(device)
            attack = Attack(512).to(device)
            attack.apply(weight_reset)
            attack2.apply(weight_reset)
            optimizer = optim.SGD(attack.parameters(), lr=learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            optimizer2 = optim.SGD(attack2.parameters(), lr=learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            _, valid = random_split(trainset, [50000 - samples, samples])
            validloader = DataLoader(valid, batch_size=batch, shuffle=True)

            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch, shuffle=False, num_workers=2)

            for epoch in range(100):
                attack2_loss = train_attack2(epoch, optimizer2, attack2, validloader)
                attack_loss = train_attack(epoch, optimizer, attack, attack2, validloader)
                acc, test_loss = test(attack, attack2, testloader)
                attack2_list_loss.append(attack2_loss)
                attack_list_loss.append(attack_loss)
                test_list_loss.append(test_loss)
                if acc > best_acc:
                    best_batch = batch
                    best_lr = learning_rate
                    best_acc = acc
                scheduler.step()

            with open("attacker-knows-archi.txt", "a+") as f:
                text = "with # samples " + str(samples) + " we got best acc of " + str(best_acc) + ", batch of " + str(best_batch) + ", lr = " + str(best_lr)
                f.write(text)
                f.write('\n')
            dictionary = {1: attack2_list_loss, 2: attack_list_loss, 3: test_list_loss}
            big_json[f"{learning_rate}-{batch}-{samples}"] = dictionary
with open("losses.json", "a+") as f:
   json.dump(big_json,f)
