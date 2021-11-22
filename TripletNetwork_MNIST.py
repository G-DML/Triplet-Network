from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.nn.modules.loss import TripletMarginLoss
from itertools import permutations
from datetime import datetime
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def Metrics(y_real, y_pred):
    acc = metrics.accuracy_score(y_real, y_pred)
    prec = metrics.precision_score(y_real, y_pred, average='macro')
    rec = metrics.recall_score(y_real, y_pred, average='macro')
    f = metrics.f1_score(y_real, y_pred, average='macro')

    print("The average scores for all classes:")
    # Calculate metrics for each label, and find their unweighted mean. does not take label imbalance into account.
    print("\nAccuracy:  {:.3f}%".format(acc * 100))  # (TP+TN)/Total / number of classes
    print("Precision: {:.3f}%".format(prec * 100))  # TP/(TP+FP) / number of classes
    print("Recall:    {:.3f}%".format(rec * 100))  # TP/(TP+FN) / number of classes
    print("F-measure: {:.3f}%".format(f * 100))  # 2 * (prec*rec)/(prec+rec) / number of classes

    print("\nThe scores for each class:")
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_real, y_pred)

    print("\n|    Label    |  Precision |  Recall  | F1-Score | Support")
    print("|-------------|------------|----------|----------|---------")
    for i in range(num_classes):
        print(
            f"| {classes[i]:<11} |  {precision[i] * 100:<7.2f}%  | {recall[i] * 100:<7.2f}% |   {fscore[i]:<4.2f}   | {support[i]}")

    return rec

color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
marker = ['.','+','x','1','^','s','p','*','d','X']
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def Ploting2D(embeddings_plot,labels_plot,tit="default",x_axis="X",y_axis="Y"):
    ax = plt.figure().add_subplot(111)
    for i in range(num_classes):
        index = labels_plot == i
        plt.scatter(embeddings_plot[0, index], embeddings_plot[1, index], s=3, marker='.', c=color[i], label=classes[i])
    ax.legend(loc='best', title="Labels", markerscale=5.0)

    # add grid
    plt.grid(True,linestyle='--')

    # add title
    plt.title(tit)
    plt.tight_layout()

    # add x,y axes labels
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

def Ploting3D(embeddings_plot, labels_plot, tit="default",x_axis="X",y_axis="Y",z_axis="Z"):
    ax = plt.figure().gca(projection='3d')
    for i in range(num_classes):
        index = labels_plot == i
        ax.scatter(embeddings_plot[0, index], embeddings_plot[1, index], embeddings_plot[2, index], s=3, marker='.',c=color[i], label=classes[i])
    ax.legend(loc='best', title="Labels", markerscale=5.0)

    # add title
    plt.title(tit)
    plt.tight_layout()

    # add x,y axes labels
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)

#####################################################################################################################
print("\nLOAD DATA\n")

mean, std = 0.1307, 0.3081 #MNIST

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)) ])

class TripletDataset(Dataset):
    def __init__(self, dataset, batch_size=2, transform=None):
        self.Anchor = torch.tensor([])
        self.Positive = torch.tensor([])
        self.Negative = torch.tensor([])
        #self.Labels = []
        self.batch_size = batch_size
        self.transform = transform

        samples, lab_set, min_size = self.split_by_label(dataset)

        self.batch_size = min(self.batch_size, min_size)

        lab_set_perm = list(permutations(lab_set, 2))
        np.random.shuffle(lab_set_perm)#################### erasable

        for i, j in lab_set_perm:
            a, p, n = self.Triplets_maker(samples[i], samples[j])
            self.Anchor = torch.cat((self.Anchor, a), 0)
            self.Positive = torch.cat((self.Positive, p), 0)
            self.Negative = torch.cat((self.Negative, n), 0)
            #self.Labels += [[i, j] for _ in range(self.batch_size)]

        print(f"Number of labels permutations: {len(lab_set_perm)}")
        print(f"Triplet samples per permutations: {self.batch_size}")
        print(f"Total number of triplet samples: {len(self.Anchor)}")

    def __len__(self):
        return len(self.Anchor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_sample = self.Anchor[idx]
        positive_sample = self.Positive[idx]
        negative_sample = self.Negative[idx]
        #landmarks = torch.tensor(self.Labels)[idx]

        if self.transform:
            anchor_sample = self.transform(anchor_sample)
            positive_sample = self.transform(positive_sample)
            negative_sample = self.transform(negative_sample)
            #landmarks = self.transform(landmarks)

        return (anchor_sample, positive_sample, negative_sample)#, landmarks

    def split_by_label(self, dataset):
        labels_set = list(dataset.class_to_idx.values())

        samples_by_label = {}
        label_size = []
        for label in labels_set:
            samples_by_label[label] = dataset.data[dataset.targets.numpy() == label]

            l, w, d = samples_by_label[label].shape
            label_size.append(l)

            samples_by_label[label] = samples_by_label[label].view(l, 1, d, w)

        return samples_by_label, labels_set, np.min(label_size)//2

    def Triplets_maker(self, class_1, class_2):
        index_ap = np.random.choice(range(len(class_1)), self.batch_size * 2, replace=False)
        index_n = np.random.choice(range(len(class_2)), self.batch_size, replace=False)

        anchor = class_1[index_ap[:self.batch_size]]
        positive = class_1[index_ap[self.batch_size:]]
        negative = class_2[index_n]

        return anchor, positive, negative

#Load Data
train_dataset = MNIST(root='dataset/', train=True, transform=preprocess, download='True')
test_dataset = MNIST(root='dataset/', train=False, transform=preprocess, download='True')

#Data to triplet format
batch_size = 512
triplet_train_ds = TripletDataset(dataset=train_dataset, batch_size=batch_size)

# Create validation & training datasets
val_size = int(len(triplet_train_ds) * 0.20)
train_size = len(triplet_train_ds) - val_size
triplet_train_ds, triplet_val_ds = random_split(triplet_train_ds, [train_size, val_size])

print("Train dataset size: ", len(triplet_train_ds))
print("Validation dataset size: ", len(triplet_val_ds), "\n")

#Dataset to Batches
triplet_train_ld = DataLoader(triplet_train_ds, batch_size=batch_size, shuffle=True)
triplet_val_ld = DataLoader(triplet_val_ds, batch_size=32, shuffle=False)

##################################################### Using a GPU #####################################################
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

print("USING", device)
if cuda:
    num_dev = torch.cuda.current_device()
    print(torch.cuda.get_device_name(num_dev),"\n")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for data in self.data:
            yield to_device(data, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.data)

#Batches to GPU
train_ld = DeviceDataLoader(triplet_train_ld, device)
val_ld = DeviceDataLoader(triplet_val_ld, device)

#####################################################################################################################
print("\nTRIPLET NETWORK TRAINING\n")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Triplet_Network(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, output_size=2, m=1.0, pow=2.0):
        super(Triplet_Network, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, output_size)

        self.triplet_loss = nn.TripletMarginLoss(margin=m, p=pow, eps=0)  # max{d(a,p)−d(a,n)+margin,0}
        self.triplet_loss_without_reduction = nn.TripletMarginLoss(margin=m, p=pow, eps=0, reduction='none')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, A, P, N):
        # Generate predictions
        anchor = self(A)
        positive = self(P)
        negative = self(N)

        # Calculate loss
        loss = self.triplet_loss(anchor, positive, negative)

        return loss

    def extract_embedding(self, dataset):
        size_batch = len(dataset)//20 #change this if not enough memory

        data_loader = DataLoader(dataset=dataset, batch_size=size_batch, shuffle=False)
        data_loader = DeviceDataLoader(data_loader, 'cuda')

        with torch.no_grad():
            self.train()
            embedding = torch.tensor([]).to('cuda')
            for batch in data_loader:
                data, _ = batch
                embedded = self.forward(data)
                embedding = torch.cat((embedding, embedded), 0)

        embedding = embedding.cpu().numpy() #embedding.t().cpu().numpy()
        label = dataset.targets.cpu().numpy()

        del data_loader, embedded, data
        torch.cuda.empty_cache()  # PyTorch thing

        return embedding, label
    
    def evaluate_step(self, val_loader):
        with torch.no_grad():
            self.eval()

            val_loss = []
            val_acc = []
            for batch in val_loader:
                (A, P, N) = batch

                # Generate predictions
                anchor = self(A)
                positive = self(P)
                negative = self(N)

                losses = self.triplet_loss_without_reduction(anchor, positive, negative)

                # Calculate loss
                loss = torch.mean(losses)

                # Calculate accuracy
                acc = (losses == 0).sum() * 1.0 / len(losses)

                val_loss.append(loss.item())
                val_acc.append(acc.item())

        epoch_loss = torch.tensor(val_loss).mean()  # Combine losses
        epoch_acc = torch.tensor(val_acc).mean()  # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def ResNet_size(size=10, in_channels=3, output_size=2, margin = 1.0):
    if size == 10:
        return Triplet_Network(BasicBlock, [1, 1, 1, 1], in_channels=in_channels, output_size=output_size, m=margin)
    elif size == 18:
        return Triplet_Network(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, output_size=output_size, m=margin)
    elif size == 34:
        return Triplet_Network(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, output_size=output_size, m=margin)
    elif size == 50:
        return Triplet_Network(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, output_size=output_size, m=margin)
    elif size == 101:
        return Triplet_Network(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, output_size=output_size, m=margin)
    elif size == 152:
        return Triplet_Network(Bottleneck, [3, 8, 36, 3], in_channels=in_channels, output_size=output_size, m=margin)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay=0.0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    # history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up learning rate scheduler
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.20, patience=2, verbose=True)

    for epoch in range(epochs):
        model.train()  # tells the model is in training mode, so batchnorm, dropout and all the ohter layer that have a training mode should get to the training mode

        train_losses = []
        lrs = []

        # Training Phase
        for (A,P,N) in train_loader:

            optimizer.zero_grad()  # Reset the gradients

            loss = model.training_step(A, P, N)  # Generate predictions, calculate loss
            train_losses.append(loss.item())

            loss.backward()  # Compute gradients

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()  # Adjust the weights

        # Record & update learning rate
        mean_loss = torch.tensor(train_losses).mean().item()
        lrs.append(get_lr(optimizer))
        sched.step(mean_loss)

        # Validation phase
        result = model.evaluate_step(val_loader)
        result['train_loss'] = mean_loss
        result['lrs'] = lrs

        print(f"Epoch [{epoch + 1}/{epochs}], last_lr: {lrs[-1]:.5f}, train_loss: {mean_loss:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


# Parameters
num_classes = 10
output_dim = 2
epochs = 50
max_lr = 0.5
margin = 10.0

grad_clip = 0.1 #0.1  # if ||g|| > u, g <- gu/||g||
weight_decay = 1e-4
opt_func = torch.optim.SGD

print(f"Output dimension: {output_dim}\n")

# Model (on Device)
tripletNetwork_model = to_device(ResNet_size(size=18, in_channels=1, output_size=output_dim, margin=margin), device)

# Train ResNet
print(f"[{datetime.now()}]")
start = time.time()

fit(epochs=epochs, max_lr=max_lr, model=tripletNetwork_model, train_loader=train_ld, val_loader=val_ld, weight_decay=weight_decay, grad_clip=grad_clip, opt_func=opt_func)

end = time.time()-start
print(f"[{datetime.now()}]")
print(f"\nTotal time = {int(end//3600):02d}:{int((end//60))%60:02d}:{end%60:.6f}")

#####################################################################################################################
del train_ld, val_ld
torch.cuda.empty_cache() # PyTorch thing
#####################################################################################################################
print("\nPLOTTING NEW SPACE\n")

reference_embeddings, reference_labels = tripletNetwork_model.extract_embedding(train_dataset)
''' Uncomment to plot the embeddings
if output_dim == 2:
    Ploting2D(reference_embeddings.T, reference_labels, "Learned Data Space")
if output_dim == 3:
    Ploting3D(reference_embeddings.T, reference_labels, "Learned Data Space")
'''
########################################### Evaluation ##############################################################
knn = KNeighborsClassifier(n_neighbors=1) #algorithm auto = ball_tree, kd_tree or brute
knn.fit(reference_embeddings, reference_labels)
#####################################################################################################################
print("\nPLOTTING GENERALIZATION\n")

query_embeddings, query_labels = tripletNetwork_model.extract_embedding(test_dataset)
''' Uncomment to plot the embeddings
if output_dim == 2:
    Ploting2D(query_embeddings.T, query_labels, "Learned Data Embedding")
if output_dim == 3:
    Ploting3D(query_embeddings.T, query_labels, "Learned Data Embedding")
'''
########################################### Evaluation ##############################################################
y_pred = knn.predict(query_embeddings)

recall = Metrics(query_labels, y_pred)

calculator = AccuracyCalculator( exclude=("AMI","mean_average_precision"),
                    avg_of_avgs=False,
                    k="max_bin_count",
                    label_comparison_fn=None)

acc_dict = calculator.get_accuracy(query_embeddings, reference_embeddings, query_labels, reference_labels, embeddings_come_from_same_source=False)

print("\nNMI: ", acc_dict["NMI"]*100)
print("p@1: ", acc_dict["precision_at_1"]*100)
print("RP: ", acc_dict["r_precision"]*100)
print("MAP@R: ", acc_dict["mean_average_precision_at_r"]*100)

#####################################################################################################################
#print("\nSAVING NETWORK MODEL\n")
'''The .state_dict method returns an OrderedDict containing all the weights and bias matrices mapped to the right attributes of the model'''
#File_name = "Log/TripletNet/MNIST/" + str(output_dim) + "d/MNIST_rec-" + str(recall) + "_bs-" + str(batch_size) + "_eps-" + str(epochs) + "_lr-" + str(max_lr) + "_m-" + str(margin) + ".pth"
#torch.save(tripletNetwork_model.state_dict(), File_name)
#####################################################################################################################
''' Uncomment to plot the embeddings
if output_dim <= 3:
    plt.show()
'''