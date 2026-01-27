import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# --- 1. ARCHITECTURE DEFINITION ---

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_residual=True):
        super(BasicBlock, self).__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_residual=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, self.use_residual))
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

# --- 2. TRAINING AND EVALUATION FUNCTIONS ---

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return 100. * (1 - correct / total)  # Return Error Rate

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * (1 - correct / total)

# --- 3. MAIN EXPERIMENT ---

def run_experiment(epochs=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Data Preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model_configs = [
        ("Plain-18", [2, 2, 2, 2], False),
        ("ResNet-18", [2, 2, 2, 2], True),
        ("Plain-34", [3, 4, 6, 3], False),
        ("ResNet-34", [3, 4, 6, 3], True),
    ]

    results = {}

    for name, layers, use_res in model_configs:
        print(f"\n--- Training {name} ---")
        model = ResNet(BasicBlock, layers, use_residual=use_res).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_errors = []
        test_errors = []

        for epoch in range(epochs):
            tr_err = train(model, trainloader, criterion, optimizer, device)
            te_err = test(model, testloader, device)
            train_errors.append(tr_err)
            test_errors.append(te_err)
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} | Train Err: {tr_err:.2f}% | Test Err: {te_err:.2f}%")

        results[name] = {'train': train_errors, 'test': test_errors}

    # --- 4. PLOTTING (FIGURE 4 STYLE) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in results.items():
        ls = '-' if 'ResNet' in name else '--'
        color = 'red' if '34' in name else 'blue'
        ax1.plot(data['train'], label=name, linestyle=ls, color=color)
        ax2.plot(data['test'], label=name, linestyle=ls, color=color)

    ax1.set_title("Training Error (%)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Test Error (%)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Error")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('./resnet_figure4_demo.png')
    plt.show()

if __name__ == "__main__":
    run_experiment(epochs=100) # Increase epochs for better convergence