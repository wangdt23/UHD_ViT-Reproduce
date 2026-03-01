# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from src.models.vit import HierarchicalViT
from src.data.loader import get_loaders

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def train():
    # load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # dataloaders(read pictures,resize,normalize,augment)
    train_loader, test_loader = get_loaders(config['train']['batch_size'])

    # init model, loss, optimizer
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    model_config = config['model'].copy()
    model_name = model_config.pop('name')
    model = HierarchicalViT(**model_config).to(device)
    criterion = nn.CrossEntropyLoss() # loss function for classification
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr']) # AdamW optimizer with weight decay for better generalization

    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs']) # cosine annealing learning rate scheduler for smoother convergence

    best_acc = 0.0

    # Training loop
    for epoch in range(config['train']['epochs']):
        # 1. Train for one epoch
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # compute loss
            loss.backward() # backpropagation
            optimizer.step() # update weights

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss/len(train_loader))

        # 2. Validate on test set
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        schedular.step() # update learning rate according to scheduler
        print(f"Learning Rate after epoch {epoch+1}: {optimizer.param_groups[0]['lr']:.6f}") # print current learning rate for debugging

        # 3. Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")

    print("Training Finished!")
    torch.save(model.state_dict(), "best_model.pth")

def validate(model,loader,criterion,device):
    model.eval() # set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # no need to compute gradients during validation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = val_loss / len(loader)
        accuracy = correct / total*100
        return avg_loss, accuracy

if __name__ == "__main__":
    train()