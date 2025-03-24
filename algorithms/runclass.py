# This is a class to run the training and evaluation. It is inspired in lab8. It will allow me to instance according to the needs
# this class will implement the training and evaluation algorithms (validation and testing)
class ImplementDLEv():

    def get_data_loaders(batch_size=64, num_workers=2):
        """Prepares data loaders for CIFAR-10 dataset.
    
        Args:
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of workers for data loading.
    
        Returns:
            tuple: train_loader, val_loader, and test_loader (DataLoader objects).
        """
        # Data transformations (resize, normalize)
        transform = transforms.Compose([
            transforms.Resize(224),  # Resizing image to 224x224 for ResNet input
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image with mean=0.5, std=0.5
        ])
    
        # Load CIFAR-10 dataset
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
        # Split dataset into training and validation sets (80/20 split)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
        # Load the test dataset
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
        # DataLoader for train, validation, and test datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
        return train_loader, val_loader, test_loader
    
    def build_model():
        """Builds the ResNet-34 model for CIFAR-10 classification.
    
        Returns:
            model: The ResNet-34 model with modified fully connected layer.
        """
        model = models.resnet34(weights=None)
        num_ftrs = model.fc.in_features  # Get the number of features in the last fully connected layer
        model.fc = nn.Linear(num_ftrs, 10)  # Modify the fully connected layer for CIFAR-10 (10 classes)
        model = model.to(device)  # Move model to device (GPU or CPU)
        return model
    
    def get_optimizer_and_loss(model, lr=0.001):
        """Returns the optimizer and loss function for training.
    
        Args:
            model: The neural network model.
            lr (float): Learning rate for the optimizer.
    
        Returns:
            tuple: The loss function (CrossEntropyLoss), and optimizer (Adam)
        """
        criterion = torch.nn.CrossEntropyLoss()# Loss function for classification # Your code Here. Aim for 1 line.
        optimizer = torch.optim.Adam(model.parameters(),lr)# Adam optimizer # Your code Here. Aim for 1 line.
        return criterion, optimizer
    
    def compute_loss(model, data, target, criterion):
        """Computes the loss.
    
        Args:
            model: The neural network model.
            data: The input data (images).
            target: The ground truth labels.
            criterion: The loss function.
    
        Returns:
            tuple: Model output and loss value.
        """
        output = model(data)# Forward pass # Your code Here. Aim for 1 line.
        loss =   criterion(output,target)# Compute loss # Your code Here. Aim for 1 line.
        return output, loss
    
    def evaluate(loader, model, criterion, epoch, mode="Validation"):
        """Evaluates the model on the given dataset (train, val, or test).
    
        Args:
            loader: DataLoader object for the dataset.
            model: The trained model.
            criterion: The loss function.
            epoch: The current epoch number.
            mode (str): Mode of evaluation (e.g., "Validation", "Test").
    
        Returns:
            tuple: Average loss and accuracy for the dataset.
        """
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        total_loss = 0.0
    
        with torch.no_grad():  # Disable gradient computation for evaluation
            for data, target in loader:
                data, target = data.to(device), target.to(device)  # Move data to device
                output, loss = compute_loss(model, data, target, criterion)  # Compute output and loss
    
                total_loss += loss.item()  # Accumulate loss
                _, predicted = torch.max(output, 1)  # Get the predicted class
                total += target.size(0)  # Number of samples
                correct += (predicted == target).sum().item()  # Count correct predictions
    
        # Compute average loss and accuracy
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def run_optimizer(optimizer, loss):
        """Performs and optimizer step.
    
        Args:
            optimizer: The optimizer for training.
            loss: The loss value.
        """
        # Your code Here. Aim for 3 lines.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def train_epoch(model, train_loader, criterion, optimizer, epoch):
        """Performs training for a single epoch.
    
        Args:
            model: The neural network model.
            train_loader: DataLoader for the training dataset.
            criterion: The loss function.
            optimizer: The optimizer for training.
            epoch (int): Current epoch number.
    
        Returns:
            tuple: Average training loss and average batch time.
        """
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        batch_times = []
    
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()  # Track batch processing time
            data, target = data.to(device), target.to(device)  # Move data to device
            output, loss = compute_loss(model, data, target, criterion)  # Compute loss
            run_optimizer(optimizer, loss)  # Update weights based on loss
            epoch_loss += loss.item()  # Accumulate loss
            batch_times.append(time.time() - start_time)  # Track batch time
    
        avg_train_loss = epoch_loss / len(train_loader)  # Compute average loss for the epoch
        avg_batch_time = sum(batch_times) / len(batch_times)  # Compute average batch time
    
        return avg_train_loss, avg_batch_time
    
    def train_model(num_epochs=3, batch_size=128, lr=0.001):
        """Main function to train the ResNet-34 model.
    
        Args:
            num_epochs (int): Number of epochs for training.
            batch_size (int): Batch size for data loading.
            lr (float): Learning rate for the optimizer.
    
        Returns:
            model: The trained ResNet-34 model.
        """
        # Check for CUDA availability, and set device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        train_loader, val_loader, test_loader = get_data_loaders(batch_size)  # Get data loaders
        model = build_model()  # Build the model
        criterion, optimizer = get_optimizer_and_loss(model, lr)  # Get optimizer and loss function
    
        for epoch in range(num_epochs):
            avg_train_loss, avg_batch_time = train_epoch(model, train_loader, criterion, optimizer, epoch)
    
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Avg Batch Time: {avg_batch_time:.4f}s")
    
            # Evaluate on validation set
            val_loss, val_acc = evaluate(val_loader, model, criterion, epoch, mode="Validation")
    
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
        # Final evaluation on test set
        test_loss, test_acc = evaluate(test_loader, model, criterion, num_epochs, mode="Test")
        print(f"Final Test Accuracy: {test_acc:.4f}")
    
        return model  # Return the trained model


    