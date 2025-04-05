
import datasetsnc.dataclasses
import torch.utils.data
import time
import inspect
from datasetsnc.dataclasses import *
from torch.utils.data import random_split
from datetime import datetime
from matplotlib.ticker import MaxNLocator


# This is a class to run the training and evaluation. It is inspired in lab8. It will allow me to instance according to the needs
# this class will implement the training and evaluation algorithms (validation and testing)
class ImplementDLEv():
    def __init__(self, fastafile="", batch_size=64,num_workers=2, 
                 model = None, device="cpu", train_ratio = 0.7,
                 val_ratio = 0.15,test_ratio = 0.15,lr=0.001, num_epochs = 3,seq_length=120,
                 subset = 0, chkpoint=False, chkpath = "",random_rev_compl_transform_prob = 0
                 
                ):
                 
        """
        Initializes the deep learning training and evaluation class.

        Parameters:
        - fastafile: 
        - batch_size (int): Batch size for data loading.
        - num_workers (int): Number of workers for data loading.
        - model: The deep learning model to be trained and evaluated.
        - device: The device to run the model on ("cpu" or "cuda").
        """
        self.fastafile = fastafile
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = model
        self.device = device
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.lr = lr
        self.num_epochs = num_epochs
        self.subset = subset
        self.seq_length = seq_length
        self.chkpoint = chkpoint
        self.chkpath = chkpath
        self.random_rev_compl_transform_prob = random_rev_compl_transform_prob
        
        # variables which are feeding from methods

        self.train_loader = None 
        self.val_loader = None 
        self.test_loader = None 
        self.output = None 
        self.loss = None
        self.criterion = None
        self.optimizer = None
        self.data = None
        self.target =None
        self.class_accuracy = None
        self.acum_train_loss = []
        self.acum_val_loss = []
        
        # Move model to the specified device
        self.model.to(self.device)

    #*************************************************
    
    def get_data_loaders(self):
       
        fastafile = self.fastafile
        batch_size = self.batch_size
        num_workers= self.num_workers
        train_ratio = self.train_ratio
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio

        tot_ratio=train_ratio + val_ratio+test_ratio
        if tot_ratio != 1:
            print(f"the data split does not sum up to 1 (or 100%):{tot_ratio}") 
    
        # Load dataset
        full_dataset = NcRnaDataset(fastafile, self.seq_length, self.random_rev_compl_transform_prob)
        
        if self.subset > 0:
            full_dataset = Subset(full_dataset,range(self.subset)) 

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size  

        # Division du dataset
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

        # DataLoader for train, validation, and test datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        return 

    #*************************************************
    
    def build_model(self):
        model = self.model
        device = self.device
       
        model = model.to(device)  # Move model to device (GPU or CPU)
        self.model = model
        return 
    
    #*************************************************
    
    def get_optimizer_and_loss(self):
        
        lr = self.lr
        model = self.model
        criterion = torch.nn.CrossEntropyLoss()# Loss function for classification 
        optimizer = torch.optim.Adam(model.parameters(),lr)# Adam optimizer
        self.criterion = criterion
        self.optimizer = optimizer
        return 
    
    #*************************************************
    
    def compute_loss(self):
       
        model = self.model
        data = self.data
        target = self.target
        criterion = self.criterion
        
        output = model(data) # Forward pass 
        loss =   criterion(output,target)# Compute loss 
        self.output = output
        self.loss = loss
        return 
    
    #*************************************************
    
    def evaluate(self,epoch,mode="Validation"):
      
        loader = None
        if mode == "Validation":
            loader = self.val_loader
        if mode == "Test":
            loader = self.test_loader
            
        
        self.model.eval()  
        correct = 0
        total = 0
        total_loss = 0.0

        class_correct = [0] * self.model.num_classes  
        class_total = [0] * self.model.num_classes  
    
    
        with torch.no_grad():  
            for data, target in loader:
                self.data, self.target = data.to(self.device).float(), target.to(self.device)  
                
                self.compute_loss()  
    
                total_loss += self.loss.item()  
                _, predicted = torch.max(self.output, 1)  
                total += self.target.size(0)  
                correct += (predicted == self.target).sum().item()  

                 # Update class-wise correct and total counts
                for i in range(self.target.size(0)):  
                    label = self.target[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1

    
        # Compute average loss and accuracy
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        # Compute accuracy per class
        self.class_accuracy = [correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
        self.class_accuracy = [round(nnn, 3) for nnn in self.class_accuracy]
    
        
        return avg_loss, accuracy
    
    #*************************************************
    
    def run_optimizer(self):
    
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    #*************************************************    
    
    def train_epoch(self,epoch):
        device = self.device
        
        self.model.train() 
        epoch_loss = 0.0
        batch_times = []
    
        for batch_idx, (data, target) in enumerate(self.train_loader):
            start_time = time.time()  
            self.data, self.target = data.to(device).float(), target.to(device)  
            self.compute_loss()  
            self.run_optimizer() 
            epoch_loss += self.loss.item() 
            batch_times.append(time.time() - start_time)  
    
        avg_train_loss = epoch_loss / len(self.train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times)  
    
        return avg_train_loss, avg_batch_time
    
    #*************************************************
    
    def train_model(self):
        print()
        print("Run Information")
        print("---------------")
        
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        lr = self.lr
    
    
        self.get_data_loaders()  
        self.build_model()  
        self.get_optimizer_and_loss()
    
        for epoch in range(num_epochs):
            avg_train_loss, avg_batch_time = self.train_epoch(epoch)
            
            if self.chkpoint:
                self.checkpointev(epoch)
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Avg Batch Time: {avg_batch_time:.4f}s")
    
            val_loss, val_acc = self.evaluate(epoch, mode="Validation")
    
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            print(f"Validation Class Accuracy:  {self.class_accuracy}")
            
            self.acum_train_loss.append(avg_train_loss)
            self.acum_val_loss.append(val_loss)
        
        test_loss, test_acc = self.evaluate(num_epochs, mode="Test")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Final Test Class Accuracy:  {self.class_accuracy} ")         
        print ()
        return   

    #*************************************************
    
    def run_train(self):
        model = self.train_model()
        self.model = model 
        return

    #*************************************************

    def checkpointev(self,epoch):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion': self.criterion,
            'loss': self.loss,
            'batch_size': self.batch_size,
               
        },
        (f"{self.chkpath}checkpoint_{self.model.getModelName()}_"
         f"{epoch}_{self.batch_size}_{self.optimizer.param_groups[0]['lr']}_"
         f"{timestamp}.pth")
    )
         
    #*************************************************
    # Plot performance
    def performance_plot(self,title="Model Performance: Training vs Validation Loss"):
        plt.plot(self.acum_train_loss, color='green', linestyle='--')
        plt.plot(self.acum_val_loss, color='orange', linestyle='--')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['Train Loss','Validation Loss'])
        plt.title(title)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        #plt.show()


    #************************************************
    # Accuracy x class
    def accuracy_x_class_plot(self,title="Per-Class Accuracy (Test Dataset)"):
        num_ncrna_class = len(self.class_accuracy)
        classes_ncrna = list(range(num_ncrna_class))  
        plt.bar(classes_ncrna, self.class_accuracy, color='purple')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.ylim(0, 1.05) 
        plt.xticks(classes_ncrna) 
        #plt.show()

    #*********************************************

    def print_settings(self):
        init_settings = inspect.signature(self.__init__).parameters
        class_settings = "\n".join([f"{param}: {getattr(self, param)}" for param in init_settings if param != 'self'])
        print("************************************************")
        print("Model Hyperparameters and Configuration Settings")
        print("------------------------------------------------")
        print(class_settings)
        print()
        return

    #*********************************************

    def perform_accuracy_gether(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)  # (rows, cols, index)
        self.performance_plot()  # Your existing function
        plt.subplot(1, 2, 2)  # (rows, cols, index)
        self.accuracy_x_class_plot() # Your existing function
        plt.tight_layout()  
        plt.show()
    
    


    