from trainer.base import BaseTrainer
import os
from utils.metrics import ClassificationMetrics
import torch

class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, dataset, device, callbacks=None, lr_scheduler=None, logger=None):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.logger = logger
        self.eval_metrics = ClassificationMetrics()
        self.dataset = dataset
        self.callbacks = callbacks
        self.init_dataloader(dataset=dataset)

    def init_dataloader(self, dataset):
        self.trainloader = dataset.train_dataloader()
        self.valloader = dataset.val_dataloader()

    def load_model(self):
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name)
        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])


    def save_final_model(self):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])
        
        model_saved_name = os.path.join(self.cfg['output_dir'], 'final_model.pth')
        optimizer_saved_name = os.path.join(self.cfg['output_dir'], 'final_optimizer.pth')
        torch.save(self.network.state_dict(), model_saved_name)
        torch.save(self.optimizer.state_dict(), optimizer_saved_name)
    
    def train_one_epoch(self, epoch):
        self.network.train()
        loss = 0
        for i, (imgs, labels) in enumerate(self.trainloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            preds = self.network(imgs)
            self.optimizer.zero_grad()
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()
            loss += loss.item() * imgs.shape[0]

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        epoch_loss = loss / len(self.dataset.train_set)
        if self.logger is not None:
            self.logger.log_metric("train_loss", epoch_loss, epoch=epoch)

        return epoch_loss
    
    def train(self):
        self.network.freeze() # freeze feature extract.
        for epoch in range(self.cfg['train']['num_epochs']):
            if epoch == 5:
                self.network.unfreeze() # unfreeze feature extract after 5 epoch.
            print("="*80)
            print("Epoch: {}".format(epoch))
            train_loss = self.train_one_epoch(epoch)
            val_acc, val_prec, val_rec, val_f1, val_loss = self.validate_one_epoch(epoch)
            print("Train Loss: {:.4f}, Val Loss: {:.4f}Accuracy: {:.4f}\nPrecision: {:.4f}, Recall: {:.4f}, F1_score: {:.4f}"\
                .format(train_loss,val_loss,val_acc, val_prec, val_rec, val_f1))
            print("="*80)
            if self.callbacks is not None:
                result_val_metris = {
                    'accuracy':val_acc,
                    'precision':val_prec,
                    'recall': val_rec,
                    'f1_score': val_f1
                }
                self.callbacks(self.network, self.optimizer, result_val_metris)

                if self.callbacks.early_stop:
                    print('Custom Callback Triggered, Process Training Stopped!')
                    break
        
        self.save_final_model()
    
    def validate_one_epoch(self, epoch):
        acc = precision = recall = f1 = val_loss = 0
        self.network.eval()
        with torch.no_grad():
            for i, (imgs,labels) in enumerate(self.valloader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self.network(imgs)
                loss = self.criterion(preds, labels)
                # print("Batch val loss", loss)
                val_loss += loss.item() * imgs.shape[0]
                classification_metrics = self.eval_metrics(labels, preds)
                acc += classification_metrics['accuracy']
                precision += classification_metrics['precision']
                recall += classification_metrics['recall']
                f1 += classification_metrics['f1_score']
                # print classification_metrics
            # print(len(self.dataset.val_set))
            total_val_dataset = len(self.dataset.val_set)
            epoch_val_loss = loss / total_val_dataset
            epoch_accuracy = acc / total_val_dataset
            epoch_precision = precision / total_val_dataset
            epoch_recall = recall / total_val_dataset
            epoch_f1_score = f1 / total_val_dataset
            
            if self.logger is not None:
                self.logger.log_metric("epoch_val_loss", epoch_val_loss, epoch=epoch)
                self.logger.log_metric("epoch_accuracy", epoch_accuracy, epoch=epoch)
                self.logger.log_metric("epoch_precision", epoch_precision, epoch=epoch)
                self.logger.log_metric("epoch_recall", epoch_recall, epoch=epoch)
                self.logger.log_metric("epoch_f1_score", epoch_f1_score, epoch=epoch)

        return epoch_accuracy, epoch_precision, epoch_recall, epoch_f1_score, epoch_val_loss
