import numpy as np
import sys 
import torch
from utils.utils import *
import os
from datsets.dataset_generic import save_splits
from models.model import stableMIL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score,roc_curve,f1_score
from sklearn.metrics import auc as calc_auc


model_dict = {
'stable':stableMIL,

}

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        # print(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, stop_epoch=50,verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf


    def __call__(self, epoch, val_loss,c,model,ckpt_name = 'checkpoint.pt'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss




def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')

    loss_fn = nn.CrossEntropyLoss()

    print('Done!')
    
    print('\nInit Model {}...'.format(args.model_type), end=' ')

    
    stable_paramas = {
        'depth': args.depth,'aggregate_num' :args.aggregate_num,'num_heads':args.num_heads,
        'max_dist':args.max_dist + 1e-6 , 'k_neighbors':args.k_neighbors,'drop_rate':args.drop,
        'ratio':args.ratio,
        'drop_path_rate':args.drop_path,}
    
    param_dict = {'dim':args.input_dim,'n_classes':args.n_classes,'task':args.task}

    if args.model_type == 'stable':
        param_dict.update(stable_paramas)
    
    model = model_dict[args.model_type](**param_dict)
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch=50,verbose = True)

    else:
        early_stopping = None
    print('Done!')
    
    best_model = None
    best_val_loss = 1e9
    best_epoch = -1

    for epoch in range(args.max_epochs):

        train_error = train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,args.gacc)
        stop,best_model,best_val_loss , best_epoch= validate(best_epoch,best_val_loss,best_model,cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir,train_error)
            
        if epoch != 0 and (epoch + 1)%10==0:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_{}.pt".format(cur,epoch+1)))    
        if stop: 
            break
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    torch.save(best_model.state_dict(), os.path.join(args.results_dir, "best_val_epoch_{}_checkpoint.pt".format(best_epoch)))

    _, val_error, val_auc,val_f1, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc,test_f1, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('best/val_error', val_error, 0)
        writer.add_scalar('best/val_auc', val_auc, 0)

        writer.add_scalar('best/val_f1', val_f1, 0)
        writer.add_scalar('best/test_error', test_error, 0)
        writer.add_scalar('best/test_auc', test_auc, 0)
   
        writer.add_scalar('best/test_f1', test_f1, 0)
        writer.close()



    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error ,test_f1,val_f1

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,gacc_step= 1):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label,coords,fuse_labels, fuse_sorted_idx, region_indices, region_sort_index, attention_mask_1, attention_mask_2) in enumerate(loader):
        data, label, coords = data.to(device), label.to(device), coords.to(device)

        data = data.unsqueeze(dim = 0)
        coords = coords.unsqueeze(dim = 0)

        fuse_labels = fuse_labels.to(device)
        fuse_sorted_idx = fuse_sorted_idx.to(device)
        region_indices = region_indices.to(device)
        region_sorted_index = region_sorted_index.to(device)
        attention_mask_1 = attention_mask_1.to(device)
        attention_mask_2 = attention_mask_2.to(device)
        
        logits, Y_prob, Y_hat= model(data,coords,fuse_labels,fuse_sorted_idx,region_indices,region_sorted_index,attention_mask_1,attention_mask_2)

        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value

        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        loss= loss/gacc_step

        loss.backward()

        if (batch_idx + 1)%gacc_step == 0:
            optimizer.step()
            optimizer.zero_grad()
  
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
    return train_error
   
def validate(best_epoch,best_val_loss,best_model,cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None,train_error = 1.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    hat = np.zeros((len(loader)))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label,coords,fuse_labels, fuse_sorted_idx, region_indices, region_sort_index, attention_mask_1, attention_mask_2) in enumerate(loader):
            data, label, coords = data.to(device), label.to(device), coords.to(device)

            data = data.unsqueeze(dim = 0)
            coords = coords.unsqueeze(dim = 0)

            fuse_labels = fuse_labels.to(device)
            fuse_sorted_idx = fuse_sorted_idx.to(device)
            region_indices = region_indices.to(device)
            region_sorted_index = region_sorted_index.to(device)
            attention_mask_1 = attention_mask_1.to(device)
            attention_mask_2 = attention_mask_2.to(device)
            
            logits, Y_prob, Y_hat= model(data,coords,fuse_labels,fuse_sorted_idx,region_indices,region_sorted_index,attention_mask_1,attention_mask_2)


            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            hat[batch_idx] = Y_hat.cpu().numpy().item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
    val_f1 = f1_score(labels,hat,average='macro')

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/f1', val_f1, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f},auc: {:.4f}, val_f1: {:.4f}'
            .format(val_loss,val_error, auc,val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if best_val_loss > val_loss:
        best_val_loss = val_loss
        best_model = model
        best_epoch = epoch

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, train_error,ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True ,best_model,best_val_loss,best_epoch

    return False,best_model,best_val_loss,best_epoch

def validate_clam(best_epoch,best_val_loss,best_model,cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None,train_error=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        best_model = model
        best_epoch = epoch

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss,auc, model,train_error, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True , best_model,best_val_loss,best_epoch

    return False,best_model,best_val_loss,best_epoch

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    hat = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label,coords,fuse_labels, fuse_sorted_idx, region_indices, region_sort_index, attention_mask_1, attention_mask_2) in enumerate(loader):
        data, label, coords = data.to(device), label.to(device), coords.to(device)

        data = data.unsqueeze(dim = 0)
        coords = coords.unsqueeze(dim = 0)

        fuse_labels = fuse_labels.to(device)
        fuse_sorted_idx = fuse_sorted_idx.to(device)
        region_indices = region_indices.to(device)
        region_sorted_index = region_sorted_index.to(device)
        attention_mask_1 = attention_mask_1.to(device)
        attention_mask_2 = attention_mask_2.to(device)


        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat= model(data,coords,fuse_labels,fuse_sorted_idx,region_indices,region_sorted_index,attention_mask_1,attention_mask_2)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        hat[batch_idx] = Y_hat.cpu().numpy().item()
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error
    
    f1 = f1_score(all_labels,hat,average='macro')

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc,f1, acc_logger
