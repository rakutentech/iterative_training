from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import util.dataset
import util.io


class Patience:
    def __init__(self, capacity = 10):
        self._tolerance = self._capacity = int(capacity)
        self._calmness = 0.0
    
    def __call__(self):
        if self._tolerance > 0:
            return True
        else:
            return False

    def test(self, calmness):
        if calmness > self._calmness:
            self._calmness = calmness
            self._tolerance = self._capacity
        else:
            self._tolerance -= 1
        return self

def train(args, model, data_loaders, device, writer):
    torch.manual_seed(args.seed)

    train_loader, validation_loader, test_loader = data_loaders

    if hasattr(model, 'criterion'):
        criterion = getattr(model, 'criterion')
    else:
        print("No criterion in model. Assuming nn.CrossEntropyLoss().")
        criterion = nn.CrossEntropyLoss()

    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(optimizer)
    if hasattr(model, 'scheduler'):
        print(model.scheduler)

    model.to(device)
    if args.load_last:
        print("Loading saved model: {}".format(args.load_last_file))
        util.io.load(args.load_last_file, model, optimizer)
        state = 0, 0.0, 0, float("Inf") # best_epoch, best_accuracy, best_loss_epoch, best_loss
        print("Testing loaded model...")
        accuracy, state = test_epoch(0, args, model, test_loader, criterion, optimizer, device, writer, 'test/', state)

    state = 0, 0.0, 0, float("Inf") # best_epoch, best_accuracy, best_loss_epoch, best_loss
    train_state = state
    patient = Patience(args.patience)
    epoch = 1
    while patient() or args.patience == 0:
        if hasattr(model, 'scheduler'):
            model.scheduler.step()
        if args.reverse_layer_binarization:
            if hasattr(model, 'binarize_layer_reverse'):
                model.binarize_layer_reverse()
            else:
                raise NotImplementedError("reverse_layer_binarization requested, but binarize_layer_reverse() does not exist")
        else:
            if hasattr(model, 'binarize_layer'):
                model.binarize_layer()
        train_epoch(epoch, args, model, train_loader, criterion, optimizer, device, writer)
        if args.test_validation:
            accuracy, state = test_epoch(epoch, args, model, validation_loader, criterion, optimizer, device, writer, 'val/', state)
        else:
            accuracy, state = test_epoch(epoch, args, model, test_loader, criterion, optimizer, device, writer, 'test/', state)

        if args.test_training:
            _, train_state = test_epoch(epoch, args, model, train_loader, criterion, optimizer, device, writer, 'train/', train_state)

        if hasattr(model, 'scheduler'):
            writer.add_scalar('train/learning_rate', model.scheduler.get_last_lr()[0], epoch)

        if args.patience == 0:
            if epoch >= args.epochs: break
        else:
            patient.test(accuracy)
        epoch += 1


    best_epoch, best_accuracy, best_loss_epoch, best_loss = state
    print('Best accuracy {:.4f}% at epoch {}'.format(100*best_accuracy, best_epoch))
    print('Best loss {:.4g} at epoch {}'.format(best_loss, best_loss_epoch))
    if args.test_training:
        best_epoch, best_accuracy, best_loss_epoch, best_loss = train_state
        print('Best training accuracy {:.4f}% at epoch {}'.format(100*best_accuracy, best_epoch))
        print('Best training loss {:.4g} at epoch {}'.format(best_loss, best_loss_epoch))
    if args.save_last:
        util.io.save(epoch, model, optimizer, best_loss, args.logdir, pattern='last-{}.tar')


def train_epoch(epoch, args, model, data_loader, criterion, optimizer, device, writer):
    print("Logdir {}".format(args.logdir))
    print("Model {} {}".format(args.model, model.__class__.__name__))
    model.train()
    pid = os.getpid()
    start_epoch = time.time()
    for batch_idx, (data, target) in enumerate(data_loader, 0):
        start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #weight_stats("{} {:>3} optimizer.zero_grad()".format(epoch, batch_idx), model)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        #weight_stats("{} {:>3} loss.backward()      ".format(epoch, batch_idx), model)
        if hasattr(model, 'restore_full_precision'):
            #print("=== before")
            #model.report_weight_stats()
            model.restore_full_precision()
            #print("=== after")
            #model.report_weight_stats()
        optimizer.step()
        if hasattr(model, 'clip_weights'):
            model.clip_weights()
        #weight_stats("{} {:>3} optimizer.step()     ".format(epoch, batch_idx), model)
        if batch_idx % args.log_interval == 0:
            print('{}  Train Epoch: {} [{:>5}/{:>5} {:.0f}%]  Loss: {:.6g}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    print('Epoch duration: {:.4f} seconds'.format(time.time() - start_epoch))
    record_weight_histogram(args, epoch, model, writer)


def record_weight_histogram(args, epoch, model, writer):
    if args.record_histogram:
        start = time.time()
        for name, param in model.named_parameters():
            x = param.clone().cpu().data.numpy()
            writer.add_histogram(name+"/train", x, epoch)
            mantissa, exponent = np.frexp(x)
            writer.add_histogram(name+'.mantissa/train', mantissa, epoch)
            writer.add_histogram(name+'.exponent/train', exponent, epoch)
        print('Weight histogram written in {:.4f} seconds'.format(time.time() - start))


def test_epoch(epoch, args, model, data_loader, criterion, optimizer, device, writer, writer_prefix, state):
    model.eval()
    loss = 0.0
    correct = 0
    start = time.time()
    #criterion = getattr(model, 'criterion')
    with torch.no_grad():
        if hasattr(model, 'restore_binary_weight'):
            print("test_epoch::restore_binary_weight")
            model.restore_binary_weight()
        if hasattr(model, 'report_weight_stats'):
            print("test_epoch::report_weight_stats")
            maximum, minimum, num_zeros, num_binary = model.report_weight_stats()
            writer.add_scalar('{}weight/max'.format(writer_prefix), maximum, epoch)
            writer.add_scalar('{}weight/min'.format(writer_prefix), minimum, epoch)
            writer.add_scalar('{}weight/num_zeros'.format(writer_prefix), num_zeros, epoch)
            writer.add_scalar('{}weight/num_binary'.format(writer_prefix), num_binary, epoch)
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #loss += F.nll_loss(output, target).item() # sum up batch loss
            loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
        if hasattr(model, 'restore_full_precision'):
            print("test_epoch::restore_full_precision")
            model.restore_full_precision()

    loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    print('Prefix {}, Average loss: {:.6g}, Accuracy: {}/{} ({:.4f}%), Duration: {:.4f} seconds'.format(
        writer_prefix,
        loss, 
        correct, len(data_loader.dataset), 100. * accuracy, 
        time.time() - start))

    writer.add_scalar('{}accuracy'.format(writer_prefix), accuracy, epoch)
    writer.add_scalar('{}error'.format(writer_prefix), 1.0-accuracy, epoch)
    writer.add_scalar('{}loss'.format(writer_prefix), loss, epoch)

    best_epoch, best_accuracy, best_loss_epoch, best_loss = state
    check = epoch, accuracy
    best = best_epoch, best_accuracy
    prefix = "accuracy_"+writer_prefix.replace("/", "")
    best_epoch, best_accuracy = save_progress(prefix, args, model, optimizer, check, best)
    check = epoch, loss
    best = best_loss_epoch, best_loss
    prefix = "loss_"+writer_prefix.replace("/", "")
    best_loss_epoch, best_loss = save_progress(prefix, args, model, optimizer, check, best, higher_better=False)
    
    writer.add_scalar('{}best_accuracy'.format(writer_prefix), best_accuracy, epoch)
    writer.add_scalar('{}best_error'.format(writer_prefix), 1.0 - best_accuracy, epoch)
    writer.add_scalar('{}best_loss'.format(writer_prefix), best_loss, epoch)
    #writer.add_scalar('{}learning_rate'.format(writer_prefix), scheduler.get_lr(), epoch)

    state = best_epoch, best_accuracy, best_loss_epoch, best_loss
    return accuracy, state


def save_progress(name, args, model, optimizer, check, best, higher_better=True):
    epoch, metric = check
    best_epoch, best_metric = best
    if higher_better:
        better = metric > best_metric
    else:
        better = metric < best_metric
    if better:
        print("Better {} {:.4f} at epoch {}".format(name, metric, epoch))
        if args.save_progress:
            util.io.save(epoch, model, optimizer, metric, args.logdir, pattern=name+'-{}.tar')
        return epoch, metric
    else:
        return best_epoch, best_metric


def isnan(tensor, message):
    if torch.isnan(tensor):
        print(message)


def weight_stats(prefix, model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            high = torch.max(param)
            low = torch.min(param)
            print("{} {:>.6g} {:>.6g} {} {}".format(prefix, high, low, name, param.size()))
