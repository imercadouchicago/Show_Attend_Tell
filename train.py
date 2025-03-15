import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter 
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from dataset import ImageCaptionDataset
from model import Encoder, Decoder


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                         std=[0.229, 0.224, 0.225])  # Normalize with ImageNet standard deviation
])


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f" % (optimizer.param_groups[0]['lr'],))


def main(args):
    writer = SummaryWriter()

    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim, args.tf)

    if args.model:  # If a pre-trained model is provided
        decoder.load_state_dict(torch.load(args.model))  # Load model weights

    encoder.cuda()
    decoder.cuda()

    # Separate optimizers for encoder and decoder with different learning rates
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    
    # Encoder optimizer only if fine-tuning is enabled
    if args.fine_tune_encoder:
        encoder.fine_tune(True)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    else:
        encoder.fine_tune(False)
        encoder_optimizer = None
    
    scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, args.step_size)
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, split_type='val'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    # Create model version identifier to avoid overwriting existing models
    model_version = args.version
    if model_version is None:
        # Generate timestamp-based version if none provided
        from datetime import datetime
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Using model version: {model_version}")
    model_prefix = f"model_{args.network}_{model_version}"
    
    print('Starting training with {}'.format(args))
    best_bleu4 = 0.0
    epochs_since_improvement = 0
    for epoch in range(1, args.epochs + 1):
        # Check for early stopping
        if epochs_since_improvement == args.early_stop:
            print("Early stopping triggered after {} epochs without improvement".format(
                epochs_since_improvement))
            break
            
        # Reduce learning rate if no improvement for specified epochs
        if epochs_since_improvement > 0 and epochs_since_improvement % args.lr_decay_epochs == 0:
            adjust_learning_rate(decoder_optimizer, args.lr_decay_factor)
            if args.fine_tune_encoder and encoder_optimizer:
                adjust_learning_rate(encoder_optimizer, args.lr_decay_factor)

        if scheduler:
            scheduler.step()

        # Training phase
        train(epoch, encoder, decoder, decoder_optimizer, encoder_optimizer, 
              cross_entropy_loss, train_loader, word_dict, args.alpha_c, 
              args.log_interval, writer, args.grad_clip)

        # Validation phase
        recent_bleu4 = validate(epoch, encoder, decoder, cross_entropy_loss, 
                                val_loader, word_dict, args.alpha_c, 
                                args.log_interval, writer)
                                
        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        
        if is_best:
            epochs_since_improvement = 0
            # Save best model with unique name
            save_path = f"{args.data}/checkpoint_{model_prefix}_best.pth.tar"
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict() if encoder_optimizer else None,
                'bleu4': best_bleu4
            }, save_path)
            print(f"Saved best model to {save_path}")
        else:
            epochs_since_improvement += 1
            print(f"No improvement in BLEU-4. Epochs since last improvement: {epochs_since_improvement}")

        # Save regular checkpoint with unique name
        save_path = f"{args.data}/checkpoint_{model_prefix}_epoch{epoch}.pth.tar"
        torch.save({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict() if encoder_optimizer else None,
            'bleu4': recent_bleu4
        }, save_path)
        print(f"Saved checkpoint to {save_path}")


def train(epoch, encoder, decoder, decoder_optimizer, encoder_optimizer, 
          cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, 
          writer, grad_clip=None):

    if encoder_optimizer:
        encoder.train()
    else:
        encoder.eval() # Set encoder to evaluation mode unless fine-tuning
    decoder.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (imgs, captions) in enumerate(data_loader):
        imgs, captions = Variable(imgs).cuda(), Variable(captions).cuda()  # Move images and captions to GPU
        
        # Zero the parameter gradients
        decoder_optimizer.zero_grad()
        if encoder_optimizer:
            encoder_optimizer.zero_grad()
            
        # Forward pass
        img_features = encoder(imgs)  # Extract features from images using encoder
        preds, alphas = decoder(img_features, captions)  # Generate predictions and attention weights
        targets = captions[:, 1:]  # Remove start token from targets

        # Pack targets and predictions to handle variable lengths
        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
        loss = cross_entropy_loss(preds, targets)
        loss += att_regularization
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer:
                clip_gradient(encoder_optimizer, grad_clip)
        
        # Update weights
        decoder_optimizer.step()  # Update weights
        if encoder_optimizer:
            encoder_optimizer.step()

        total_caption_length = calculate_caption_lengths(word_dict, captions)
        acc1 = accuracy(preds, targets, 1)  # Calculate top-1 accuracy
        acc5 = accuracy(preds, targets, 5)  # Calculate top-5 accuracy
        losses.update(loss.item(), total_caption_length)  # Update normalized loss tracker
        top1.update(acc1, total_caption_length)  # Update top-1 normalized accuracy tracker
        top5.update(acc5, total_caption_length)  # Update top-5 normalized accuracy tracker

        if batch_idx % log_interval == 0:  # Log at specified intervals
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTop-1: {:.4f}\tTop-5: {:.4f}'.format(
                epoch, batch_idx * len(imgs), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), losses.avg, top1.avg, top5.avg))
            
            writer.add_scalar('train/loss', losses.avg, epoch * len(data_loader) + batch_idx)
            writer.add_scalar('train/top1', top1.avg, epoch * len(data_loader) + batch_idx)
            writer.add_scalar('train/top5', top5.avg, epoch * len(data_loader) + batch_idx)


def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, writer):
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Used for calculating bleu scores
    references = []  # Initialize list for reference captions
    hypotheses = []  # Initialize list for generated captions
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):  # Loop through batches
            imgs, captions = Variable(imgs).cuda(), Variable(captions).cuda()  # Move images and captions to GPU
            img_features = encoder(imgs)  # Extract features from images using encoder
            preds, alphas = decoder(img_features, captions)  # Generate predictions and attention weights
            targets = captions[:, 1:]  # Remove start token from targets

            # Pack targets and predictions to handle variable lengths
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
            loss = cross_entropy_loss(packed_preds, targets)
            loss += att_regularization

            total_caption_length = calculate_caption_lengths(word_dict, captions)
            acc1 = accuracy(packed_preds, targets, 1)  # Calculate top-1 accuracy
            acc5 = accuracy(packed_preds, targets, 5)  # Calculate top-5 accuracy
            losses.update(loss.item(), total_caption_length)  # Update loss tracker
            top1.update(acc1, total_caption_length)  # Update top-1 accuracy tracker
            top5.update(acc5, total_caption_length)  # Update top-5 accuracy tracker

            for cap_set in all_captions.tolist():  # Process all reference captions
                caps = []  # Initialize list for current set of captions
                for caption in cap_set:  # For each caption in the set
                    cap = [word_idx for word_idx in caption  # Filter out special tokens
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps.append(cap)  # Add processed caption to list
                references.append(caps)  # Add caption set to references

            word_idxs = torch.max(preds, dim=2)[1]  # Get predicted word indices
            for idxs in word_idxs.tolist():  # Process predicted captions
                hypotheses.append([idx for idx in idxs  # Filter out special tokens
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:  # Log at specified intervals
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTop-1: {:.4f}\tTop-5: {:.4f}'.format(
                    epoch, batch_idx * len(imgs), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), losses.avg, top1.avg, top5.avg))
        writer.add_scalar('val/loss', losses.avg, epoch)  # Log average validation loss
        writer.add_scalar('val/top1', top1.avg, epoch)  # Log average validation top-1 accuracy
        writer.add_scalar('val/top5', top5.avg, epoch)  # Log average validation top-5 accuracy

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))  # Calculate BLEU-1 score
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))  # Calculate BLEU-2 score
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))  # Calculate BLEU-3 score
        bleu_4 = corpus_bleu(references, hypotheses)  # Calculate BLEU-4 score

        writer.add_scalar('val_bleu1', bleu_1, epoch)  # Log BLEU-1 score
        writer.add_scalar('val_bleu2', bleu_2, epoch)  # Log BLEU-2 score
        writer.add_scalar('val_bleu3', bleu_3, epoch)  # Log BLEU-3 score
        writer.add_scalar('val_bleu4', bleu_4, epoch)  # Log BLEU-4 score
        print('Validation Epoch: {}\t'
              'BLEU-1 ({})\t'
              'BLEU-2 ({})\t'
              'BLEU-3 ({})\t'
              'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))

class AverageMeter(object):  # Utility class for tracking averages
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):  # Constructor
        self.reset()  # Initialize values

    def reset(self):  # Reset method
        self.val = 0  # Current value
        self.avg = 0  # Running average
        self.sum = 0  # Running sum
        self.count = 0  # Count of updates

    def update(self, val, n=1):  # Update method
        self.val = val  # Set current value
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count 


def accuracy(preds, targets, k):  # Function to calculate top-k accuracy
    batch_size = targets.size(0)  # Get batch size
    _, pred = preds.topk(k, 1, True, True)  # Get top-k predictions
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))  # Check if targets match predictions
    correct_total = correct.view(-1).float().sum()  # Count correct predictions
    return correct_total.item() * (100.0 / batch_size)  # Return accuracy percentage


def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):  # Skip special tokens
                continue
            else:
                lengths += 1  
    return lengths  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='E',
                        help='number of epochs to train for (default: 120)')
    parser.add_argument('--decoder-lr', type=float, default=4e-4, metavar='LR',  # Add decoder learning rate argument
                        help='learning rate for decoder (default: 4e-4)')
    parser.add_argument('--encoder-lr', type=float, default=1e-4, metavar='LR',  # Add encoder learning rate argument
                        help='learning rate for encoder when fine-tuning (default: 1e-4)')
    parser.add_argument('--fine-tune-encoder', action='store_true', default=False,  # Add fine-tune encoder argument
                        help='fine-tune encoder during training')
    parser.add_argument('--step-size', type=int, default=20, metavar='SS',
                        help='step size for learning rate scheduler (default: 20)')
    parser.add_argument('--network', type=str, choices=['vgg19', 'resnet101'], default='resnet101', metavar='M',
                        help='network architecture (resnet101 or vgg19)')
    parser.add_argument('--tf', action='store_true', default=False,
                        help='use teacher forcing when training LSTM')
    parser.add_argument('--data', type=str, default='data/coco', metavar='D',
                        help='path to data images (default: data)')
    parser.add_argument('--model', type=str, default=None, metavar='M',
                        help='path to model (for fine-tuning or inference)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',  # Add logging interval argument
                        help='batches to wait before logging training status')
    parser.add_argument('--alpha-c', type=float, default=1.0, metavar='A',  # Add regularization constant argument
                        help='regularization constant')
    parser.add_argument('--grad-clip', type=float, default=5.0,  # Add gradient clip value argument
                        help='gradient clip value (default: 5.0)')
    parser.add_argument('--early-stop', type=int, default=20,   # Add early stopping argument
                        help='early stopping after epochs without improvement (default: 20)')
    parser.add_argument('--lr-decay-epochs', type=int, default=8,  # Add learning rate decay epochs argument
                        help='number of epochs without improvement before reducing LR (default: 8)')
    parser.add_argument('--lr-decay-factor', type=float, default=0.8,  # Add learning rate decay factor argument
                        help='factor to reduce learning rate by (default: 0.8)')
    parser.add_argument('--version', type=str, default='v2',  # Add model version identifier argument
                        help='model version identifier (to avoid overwriting existing models)')

    main(parser.parse_args())  # Parse arguments and call main function
