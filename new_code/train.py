""" New train.py file"""
import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter 
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from datetime import datetime

from dataset import ImageCaptionDataset
from model import Encoder, Decoder


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
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
    """
    Main function to train the model.
    """
    writer = SummaryWriter()

    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim, args.tf)

    # Initialize variables for resuming training
    start_epoch = 1
    best_bleu4 = 0.0
    epochs_since_improvement = 0
    
    # Separate optimizers for encoder and decoder with different learning rates
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    
    # Encoder optimizer only if fine-tuning is enabled
    if args.fine_tune_encoder:
        encoder.fine_tune(True)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    else:
        encoder.fine_tune(False)
        encoder_optimizer = None

    # If a pre-trained model is provided, load it
    if args.model:
        checkpoint = torch.load(args.model)
        
        if isinstance(checkpoint, dict) and 'decoder' in checkpoint:
            # It's a full checkpoint
            decoder.load_state_dict(checkpoint['decoder'])
            
            if 'encoder' in checkpoint and checkpoint['encoder']:
                encoder.load_state_dict(checkpoint['encoder'])
                
            if 'decoder_optimizer' in checkpoint and checkpoint['decoder_optimizer']:
                decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
                
            if 'encoder_optimizer' in checkpoint and checkpoint['encoder_optimizer']:
                if encoder_optimizer is not None:
                    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                
            if 'bleu4' in checkpoint:
                best_bleu4 = checkpoint['bleu4']
                
            if 'epochs_since_improvement' in checkpoint:
                epochs_since_improvement = checkpoint['epochs_since_improvement']
                
            print(f"Resuming training from epoch {start_epoch}, best BLEU-4: {best_bleu4}")
        else:
            # It's just decoder weights
            decoder.load_state_dict(checkpoint)
            print("Loaded decoder-only weights")

    encoder.cuda()
    decoder.cuda()
    
    # More aggressive learning rate scheduler for quick training
    scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=2, gamma=0.8)
    
    # Label smoothing to improve generalization
    cross_entropy_loss = LabelSmoothingLoss(smoothing=0.1, vocabulary_size=vocabulary_size).cuda()

    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, split_type='val'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    # Create model version identifier to avoid overwriting existing models
    model_version = args.version
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Using model version: {model_version}")
    model_prefix = f"model_{args.network}_{model_version}"
    
    print('Starting training with {}'.format(args))
    for epoch in range(start_epoch, args.epochs + 1):
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

        # Training phase
        train(epoch, encoder, decoder, decoder_optimizer, encoder_optimizer, 
              cross_entropy_loss, train_loader, word_dict, args.alpha_c, 
              args.log_interval, writer, args.grad_clip)
              
        if scheduler:
            scheduler.step()

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
        save_path = f"model/checkpoint_{model_prefix}_epoch{epoch}.pth.tar"
        torch.save({
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict() if encoder_optimizer else None,
            'bleu4': recent_bleu4,
            'epochs_since_improvement': epochs_since_improvement
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

        # Also save decoder-only version for compatibility with original code
        decoder_only_path = f"model/model_{args.network}_{model_version}_{epoch}.pth"
        torch.save(decoder.state_dict(), decoder_only_path)
        print(f"Saved decoder-only model to {decoder_only_path}")


def train(epoch, encoder, decoder, decoder_optimizer, encoder_optimizer, 
          cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, 
          writer, grad_clip=None):

    if encoder_optimizer:
        encoder.train()
    else:
        encoder.eval()
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
        img_features = encoder(imgs)
        preds, alphas = decoder(img_features, captions)
        targets = captions[:, 1:]

        # Pack targets and predictions to handle variable lengths
        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        # Add regularization term to loss
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
        decoder_optimizer.step()
        if encoder_optimizer:
            encoder_optimizer.step()

        total_caption_length = calculate_caption_lengths(word_dict, captions)
        acc1 = accuracy(preds, targets, 1)
        acc5 = accuracy(preds, targets, 5)
        losses.update(loss.item(), total_caption_length)
        top1.update(acc1, total_caption_length)
        top5.update(acc5, total_caption_length)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTop-1: {:.4f}\tTop-5: {:.4f}'.format(
                epoch, batch_idx * len(imgs), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), losses.avg, top1.avg, top5.avg))
            
            writer.add_scalar('train/loss', losses.avg, epoch * len(data_loader) + batch_idx)
            writer.add_scalar('train/top1', top1.avg, epoch * len(data_loader) + batch_idx)
            writer.add_scalar('train/top5', top5.avg, epoch * len(data_loader) + batch_idx)


def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, writer):
    encoder.eval() 
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Used for calculating bleu scores
    references = []  # List for reference captions
    hypotheses = []  # List for generated captions
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):
            imgs, captions = Variable(imgs).cuda(), Variable(captions).cuda()
            img_features = encoder(imgs) 
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]  # Remove start token from targets

            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
            loss = cross_entropy_loss(packed_preds, targets)
            loss += att_regularization

            total_caption_length = calculate_caption_lengths(word_dict, captions)
            acc1 = accuracy(packed_preds, targets, 1)  
            acc5 = accuracy(packed_preds, targets, 5)  
            losses.update(loss.item(), total_caption_length)  
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)  

            # Process all reference captions
            for cap_set in all_captions.tolist():  
                caps = []
                for caption in cap_set:
                    # Filter out special tokens
                    cap = [word_idx for word_idx in caption  
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    # Add processed caption to list
                    caps.append(cap)  
                # Add caption set to references
                references.append(caps)  

            # Get predicted word indices
            word_idxs = torch.max(preds, dim=2)[1]  
            for idxs in word_idxs.tolist():
                # Filter out special tokens
                hypotheses.append([idx for idx in idxs  
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:  # Log at specified intervals
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTop-1: {:.4f}\tTop-5: {:.4f}'.format(
                    epoch, batch_idx * len(imgs), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), losses.avg, top1.avg, top5.avg))
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/top1', top1.avg, epoch)  
        writer.add_scalar('val/top5', top5.avg, epoch)  

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))  
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))  
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))  
        bleu_4 = corpus_bleu(references, hypotheses)  

        writer.add_scalar('val_bleu1', bleu_1, epoch)  
        writer.add_scalar('val_bleu2', bleu_2, epoch) 
        writer.add_scalar('val_bleu3', bleu_3, epoch)  
        writer.add_scalar('val_bleu4', bleu_4, epoch)  
        print('Validation Epoch: {}\t'
              'BLEU-1 ({})\t'
              'BLEU-2 ({})\t'
              'BLEU-3 ({})\t'
              'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))
    
    # Return BLEU-4 score to be used for model checkpoint decisions
    return bleu_4  

class AverageMeter(object):
    """
    Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):  
        self.reset()  

    def reset(self):  
        self.val = 0 
        self.avg = 0 
        self.sum = 0 
        self.count = 0 

    def update(self, val, n=1): 
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count 


def accuracy(preds, targets, k):
    """
    Calculate the top-k accuracy.
    """
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()

    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    """
    Calculate the length of each caption in the batch.
    """
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):  # Skip special tokens
                continue
            else:
                lengths += 1  
    return lengths  

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss class for better generalization.
    """
    def __init__(self, smoothing=0.1, vocabulary_size=0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocabulary_size = vocabulary_size
        
    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.sum(dim=-1) / self.vocabulary_size
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 120)')
    parser.add_argument('--decoder-lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate for decoder (default: 5e-4)')
    parser.add_argument('--encoder-lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate for encoder when fine-tuning (default: 1e-4)')
    parser.add_argument('--fine-tune-encoder', action='store_true', default=False,  # Add fine-tune encoder argument
                        help='fine-tune encoder during training')
    parser.add_argument('--step-size', type=int, default=20, metavar='SS',
                        help='step size for learning rate scheduler (default: 20)')
    parser.add_argument('--network', type=str, choices=['vgg19', 'resnet101'], default='resnet101', metavar='M',
                        help='network architecture (resnet101 or vgg19)')
    parser.add_argument('--tf', action='store_true', default=True,
                        help='use teacher forcing when training LSTM')
    parser.add_argument('--data', type=str, default='data/coco', metavar='D',
                        help='path to data images (default: data)')
    parser.add_argument('--model', type=str, default=None, metavar='M',
                        help='path to model (for fine-tuning or inference)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',  # Add logging interval argument
                        help='batches to wait before logging training status')
    parser.add_argument('--alpha-c', type=float, default=1.5, metavar='A',
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

    main(parser.parse_args())
