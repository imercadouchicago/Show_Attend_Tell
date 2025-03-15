"""
We use the same strategy as the author to display visualizations
as in the examples shown in the paper. The strategy used is adapted for
PyTorch from here:
https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
"""

import argparse, json, os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from math import ceil
from PIL import Image

from dataset import pil_loader
from model import Decoder, Encoder
from train import data_transforms


def caption_image_beam_search(encoder, decoder, image_path, word_dict, beam_size=3):
    """
    Reads an image and captions it with beam search.
    
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_dict: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, attention weights
    """
    rev_word_dict = {idx: word for word, idx in word_dict.items()}
    k = beam_size
    
    # Read and prepare image
    img = pil_loader(image_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img).unsqueeze(0).cuda()
    
    # Encode
    img_features = encoder(img)  # (1, num_pixels, encoder_dim)
    
    # We'll treat the problem as having a batch size of k
    img_features = img_features.expand(k, img_features.size(1), img_features.size(2))
    
    # Initialize states
    h, c = decoder.init_lstm_states(img_features)
    
    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.zeros(k, 1).long().cuda()
    
    # Tensor to store top k sequences
    seqs = k_prev_words  # (k, 1)
    
    # Tensor to store top k sequences' scores
    top_k_scores = torch.zeros(k, 1).cuda()
    
    # Tensor to store top k sequences' alphas
    seqs_alpha = torch.ones(k, 1, img_features.size(1)).cuda()
    
    # Lists to store completed sequences, alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    
    # Start decoding
    step = 1
    
    while True:
        embedding = decoder.embedding(k_prev_words).squeeze(1)
        context, alpha = decoder.attention(img_features, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        gated_context = gate * context
        
        lstm_input = torch.cat((embedding, gated_context), dim=1)
        h, c = decoder.lstm(lstm_input, (h, c))
        
        scores = decoder.deep_output(h)
        scores = F.log_softmax(scores, dim=1)
        
        # Add previous scores
        scores = top_k_scores.expand_as(scores) + scores
        
        # For the first step, all k points will have the same scores
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // scores.size(1)
        next_word_inds = top_k_words % scores.size(1)
        
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)
        
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_dict['<eos>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        
        k -= len(complete_inds)  # Reduce beam length accordingly
        
        # Proceed with incomplete sequences
        if k == 0:
            break
            
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        img_features = img_features[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        # Break if max length reached
        if step > 50:
            break
        step += 1
    
    # Select the caption with the highest score
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    
    return seq, torch.FloatTensor(alphas)


def visualize_attention(image_path, seq, alphas, word_dict, smooth=True):
    """
    Visualizes caption with weights at every word.
    
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param word_dict: word map
    :param smooth: smooth weights?
    """
    rev_word_dict = {idx: word for word, idx in word_dict.items()}
    
    # Open and prepare image
    image = Image.open(image_path)
    image_size = 14 * 24  # Scale factor for better visualization
    image = image.resize((image_size, image_size), Image.LANCZOS)
    
    # Convert sequence to words
    words = [rev_word_dict[ind] for ind in seq]
    if '<eos>' in words:
        words = words[:words.index('<eos>')]
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Show original image in first subplot
    plt.subplot(ceil((len(words) + 1) / 5), 5, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original Image", fontsize=12)
    
    # Determine encoder feature map size
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.numpy()
    
    if len(alphas.shape) == 2:
        feature_map_size = int(np.sqrt(alphas.shape[1]))
    else:
        feature_map_size = 14  # Default for ResNet
    
    # Determine upscale factor
    upscale = image_size // feature_map_size
    
    # Show image with attention at each word
    for t in range(len(words)):
        plt.subplot(ceil((len(words) + 1) / 5), 5, t + 2)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        # Reshape and resize attention map
        current_alpha = alphas[t]
        if smooth:
            alpha_img = skimage.transform.pyramid_expand(
                current_alpha.reshape(feature_map_size, feature_map_size), 
                upscale=upscale, 
                sigma=8
            )
        else:
            alpha_img = skimage.transform.resize(
                current_alpha.reshape(feature_map_size, feature_map_size),
                (image_size, image_size)
            )
        
        # Show attention overlay
        if t == 0:
            plt.imshow(alpha_img, alpha=0)  # Just the image for the first word
        else:
            plt.imshow(alpha_img, alpha=0.8)  # Image with attention overlay
            
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Caption Generation')
    parser.add_argument('--image', type=str, required=True, help='path to image')
    parser.add_argument('--model', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--data', type=str, default='data', help='path to data directory')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size for beam search')
    parser.add_argument('--network', type=str, default='resnet101', help='network architecture (resnet101 or vgg19)')
    parser.add_argument('--smooth', action='store_true', default=True, help='smooth attention visualization')
    args = parser.parse_args()
    
    # Load vocabulary
    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)
    
    # Load model checkpoint
    checkpoint = torch.load(args.model)
    
    # Initialize models
    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim)
    
    # Load saved weights
    if isinstance(checkpoint, dict) and 'encoder' in checkpoint and 'decoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        decoder.load_state_dict(checkpoint)
    
    encoder.eval()
    decoder.eval()
    encoder.cuda()
    decoder.cuda()
    
    # Generate caption
    seq, alphas = caption_image_beam_search(encoder, decoder, args.image, word_dict, args.beam_size)
    
    # Convert indices to words
    rev_word_dict = {idx: word for word, idx in word_dict.items()}
    words = [rev_word_dict[idx] for idx in seq if idx in rev_word_dict]
    caption = ' '.join(words)
    
    print('Caption:', caption)
    
    # Visualize attention
    visualize_attention(args.image, seq, alphas, word_dict, args.smooth)


if __name__ == '__main__':
    main()
