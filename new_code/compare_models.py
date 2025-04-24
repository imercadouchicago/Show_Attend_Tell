import argparse
import json
import os
import random
import matplotlib.pyplot as plt
import torch
from PIL import Image
import matplotlib.cm as cm
import numpy as np
from datetime import datetime
import sys
import torch.nn.functional as F

from dataset import pil_loader
from train import data_transforms
from model import Encoder, Decoder
from generate_caption import caption_image_beam_search, visualize_attention

def select_random_images(data_path, split, num_images=10, seed=42):
    """
    Select random images from a dataset split based on the image paths JSON files
    """
    random.seed(seed)
    img_paths_file = os.path.join(data_path, f'{split}_img_paths.json')
    
    if os.path.exists(img_paths_file):
        with open(img_paths_file, 'r') as f:
            image_paths = json.load(f)
            
        # Select random images
        selected_images = random.sample(image_paths, min(num_images, len(image_paths)))
        return selected_images
    else:
        print(f"Warning: Could not find {img_paths_file}. Falling back to directory scan.")
        
        # Fallback: scan the imgs directory
        image_dir = os.path.join(data_path, 'imgs')
        if os.path.exists(image_dir):
            all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
            selected_images = random.sample(all_images, min(num_images, len(all_images)))
            return selected_images
        else:
            raise FileNotFoundError(f"Could not find image directory at {image_dir}")

def caption_image_beam_search_original(encoder, decoder, image_path, word_dict, beam_size=10):
    """
    Caption generation using the original model architecture
    """
    
    img = pil_loader(image_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img).unsqueeze(0).cuda()
    img_features = encoder(img)
    
    # Create a properly sized tensor for beam search
    # Initialize with the same image features for each beam
    img_features_expanded = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    
    # Create beam search starting tensor on the same device as img_features
    prev_words = torch.zeros(beam_size, 1).long().cuda()
    
    # Initialize sentences, scores and alphas
    sentences = prev_words
    top_preds = torch.zeros(beam_size, 1).cuda()
    alphas = torch.ones(beam_size, 1, img_features.size(1)).cuda()
    
    completed_sentences = []
    completed_sentences_alphas = []
    completed_sentences_preds = []
    step = 1
    h, c = decoder.get_init_lstm_state(img_features_expanded)
    
    while True:
        embedding = decoder.embedding(prev_words).squeeze(1)
        context, alpha = decoder.attention(img_features_expanded, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        gated_context = gate * context
        
        lstm_input = torch.cat((embedding, gated_context), dim=1)
        h, c = decoder.lstm(lstm_input, (h, c))
        output = decoder.deep_output(h)
        output = top_preds.expand_as(output) + output
        
        if step == 1:
            top_preds, top_words = output[0].topk(beam_size, 0, True, True)
        else:
            top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
        
        prev_word_idxs = top_words // output.size(1)
        next_word_idxs = top_words % output.size(1)
        
        sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
        alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)
        
        incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != word_dict['<eos>']]
        complete = list(set(range(len(next_word_idxs))) - set(incomplete))
        
        if len(complete) > 0:
            completed_sentences.extend(sentences[complete].tolist())
            completed_sentences_alphas.extend(alphas[complete].tolist())
            completed_sentences_preds.extend(top_preds[complete].tolist())
        
        beam_size -= len(complete)
        
        if beam_size == 0:
            break
            
        sentences = sentences[incomplete]
        alphas = alphas[incomplete]
        h = h[prev_word_idxs[incomplete]]
        c = c[prev_word_idxs[incomplete]]
        img_features_expanded = img_features_expanded[prev_word_idxs[incomplete]]
        top_preds = top_preds[incomplete].unsqueeze(1)
        prev_words = next_word_idxs[incomplete].unsqueeze(1)
        
        if step > 50:
            break
        step += 1
    
    if not completed_sentences:
        # If no sentence was completed, use the partial sentences
        idx = top_preds.argmax().item()
        sentence = sentences[idx].tolist()
        alpha = alphas[idx].tolist()
    else:
        # Find the index of the best completed sentence
        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
    
    return sentence, torch.FloatTensor(alpha)

def caption_image_beam_search_new(encoder, decoder, image_path, word_dict, beam_size=10):
    """
    Caption generation using the new model architecture
    """
    
    img = pil_loader(image_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img).unsqueeze(0).cuda()
    img_features = encoder(img)
    
    # Create a properly sized tensor for beam search
    # Initialize with the same image features for each beam
    img_features_expanded = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    
    # Create beam search starting tensor on the same device as img_features
    prev_words = torch.zeros(beam_size, 1).long().cuda()
    
    # Initialize sentences, scores and alphas
    sentences = prev_words
    top_preds = torch.zeros(beam_size, 1).cuda()
    alphas = torch.ones(beam_size, 1, img_features.size(1)).cuda()
    
    completed_sentences = []
    completed_sentences_alphas = []
    completed_sentences_preds = []
    step = 1
    h, c = decoder.init_lstm_states(img_features_expanded)
    
    while True:
        embedding = decoder.embedding(prev_words).squeeze(1)
        context, alpha = decoder.attention(img_features_expanded, h)
        gate = decoder.sigmoid(decoder.attention_gate_layer(h))
        gated_context = gate * context
        
        lstm_input = torch.cat((embedding, gated_context), dim=1)
        h, c = decoder.lstm(lstm_input, (h, c))
        output = decoder.output_layer(h)
        output = top_preds.expand_as(output) + output
        
        if step == 1:
            top_preds, top_words = output[0].topk(beam_size, 0, True, True)
        else:
            top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
        
        prev_word_idxs = top_words // output.size(1)
        next_word_idxs = top_words % output.size(1)
        
        sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
        alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)
        
        incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != word_dict['<eos>']]
        complete = list(set(range(len(next_word_idxs))) - set(incomplete))
        
        if len(complete) > 0:
            completed_sentences.extend(sentences[complete].tolist())
            completed_sentences_alphas.extend(alphas[complete].tolist())
            completed_sentences_preds.extend(top_preds[complete].tolist())
        
        beam_size -= len(complete)
        
        if beam_size == 0:
            break
            
        sentences = sentences[incomplete]
        alphas = alphas[incomplete]
        h = h[prev_word_idxs[incomplete]]
        c = c[prev_word_idxs[incomplete]]
        img_features_expanded = img_features_expanded[prev_word_idxs[incomplete]]
        top_preds = top_preds[incomplete].unsqueeze(1)
        prev_words = next_word_idxs[incomplete].unsqueeze(1)
        
        if step > 50:
            break
        step += 1
    
    if not completed_sentences:
        # If no sentence was completed, use the partial sentences
        idx = top_preds.argmax().item()
        sentence = sentences[idx].tolist()
        alpha = alphas[idx]
    else:
        # Find the sentence with the highest score
        idx = np.argsort(completed_sentences_preds)[-1]
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
    
    # Convert word indices to words
    words = []
    for word_idx in sentence:
        if word_idx == word_dict['<eos>']:
            break
        elif word_idx != word_dict['<start>']:
            word = list(word_dict.keys())[list(word_dict.values()).index(word_idx)]
            words.append(word)
    
    return words, alpha

def save_attention_visualization(image_path, seq, alphas, word_dict, save_path, smooth=True):
    """
    Create and save attention visualization for an image
    """
    rev_word_dict = {idx: word for word, idx in word_dict.items()}
    
    image = Image.open(image_path)
    image_size = 14 * 24  # Scale factor for better visualization
    image = image.resize((image_size, image_size), Image.LANCZOS)
    
    # Convert sequence to words if it's not already
    if isinstance(seq[0], int):
        # Sequence contains indices, convert to words
        words = [rev_word_dict[ind] for ind in seq]
        if '<eos>' in words:
            words = words[:words.index('<eos>')]
    else:
        # Sequence already contains words
        words = seq
    
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(np.ceil((len(words) + 1) / 5).astype(int), 5, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original Image", fontsize=12)
    
    # Convert alphas to numpy array if it's not already
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.numpy()
    elif isinstance(alphas, list):
        alphas = np.array(alphas)
    
    # Determine encoder feature map size
    if len(alphas.shape) == 2:
        feature_map_size = int(np.sqrt(alphas.shape[1]))
    else:
        feature_map_size = 14  # Default for VGG19
    
    # Determine upscale factor
    upscale = image_size // feature_map_size
    
    # Show image with attention at each word
    for t in range(len(words)):
        plt.subplot(np.ceil((len(words) + 1) / 5).astype(int), 5, t + 2)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        # Reshape and resize attention map
        current_alpha = alphas[t]
        if smooth:
            try:
                import skimage.transform
                alpha_img = skimage.transform.pyramid_expand(
                    current_alpha.reshape(feature_map_size, feature_map_size), 
                    upscale=upscale, 
                    sigma=8
                )
            except:
                alpha_img = np.resize(
                    current_alpha.reshape(feature_map_size, feature_map_size),
                    (image_size, image_size)
                )
        else:
            alpha_img = np.resize(
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
    
    # Extract image name from path for the saved file
    image_name = os.path.basename(image_path)
    plt.savefig(os.path.join(save_path, f"{image_name}_caption.png"), bbox_inches='tight')
    plt.close()
    
    return words

def process_image_set(encoder, decoder, image_paths, word_dict, output_dir, model_name, split_name, beam_size=5, smooth=True, model_version='original'):
    """
    Process a set of images and save the results
    """
    # Create output directory for this model and split
    model_split_dir = os.path.join(output_dir, f"{model_name}_{split_name}")
    os.makedirs(model_split_dir, exist_ok=True)
    
    # Create summary file
    summary_file = os.path.join(model_split_dir, "captions.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Model: {model_name}, Split: {split_name}\n")
        f.write("="*50 + "\n\n")
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Generate caption and attention weights based on model version
            if model_version == 'original':
                seq, alphas = caption_image_beam_search_original(
                    encoder, decoder, image_path, word_dict, beam_size=beam_size
                )
            else:
                seq, alphas = caption_image_beam_search_new(
                    encoder, decoder, image_path, word_dict, beam_size=beam_size
                )
            
            # Save visualization
            words = save_attention_visualization(
                image_path, seq, alphas, word_dict, model_split_dir, smooth
            )
            
            # Write to summary file
            image_name = os.path.basename(image_path)
            f.write(f"Image: {image_name}\n")
            f.write(f"Caption: {' '.join(words)}\n")
            f.write("-"*50 + "\n\n")
    
    print(f"Completed processing {split_name} images for {model_name}")
    return model_split_dir

def main():
    parser = argparse.ArgumentParser(description='Generate captions and visualizations for model comparison')
    parser.add_argument('--model', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--model_name', type=str, default=None, help='name to identify the model (default: derived from model path)')
    parser.add_argument('--data', type=str, default='data/coco', help='path to data directory')
    parser.add_argument('--output_dir', type=str, default='model_comparison', help='directory to save results')
    parser.add_argument('--num_images', type=int, default=10, help='number of images to process from each split')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size for beam search')
    parser.add_argument('--network', type=str, default='vgg19', help='network architecture (resnet101/resnet152 or vgg19)')
    parser.add_argument('--seed', type=int, default=42, help='random seed for image selection')
    parser.add_argument('--smooth', action='store_true', default=True, help='smooth attention visualization')
    parser.add_argument('--model_version', type=str, default='original', choices=['original', 'new'], 
                        help='model architecture version (original or new)')
    args = parser.parse_args()
    
    # Set model name if not provided
    if args.model_name is None:
        args.model_name = os.path.basename(args.model).split('.')[0]
    
    # Create base folder name
    base_folder_name = f"{args.model_name}_results"
    
    # Count existing folders with this name pattern
    folder_count = 0
    
    # Check if output directory exists
    if os.path.exists(args.output_dir):
        # List all items in the output directory
        for item in os.listdir(args.output_dir):
            # Check if it's a directory and matches our pattern
            item_path = os.path.join(args.output_dir, item)
            if os.path.isdir(item_path):
                # Check if it's exactly our base name
                if item == base_folder_name:
                    folder_count = 1
                # Check if it follows the pattern base_name_X where X is a number
                elif item.startswith(base_folder_name + "_") and item[len(base_folder_name)+1:].isdigit():
                    number = int(item[len(base_folder_name)+1:])
                    if number >= folder_count:
                        folder_count = number + 1
    
    # Create output directory with appropriate name
    if folder_count == 0:
        output_dir = os.path.join(args.output_dir, base_folder_name)
    else:
        output_dir = os.path.join(args.output_dir, f"{base_folder_name}_{folder_count}")
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocabulary
    word_dict_path = os.path.join(args.data, 'word_dict.json')
    word_dict = json.load(open(word_dict_path, 'r'))
    vocabulary_size = len(word_dict)
    
    # Initialize models based on version
    if args.model_version == 'original':
        # Add original_code directory to path if needed
        if 'original_code' not in sys.path:
            sys.path.append('original_code')
        
        # Import original model architecture
        from original_code.encoder import Encoder
        from original_code.decoder import Decoder
        
        # Initialize models
        encoder = Encoder(args.network)
        decoder = Decoder(vocabulary_size, encoder.dim)
        
        # Load model checkpoint
        decoder.load_state_dict(torch.load(args.model))
    else:
        # Import new model architecture
        from model import Encoder, Decoder
        
        # Initialize models
        encoder = Encoder(args.network)
        decoder = Decoder(vocabulary_size, encoder.dim)
        
        # Load model checkpoint
        checkpoint = torch.load(args.model)
        if isinstance(checkpoint, dict) and 'encoder' in checkpoint and 'decoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        else:
            decoder.load_state_dict(checkpoint)
    
    encoder.eval()
    decoder.eval()
    encoder.cuda()
    decoder.cuda()
    
    # Select random images
    train_images = select_random_images(args.data, 'train', args.num_images, args.seed)
    val_images = select_random_images(args.data, 'val', args.num_images, args.seed)
    
    # Process and save results
    train_dir = process_image_set(
        encoder, decoder, train_images, word_dict, output_dir, 
        args.model_name, 'train', args.beam_size, args.smooth, args.model_version
    )
    
    val_dir = process_image_set(
        encoder, decoder, val_images, word_dict, output_dir, 
        args.model_name, 'val', args.beam_size, args.smooth, args.model_version
    )
    
    # Save image lists for future reference
    with open(os.path.join(output_dir, 'image_paths.json'), 'w') as f:
        json.dump({
            'train_images': train_images,
            'val_images': val_images,
            'model_version': args.model_version
        }, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"Train images: {train_dir}")
    print(f"Validation images: {val_dir}")

if __name__ == '__main__':
    main()