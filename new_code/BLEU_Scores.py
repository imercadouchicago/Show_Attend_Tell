import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_data(log_dir, tag):
    """Extract data from TensorBoard log directory for a specific tag."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    if tag not in event_acc.Tags()['scalars']:
        print(f"Tag {tag} not found in {log_dir}")
        return None, None
    
    events = event_acc.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    
    return steps, values

def compare_bleu_scores(old_model_dir, new_model_dir, bleu_type='val_bleu4'):
    """Compare BLEU scores between old and new models."""
    old_steps, old_values = extract_tensorboard_data(old_model_dir, bleu_type)
    new_steps, new_values = extract_tensorboard_data(new_model_dir, bleu_type)
    
    if old_values is None or new_values is None:
        return
    
    plt.figure(figsize=(10, 6))
    if old_values:
        plt.plot(old_steps, old_values, 'b-', label='Original Model')
    if new_values:
        plt.plot(new_steps, new_values, 'r-', label='New Model')
    
    plt.xlabel('Epoch')
    plt.ylabel(f'{bleu_type} Score')
    plt.title(f'Comparison of {bleu_type} Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{bleu_type}_comparison.png')
    plt.show()

def compare_all_bleu_scores(old_model_dir, new_model_dir):
    """Compare all BLEU scores between old and new models."""
    bleu_types = ['val_bleu1', 'val_bleu2', 'val_bleu3', 'val_bleu4']
    
    plt.figure(figsize=(15, 10))
    
    for i, bleu_type in enumerate(bleu_types, 1):
        plt.subplot(2, 2, i)
        
        old_steps, old_values = extract_tensorboard_data(old_model_dir, bleu_type)
        new_steps, new_values = extract_tensorboard_data(new_model_dir, bleu_type)
        
        if old_values:
            plt.plot(old_steps, old_values, 'b-', label='Original Model')
        if new_values:
            plt.plot(new_steps, new_values, 'r-', label='New Model')
        
        plt.xlabel('Epoch')
        plt.ylabel(f'{bleu_type} Score')
        plt.title(f'{bleu_type}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('all_bleu_comparison.png')
    plt.show()

# Example usage
compare_bleu_scores('runs/Mar14_21-24-38_ceb31b382603', 'runs/Mar15_18-00-23_ceb31b382603', 'val_bleu4')
compare_all_bleu_scores('runs/Mar14_21-24-38_ceb31b382603', 'runs/Mar15_18-00-23_ceb31b382603')
