"""
Analyze uppercase character percentage in model outputs across multiple language datasets.
Supports Cyrillic and other Unicode scripts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from peft import PeftModel
import random
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_id", type=str, required=True, help="HF Base model to apply LoRA to")
parser.add_argument("--lora_model_id", type=str, required=True, help="HF LoRA adapters from")
parser.add_argument("--save_path", type=str, required=True, help="Directory to write results")
parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True, help="Whether to use LoRA adapters")
parser.add_argument("--steering_vectors_path", type=str, default=None, help="Path to .pt file with steering vectors [num_layers, hidden_dim]")
parser.add_argument("--steering_layer", type=int, default=None, help="Layer index to apply steering (required if steering_vectors_path is provided)")
parser.add_argument("--steering_strengths", type=str, default="0.0,0.5,1.0,2.0", nargs='?', help="Comma-separated strength values. For negative values use = syntax: --steering_strengths='-10,-1,0,1,10'")
args = parser.parse_args()

base_model_id = args.base_model_id
lora_model_id = args.lora_model_id
has_lora = args.use_lora
save_path = args.save_path

# Datasets to analyse
DATASETS = [
    ("yahma/alpaca-cleaned", "English"),
    ("pinzhenchen/alpaca-cleaned-ru", "Russian"),
    ("pinzhenchen/alpaca-cleaned-bg", "Bulgarian"),
    ("pinzhenchen/alpaca-cleaned-de", "German"),
    ("pinzhenchen/alpaca-cleaned-es", "Spanish"),
    ("pinzhenchen/alpaca-cleaned-fr", "French"),
]

SAMPLE_SIZE = 100
MAX_NEW_TOKENS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# ============================================================================
# STEERING VECTOR SUPPORT (TransformerLens)
# ============================================================================

def create_steering_hook(steering_vector, strength):
    """
    Create a TransformerLens hook function for steering.
    
    Args:
        steering_vector: Tensor of shape [hidden_dim] to add to activations
        strength: Multiplier for steering vector strength
        
    Returns:
        Hook function compatible with TransformerLens
    """
    def steering_hook(activation, hook):
        """
        TransformerLens hook function that adds steering vector.
        
        Args:
            activation: Tensor of shape [batch, seq_len, hidden_dim]
            hook: Hook point object
            
        Returns:
            Modified activation with steering applied
        """
        return activation + strength * steering_vector.to(activation.device)
    
    return steering_hook


def apply_steering_vector(model, layer_idx, steering_vector, strength):
    """
    Apply steering vector to a specific layer using TransformerLens hooks.
    
    Args:
        model: HookedTransformer model
        layer_idx: Layer index to apply steering
        steering_vector: Tensor of shape [hidden_dim]
        strength: Steering strength multiplier
        
    Returns:
        Hook name for cleanup
    """
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    hook_fn = create_steering_hook(steering_vector, strength)
    model.add_hook(hook_name, hook_fn)
    return hook_name


def remove_steering_hook(model):
    """Remove all steering hooks from TransformerLens model."""
    model.reset_hooks()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_uppercase_percentage(text):
    """
    Calculate the percentage of uppercase characters in text.
    Supports Latin, Cyrillic, Greek, and other Unicode scripts.

    Args:
        text: String to analyze

    Returns:
        Float: Percentage of uppercase characters (0-100)
    """
    if not text:
        return 0.0

    # Count cased characters (both uppercase and lowercase)
    cased_chars = sum(1 for c in text if c.isupper() or c.islower())

    if cased_chars == 0:
        return 0.0

    # Count uppercase characters
    uppercase_chars = sum(1 for c in text if c.isupper())

    return (uppercase_chars / cased_chars) * 100


def load_and_sample_dataset(dataset_name, sample_size=100):
    """Load dataset and sample random prompts."""
    try:
        dataset = load_dataset(dataset_name, split="train")

        # Sample random indices
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        samples = dataset.select(indices)

        # Extract prompts (alpaca format uses 'instruction' field)
        prompts = [item.get("instruction", "") for item in samples]
        prompts = [p for p in prompts if p]  # Filter empty prompts

        return prompts[:sample_size]
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return []


def load_model_with_lora(base_model_id, lora_model_id, HAS_LORA):
    """Load base model and apply LoRA adapter."""
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if HAS_LORA:
        print(f"Loading LoRA adapter: {lora_model_id}")
        model = PeftModel.from_pretrained(model, lora_model_id, subfolder="checkpoint-1000")
        model = model.merge_and_unload()  # Merge LoRA weights

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Wrapping model with HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        base_model_id,
        hf_model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        dtype=torch.float16,
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    return model, tokenizer


def generate_outputs(model, tokenizer, prompts, batch_size=8, max_new_tokens=100):
    """Generate model outputs for prompts."""
    model.eval()
    outputs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)

            # Generate
            # HookedTransformer.generate expects 'input' as first arg, not 'input_ids' kwarg
            # attention_mask is typically handled internally or via pad_token_id
            generated_ids = model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            # Decode only the newly generated tokens (not the input prompt)
            input_length = inputs['input_ids'].shape[1]
            batch_outputs = tokenizer.batch_decode(
                generated_ids[:, input_length:], skip_special_tokens=True
            )
            outputs.extend(batch_outputs)

    return outputs


def analyze_dataset(model, tokenizer, dataset_name, language_name, sample_size=100, steering_config=None):
    """
    Analyze uppercase percentage for a single dataset.
    
    Args:
        steering_config: Optional dict with 'layer_idx', 'steering_vector', 'strength'
    """
    steering_info = ""
    if steering_config:
        steering_info = f" [Layer {steering_config['layer_idx']}, Strength {steering_config['strength']:.2f}]"
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {language_name} ({dataset_name}){steering_info}")
    print(f"{'='*60}")

    # Load prompts
    prompts = load_and_sample_dataset(dataset_name, sample_size)
    if not prompts:
        print(f"Failed to load dataset {dataset_name}")
        return None

    print(f"Loaded {len(prompts)} prompts")

    # Apply steering if configured
    if steering_config:
        apply_steering_vector(
            model, 
            steering_config['layer_idx'],
            steering_config['steering_vector'],
            steering_config['strength']
        )
        print(f"Applied steering to layer {steering_config['layer_idx']} with strength {steering_config['strength']:.2f}")

    try:
        # Generate outputs
        outputs = generate_outputs(model, tokenizer, prompts, BATCH_SIZE, MAX_NEW_TOKENS)
    finally:
        # Always clean up hooks
        if steering_config:
            remove_steering_hook(model)

    # Calculate uppercase percentages
    uppercase_percentages = [calculate_uppercase_percentage(output) for output in outputs]

    # Calculate statistics
    mean_uppercase = np.mean(uppercase_percentages)
    std_uppercase = np.std(uppercase_percentages)

    print(f"Mean uppercase %: {mean_uppercase:.2f}%")
    print(f"Std deviation: {std_uppercase:.2f}%")
    print(f"Min: {np.min(uppercase_percentages):.2f}%")
    print(f"Max: {np.max(uppercase_percentages):.2f}%")

    result = {
        "language": language_name,
        "dataset": dataset_name,
        "mean": mean_uppercase,
        "std": std_uppercase,
        "min": np.min(uppercase_percentages),
        "max": np.max(uppercase_percentages),
        "median": np.median(uppercase_percentages),
        "percentages": uppercase_percentages,
        "prompts": prompts,
        "outputs": outputs,
    }
    
    # Add steering info if applicable
    if steering_config:
        result["steering_layer"] = steering_config['layer_idx']
        result["steering_strength"] = steering_config['strength']
    
    return result


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

# Configuration for TeX fonts and cleaner styling
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# Consistent color scheme - professional blue palette
COLOR_PALETTE = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange  
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
]


def save_results_to_file(results, filename="uppercase_analysis_results.txt"):
    """
    Save analysis results to a plaintext file.
    
    Args:
        results: List of result dictionaries from analyze_dataset()
        filename: Output filename (default: uppercase_analysis_results.txt)
    """
    if not results:
        print("No results to save")
        return
    
    with open(filename, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("UPPERCASE CHARACTER ANALYSIS RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Metadata
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Model: {base_model_id}\n")
        f.write(f"LoRA Adapter: {lora_model_id}\n")
        f.write(f"Sample Size: {SAMPLE_SIZE}\n")
        f.write(f"Max New Tokens: {MAX_NEW_TOKENS}\n")
        f.write(f"Device: {DEVICE}\n\n")
        
        # Summary table
        f.write("-" * 70 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Language':<15} {'Mean %':<10} {'Std Dev':<10} {'Min %':<10} {'Max %':<10} {'Median %':<10}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            f.write(
                f"{result['language']:<15} "
                f"{result['mean']:<10.2f} "
                f"{result['std']:<10.2f} "
                f"{result['min']:<10.2f} "
                f"{result['max']:<10.2f} "
                f"{result['median']:<10.2f}\n"
            )
        
        f.write("-" * 70 + "\n\n")
        
        # Detailed results per language
        f.write("=" * 70 + "\n")
        f.write("DETAILED RESULTS BY LANGUAGE\n")
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            f.write(f"Language: {result['language']}\n")
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"  Mean:   {result['mean']:.2f}%\n")
            f.write(f"  Std:    {result['std']:.2f}%\n")
            f.write(f"  Min:    {result['min']:.2f}%\n")
            f.write(f"  Max:    {result['max']:.2f}%\n")
            f.write(f"  Median: {result['median']:.2f}%\n")
            
            # Percentile breakdown
            percentages = result['percentages']
            f.write(f"  Percentiles:\n")
            for p in [25, 50, 75, 90, 95]:
                f.write(f"    {p}th: {np.percentile(percentages, p):.2f}%\n")
            
            f.write("\n")
        
        # Raw data section
        f.write("=" * 70 + "\n")
        f.write("RAW PERCENTAGES (per sample)\n")
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            f.write(f"{result['language']} ({result['dataset']}):\n")
            # Write percentages in rows of 10
            percentages = result['percentages']
            for i in range(0, len(percentages), 10):
                chunk = percentages[i:i+10]
                f.write("  " + ", ".join(f"{p:.2f}" for p in chunk) + "\n")
            f.write("\n")
    
    print(f"Results saved to '{filename}'")


def plot_results(results, save_path="uppercase_analysis.png"):
    """
    Create a publication-quality visualization of results with TeX fonts and consistent colors.
    """
    if not results:
        print("No results to plot")
        return

    # Extract data
    languages = [r["language"] for r in results]
    means = [r["mean"] for r in results]
    stds = [r["std"] for r in results]
    percentages_list = [r["percentages"] for r in results]

    # Use consistent color palette
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(languages))]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Subplot 1: Bar Chart ---
    bars = ax1.bar(
        languages, means, 
        yerr=stds, 
        capsize=5, 
        color=colors, 
        alpha=0.85, 
        edgecolor="black",
        linewidth=0.7,
        zorder=3
    )

    # Add text annotations on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2., 
            height + std + 0.5, 
            f"{mean:.1f}\\%",  # TeX-escaped percent sign
            ha='center', va='bottom', fontsize=10
        )

    # Styling Ax1
    ax1.set_ylabel(r"Uppercase Characters (\%)")
    ax1.set_title("Mean Percentage by Language")
    
    # --- Subplot 2: Box Plot ---
    bp = ax2.boxplot(
        percentages_list,
        labels=languages,
        patch_artist=True,
        widths=0.6,
        zorder=3,
        flierprops=dict(marker='o', color='black', alpha=0.5, markersize=4),
        medianprops=dict(color='white', linewidth=1.5)
    )

    # Apply colors to boxplots to match bars
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)

    # Styling Ax2
    ax2.set_ylabel(r"Distribution (\%)")
    ax2.set_title("Distribution Density")

    # --- Global Styling for both axes ---
    for ax in (ax1, ax2):
        # Grid settings
        ax.grid(axis="y", linestyle='--', alpha=0.5, zorder=0, color='gray')
        
        # Remove top and right spines (Tufte style)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Tick rotation
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as '{save_path}'")
    plt.close()


def save_input_output_pairs(results, filename="input_output_pairs.json"):
    """
    Save input-output pairs to a separate JSON file.
    
    Args:
        results: List of result dictionaries from analyze_dataset()
        filename: Output filename (default: input_output_pairs.json)
    """
    if not results:
        print("No input-output pairs to save")
        return
    
    output_data = []
    
    for result in results:
        language = result['language']
        prompts = result.get('prompts', [])
        outputs = result.get('outputs', [])
        percentages = result.get('percentages', [])
        
        for i, (prompt, output, percentage) in enumerate(zip(prompts, outputs, percentages)):
            output_data.append({
                "id": i,
                "language": language,
                "dataset": result['dataset'],
                "input": prompt,
                "output": output,
                "uppercase_percentage": round(percentage, 2)
            })
    
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Input-output pairs saved to '{filename}'")


def plot_steering_comparison(results, save_path="steering_comparison.png"):
    """
    Create visualization comparing uppercase percentages across steering strengths.
    
    Args:
        results: List of result dictionaries with steering info
        save_path: Path to save the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Check if results have steering info
    if 'steering_strength' not in results[0]:
        print("Results don't contain steering information")
        return
    
    # Organize data by language and strength
    from collections import defaultdict
    data_by_language = defaultdict(lambda: {'strengths': [], 'means': [], 'stds': []})
    
    for result in results:
        lang = result['language']
        strength = result.get('steering_strength', 0.0)
        data_by_language[lang]['strengths'].append(strength)
        data_by_language[lang]['means'].append(result['mean'])
        data_by_language[lang]['stds'].append(result['std'])
    
    # Sort by strength for each language
    for lang in data_by_language:
        sorted_data = sorted(zip(
            data_by_language[lang]['strengths'],
            data_by_language[lang]['means'],
            data_by_language[lang]['stds']
        ))
        data_by_language[lang]['strengths'] = [x[0] for x in sorted_data]
        data_by_language[lang]['means'] = [x[1] for x in sorted_data]
        data_by_language[lang]['stds'] = [x[2] for x in sorted_data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    languages = sorted(data_by_language.keys())
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(languages))]
    
    # --- Subplot 1: Line plot with error bars ---
    for lang, color in zip(languages, colors):
        strengths = data_by_language[lang]['strengths']
        means = data_by_language[lang]['means']
        stds = data_by_language[lang]['stds']
        
        ax1.errorbar(strengths, means, yerr=stds, marker='o', linewidth=2,
                    markersize=8, capsize=5, label=lang, color=color, alpha=0.85)
    
    ax1.set_xlabel(r"Steering Strength")
    ax1.set_ylabel(r"Uppercase Characters (\\%)")
    ax1.set_title("Mean Uppercase \\% vs Steering Strength")
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Subplot 2: Grouped bar chart for all strengths ---
    unique_strengths = sorted(set(data_by_language[languages[0]]['strengths']))
    x = np.arange(len(languages))
    width = 0.8 / len(unique_strengths)
    
    for i, strength in enumerate(unique_strengths):
        means_at_strength = []
        for lang in languages:
            idx = data_by_language[lang]['strengths'].index(strength)
            means_at_strength.append(data_by_language[lang]['means'][idx])
        
        offset = (i - len(unique_strengths)/2 + 0.5) * width
        ax2.bar(x + offset, means_at_strength, width, 
               label=f'Strength {strength:.2f}', alpha=0.85)
    
    ax2.set_xlabel(r"Language")
    ax2.set_ylabel(r"Uppercase Characters (\\%)")
    ax2.set_title("Uppercase \\% by Language and Strength")
    ax2.set_xticks(x)
    ax2.set_xticklabels(languages, rotation=45, ha='right')
    ax2.legend(loc='best', ncol=2)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Steering comparison plot saved as '{save_path}'")
    plt.close()



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Base Model: {base_model_id}")
    print(f"LoRA Adapter: {lora_model_id}")

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_with_lora(base_model_id, lora_model_id, has_lora)
    model.to(DEVICE)
    
    # Create clean identifier from lora_model_id for filenames
    model_identifier = lora_model_id.split('/')[-1].replace(' ', '_')
    
    # Load steering vectors if provided
    steering_vector = None
    steering_layer = None
    steering_strengths = [0.0]  # Default: no steering
    
    if args.steering_vectors_path:
        print(f"\nLoading steering vector from: {args.steering_vectors_path}")
        loaded_data = torch.load(args.steering_vectors_path)
        
        if isinstance(loaded_data, dict):
            if 'steering_vector' in loaded_data:
                steering_vector = loaded_data['steering_vector']
            elif 'vector' in loaded_data:
                steering_vector = loaded_data['vector']
            else:
                # Try to find a tensor in the values
                tensors = [v for v in loaded_data.values() if isinstance(v, torch.Tensor)]
                if len(tensors) >= 1:
                     # Use the first one found, assuming standard single-vector file
                     steering_vector = tensors[0]
                else:
                     raise ValueError(f"Could not find tensor in loaded dict. Keys: {list(loaded_data.keys())}")
        else:
            steering_vector = loaded_data
            
        print(f"Loaded steering vector with shape: {steering_vector.shape}")
        
        if args.steering_layer is None:
            raise ValueError("--steering_layer must be specified when using steering vectors")
        
        steering_layer = args.steering_layer
        steering_strengths = [float(x.strip()) for x in args.steering_strengths.split(',')]
        
        print(f"Applying to layer: {steering_layer}")
        print(f"Testing strengths: {steering_strengths}")
    
    # Run evaluations
    all_results = []
    
    if steering_vector is not None:
        # Iterate through steering strengths
        for strength in steering_strengths:
            print(f"\n{'#'*60}")
            print(f"Strength: {strength:.2f}")
            print(f"{'#'*60}")
            
            # Analyze each dataset with this strength
            strength_results = []
            for dataset_name, language_name in DATASETS:
                steering_config = {
                    'layer_idx': steering_layer,
                    'steering_vector': steering_vector,  # Use the loaded vector directly
                    'strength': strength
                }
                
                result = analyze_dataset(
                    model, tokenizer, dataset_name, language_name, 
                    SAMPLE_SIZE, steering_config=steering_config
                )
                if result:
                    strength_results.append(result)
            
            all_results.extend(strength_results)
            
            # Save intermediate results for this strength
            if strength_results:
                strength_path = Path(save_path) / f"layer{steering_layer}_strength{strength:.2f}"
                strength_path.mkdir(parents=True, exist_ok=True)
                
                results_file = strength_path / f"{model_identifier}_uppercase_analysis_results.txt"
                pairs_file = strength_path / f"{model_identifier}_input_output_pairs.json"
                plot_file = strength_path / f"{model_identifier}_uppercase_analysis.png"
                
                save_results_to_file(strength_results, results_file)
                save_input_output_pairs(strength_results, pairs_file)
                plot_results(strength_results, plot_file)
    else:
        # Baseline: no steering
        print("\nRunning baseline evaluation (no steering)")
        for dataset_name, language_name in DATASETS:
            result = analyze_dataset(
                model, tokenizer, dataset_name, language_name, SAMPLE_SIZE
            )
            if result:
                all_results.append(result)

    # Save combined results
    if all_results:
        # Create filenames with model identifier
        combined_results_file = Path(save_path) / f"{model_identifier}_uppercase_analysis_results_all.txt"
        combined_pairs_file = Path(save_path) / f"{model_identifier}_input_output_pairs_all.json"
        
        # Save combined summary statistics
        save_results_to_file(all_results, combined_results_file)
        
        # Save combined input-output pairs
        save_input_output_pairs(all_results, combined_pairs_file)
        
        # Create combined steering comparison plot if using steering
        if steering_vector is not None:
            combined_plot_file = Path(save_path) / f"{model_identifier}_steering_comparison.png"
            plot_steering_comparison(all_results, combined_plot_file)
        
        # If no steering, also create single plot
        if steering_vector is None:
            plot_file = Path(save_path) / f"{model_identifier}_uppercase_analysis.png"
            plot_results(all_results, plot_file)
        
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if steering_vector is not None:
            print(f"{'Language':<15} {'Strength':<10} {'Mean %':<12} {'Std Dev':<12}")
            print("-"*80)
            for result in all_results:
                strength = result.get('steering_strength', 0.0)
                print(f"{result['language']:<15} {strength:<10.2f} "
                      f"{result['mean']:<12.2f} {result['std']:<12.2f}")
        else:
            print(f"{'Language':<15} {'Mean %':<12} {'Std Dev':<12}")
            print("-"*80)
            for result in all_results:
                print(f"{result['language']:<15} {result['mean']:<12.2f} {result['std']:<12.2f}")
        
        print(f"\n{'='*80}")
        print("OUTPUT FILES")
        print(f"{'='*80}")
        print(f"Combined Summary: {combined_results_file}")
        print(f"Combined Input-Output Pairs: {combined_pairs_file}")
        if steering_vector is not None:
            print(f"Per-strength results saved in: {Path(save_path)}/layer{steering_layer}_strength*/")
        print(f"{'='*80}")
    else:
        print("No results to display")


if __name__ == "__main__":
    main() 