"""Render the violin plots used in Figures 1, 2, and 3 of the paper."""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parent.parent
F1_SCORES_PATH = REPO_ROOT / "results" / "f1_scores" / "combined_f1_scores.csv"
OUT_DIR = REPO_ROOT / "results" / "plots" / "violins"

# Set style for better looking plots with Arial font and bold text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare data for visualization."""
    df = pd.read_csv(F1_SCORES_PATH)
    df.fillna('Not Applicable', inplace=True)
    
    # Add prompt complexity column
    prompt_hierarchy = {
        'vanilla': 1,
        'knowledge_infused': 2, 
        'few_shot': 3,
        'cot': 2,
        'cot_knowledge': 4,
        'cot_knowledge_few_shot': 5
    }
    df['prompt_complexity'] = df['prompt_type'].map(prompt_hierarchy)
    
    # Add model size information
    MODEL_SIZES = {
        'phi': {'size': '3B', 'params': 3, 'category': 'Small'},
        'mistral_7b': {'size': '7B', 'params': 7, 'category': 'Medium'},
        'llama3_8b': {'size': '8B', 'params': 8, 'category': 'Medium'},
        'mistral_24b': {'size': '24B', 'params': 24, 'category': 'Large'}
    }
    
    df['model_size'] = df['model'].map(lambda x: MODEL_SIZES[x]['size'])
    df['model_params'] = df['model'].map(lambda x: MODEL_SIZES[x]['params'])
    df['model_category'] = df['model'].map(lambda x: MODEL_SIZES[x]['category'])
    
    # Separate data by experiment type
    in_target_df = df[df['experiment_type'] == 'in_target'].copy()
    cross_target_df = df[df['experiment_type'] == 'cross_target'].copy()
    
    return df, in_target_df, cross_target_df

def create_base_vs_lora_comparison(df, in_target_df, cross_target_df):
    """
    1. Comparison between Base (Non Fine Tuned) and LoRA f1 scores across all models, 
    for In target, and cross target.
    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define model order
    model_order = ['phi', 'mistral_7b', 'llama3_8b', 'mistral_24b']
    
    # In-Target Comparison
    in_target_melted = in_target_df.melt(
        id_vars=['phase', 'model'], 
        value_vars=['f1_score'], 
        var_name='metric', 
        value_name='score'
    )
    in_target_melted['phase_label'] = in_target_melted['phase'].map({
        'phase1': 'Base (Non Fine-Tuned)', 
        'phase3': 'LoRA Fine-Tuned'
    })
    
    sns.violinplot(data=in_target_melted, x='model', y='score', hue='phase_label', 
                   ax=ax1, inner='quart', palette=['lightcoral', 'skyblue'],
                   order=model_order)

    # Add connecting lines for in-target
    for phase, color, label in [('phase1', 'darkred', 'Base Mean'), 
                                  ('phase3', 'darkblue', 'LoRA Mean')]:
        phase_data = in_target_melted[in_target_melted['phase'] == phase]
        if not phase_data.empty:
            means = phase_data.groupby('model')['score'].mean()
            means = means.reindex(model_order)
            x_positions = range(len(means))
            y_values = means.values
            ax1.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                    color=color, alpha=0.6, zorder=10)

    # Fix model names to remove underscores
    model_labels = [model.replace('_', ' ').title() for model in model_order]
    ax1.set_xticklabels(model_labels, fontweight='bold')
    ax1.set_title('SD', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xlabel('Model', fontsize=24, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=24, fontweight='bold')
    ax1.set_ylim(-0.09, 1.09)
    ax1.tick_params(axis='x', rotation=45, labelsize=26, which='major')
    ax1.tick_params(axis='y', labelsize=26, which='major')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(title='Model Type', loc='best', title_fontsize=18, fontsize=16, prop={'weight': 'bold'})
    
    # Cross-Target Comparison
    cross_target_melted = cross_target_df.melt(
        id_vars=['phase', 'model'], 
        value_vars=['f1_score'], 
        var_name='metric', 
        value_name='score'
    )
    cross_target_melted['phase_label'] = cross_target_melted['phase'].map({
        'phase1': 'Base (Non Fine-Tuned)', 
        'phase3': 'LoRA Fine-Tuned'
    })
    
    sns.violinplot(data=cross_target_melted, x='model', y='score', hue='phase_label', 
                   ax=ax2, inner='quart', palette=['lightcoral', 'skyblue'],
                   order=model_order)

    # Add connecting lines for cross-target
    for phase, color in [('phase1', 'darkred'), ('phase3', 'darkblue')]:
        phase_data = cross_target_melted[cross_target_melted['phase'] == phase]
        if not phase_data.empty:
            means = phase_data.groupby('model')['score'].mean()
            means = means.reindex(model_order)
            x_positions = range(len(means))
            y_values = means.values
            ax2.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                    color=color, alpha=0.6, zorder=10)

    # Fix model names to remove underscores  
    model_labels = [model.replace('_', ' ').title() for model in model_order]
    ax2.set_xticklabels(model_labels, fontweight='bold')
    ax2.set_title('$SD_{CT}$ + $SD_{CD}$', fontsize=26, fontweight='bold', pad=20)
    ax2.set_xlabel('Model', fontsize=24, fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_ylim(-0.09, 1.09)
    ax2.tick_params(axis='x', rotation=45, labelsize=26, which='major')
    ax2.tick_params(axis='y', labelsize=0, which='major')
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(title='Model Type', title_fontsize=18, fontsize=16, loc='best', prop={'weight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'violin_1_base_vs_lora_comparison.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'violin_1_base_vs_lora_comparison.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.show()

def create_prompt_comparison(df, in_target_df, cross_target_df):
    """
    2. Comparison between each prompt, Base (Non Fine Tuned) and LoRA f1 scores 
    across all models, for In target, and cross target.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Define the specific order
    prompt_order = [
        'vanilla',
        'few_shot',
        'knowledge_infused',
        'cot',
        'cot_knowledge',
        'cot_knowledge_few_shot'
    ]
    
    # Define consistent naming
    prompt_titles = {
        'vanilla': 'Zero-Shot',
        'few_shot': 'Few-Shot',
        'knowledge_infused': 'Knowledge Infused',
        'cot': 'CoT',
        'cot_knowledge': 'CoT + Knowledge',
        'cot_knowledge_few_shot': 'CoT + Knowledge + Few-Shot'
    }
    
    # Define experiment type order
    exp_type_order = ['in_target', 'cross_target']
    
    for i, prompt in enumerate(prompt_order):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if prompt not in df['prompt_type'].unique():
            ax.set_visible(False)
            continue
        
        prompt_data = df[df['prompt_type'] == prompt].copy()
        
        prompt_melted = prompt_data.melt(
            id_vars=['phase', 'experiment_type'], 
            value_vars=['f1_score'], 
            var_name='metric', 
            value_name='score'
        )
        prompt_melted['phase_experiment'] = (
            prompt_melted['phase'].map({'phase1': 'Base', 'phase3': 'LoRA'}) + 
            ' (' + prompt_melted['experiment_type'].map({'in_target': 'SD', 'cross_target': '$SD_{CT}$ + $SD_{CD}$'}) + ')'
        )
        
        sns.violinplot(data=prompt_melted, x='experiment_type', y='score', hue='phase', 
                       ax=ax, inner='quart', palette=['lightcoral', 'skyblue'],
                       order=exp_type_order)

        # Add connecting lines for each phase
        for phase, color in [('phase1', 'darkred'), ('phase3', 'darkblue')]:
            phase_data = prompt_melted[prompt_melted['phase'] == phase]
            if not phase_data.empty:
                means = phase_data.groupby('experiment_type')['score'].mean()
                means = means.reindex(exp_type_order)
                x_positions = range(len(means))
                y_values = means.values
                ax.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                        color=color, alpha=0.6, zorder=10)

        ax.set_xticklabels(['SD', '$SD_{CT}$ + $SD_{CD}$'], fontweight='bold')
        
        title = prompt_titles.get(prompt, prompt.replace("_", " ").title())
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('')
        
        if col == 0:
            ax.set_ylabel('F1 Score', fontsize=24, fontweight='bold')
            ax.tick_params(axis='y', labelsize=26, which='major')
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelsize=0, which='major')
            ax.set_yticklabels([])
        
        ax.tick_params(axis='x', labelsize=26, which='major')
        ax.grid(True, alpha=0.3, axis='y')
        
        if i == 0:
            ax.legend(title='Model Type', labels=['Base', 'LoRA Fine-Tuned'], 
                     title_fontsize=18, fontsize=16, loc='best', prop={'weight': 'bold'})
        else:
            ax.legend().remove()
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'violin_2_prompt_comparison.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'violin_2_prompt_comparison.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.show()

def create_model_size_performance_gains(df, in_target_df, cross_target_df):
    """
    3. Comparison between each model size performance gains, 
    for in target and cross-target across all prompts.
    """
    # Calculate performance gains for each model
    performance_gains = []
    
    for exp_type, exp_df in [('SD', in_target_df), ('$SD_{CT}$ + $SD_{CD}$', cross_target_df)]:
        for model in exp_df['model'].unique():
            model_data = exp_df[exp_df['model'] == model]
            
            for prompt in model_data['prompt_type'].unique():
                prompt_data = model_data[model_data['prompt_type'] == prompt]
                
                phase1_scores = prompt_data[prompt_data['phase'] == 'phase1']['f1_score']
                phase3_scores = prompt_data[prompt_data['phase'] == 'phase3']['f1_score']
                
                if not phase1_scores.empty and not phase3_scores.empty:
                    phase1_mean = phase1_scores.mean()
                    phase3_mean = phase3_scores.mean()
                    gain = phase3_mean - phase1_mean
                    
                    model_info = {
                        'phi': {'size': '3B', 'params': 3, 'category': 'Small'},
                        'mistral_7b': {'size': '7B', 'params': 7, 'category': 'Medium'},
                        'llama3_8b': {'size': '8B', 'params': 8, 'category': 'Medium'},
                        'mistral_24b': {'size': '24B', 'params': 24, 'category': 'Large'}
                    }
                    
                    performance_gains.append({
                        'experiment_type': exp_type,
                        'model': model,
                        'model_size': model_info[model]['size'],
                        'model_category': model_info[model]['category'],
                        'model_params': model_info[model]['params'],
                        'prompt_type': prompt,
                        'performance_gain': gain,
                        'phase1_mean': phase1_mean,
                        'phase3_mean': phase3_mean
                    })
    
    gains_df = pd.DataFrame(performance_gains)
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define category order
    category_order = ['Small', 'Medium', 'Large']
    
    # In-Target Performance Gains
    in_target_gains = gains_df[gains_df['experiment_type'] == 'SD']
    
    sns.violinplot(data=in_target_gains, x='model_category', y='performance_gain', 
                   ax=ax1, inner='quart', palette='viridis', order=category_order)
    
    # Add connecting line for in-target
    if not in_target_gains.empty:
        means = in_target_gains.groupby('model_category')['performance_gain'].mean()
        means = means.reindex(category_order)
        x_positions = range(len(means))
        y_values = means.values
        ax1.plot(x_positions, y_values, 'o-', linewidth=3, markersize=8, 
                color='black', alpha=0.8, label='Mean', zorder=10)
    
    ax1.set_title('SD', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xlabel('Model Category', fontsize=24, fontweight='bold')
    ax1.set_ylabel('LoRA Performance Gain\n(F1 Score)', fontsize=24, fontweight='bold')
    ax1.set_ylim(-0.6, 0.6)
    ax1.tick_params(axis='x', labelsize=26, which='major')
    ax1.tick_params(axis='y', labelsize=26, which='major')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.legend(title='Trend Line', title_fontsize=16, fontsize=14, loc='best', prop={'weight': 'bold'})
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Cross-Target Performance Gains
    cross_target_gains = gains_df[gains_df['experiment_type'] == '$SD_{CT}$ + $SD_{CD}$']
    
    sns.violinplot(data=cross_target_gains, x='model_category', y='performance_gain', 
                   ax=ax2, inner='quart', palette='plasma', order=category_order)
    
    # Add connecting line for cross-target
    if not cross_target_gains.empty:
        means = cross_target_gains.groupby('model_category')['performance_gain'].mean()
        means = means.reindex(category_order)
        x_positions = range(len(means))
        y_values = means.values
        ax2.plot(x_positions, y_values, 'o-', linewidth=3, markersize=8, 
                color='black', alpha=0.8, label='Mean', zorder=10)
    
    ax2.set_title('$SD_{CT}$ + $SD_{CD}$', fontsize=26, fontweight='bold', pad=20)
    ax2.set_xlabel('Model Category', fontsize=24, fontweight='bold')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_ylabel('')
    ax2.tick_params(axis='x', labelsize=26, which='major')
    ax2.tick_params(axis='y', labelsize=0, which='major')
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.legend(title='Trend Line', title_fontsize=16, fontsize=14, loc='best', prop={'weight': 'bold'})
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'violin_3_model_size_performance_gains.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'violin_3_model_size_performance_gains.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.show()
    
    gains_df.to_csv(str(OUT_DIR / 'model_size_performance_gains_data.csv'), index=False)
    print("📊 Performance gains data saved to CSV")

def create_model_size_performance_scores(df, in_target_df, cross_target_df):
    """
    4. Comparison between each model size performance scores (actual F1 scores), 
    for base and fine-tuned phases across in-target and cross-target experiments.
    """
    performance_scores = []
    
    for exp_type, exp_df in [('SD', in_target_df), ('$SD_{CT}$ + $SD_{CD}$', cross_target_df)]:
        for model in exp_df['model'].unique():
            model_data = exp_df[exp_df['model'] == model]
            
            for prompt in model_data['prompt_type'].unique():
                prompt_data = model_data[model_data['prompt_type'] == prompt]
                
                model_info = {
                    'phi': {'size': '3B', 'params': 3, 'category': 'Small'},
                    'mistral_7b': {'size': '7B', 'params': 7, 'category': 'Medium'},
                    'llama3_8b': {'size': '8B', 'params': 8, 'category': 'Medium'},
                    'mistral_24b': {'size': '24B', 'params': 24, 'category': 'Large'}
                }
                
                phase1_scores = prompt_data[prompt_data['phase'] == 'phase1']['f1_score']
                for score in phase1_scores:
                    performance_scores.append({
                        'experiment_type': exp_type,
                        'model': model,
                        'model_size': model_info[model]['size'],
                        'model_category': model_info[model]['category'],
                        'model_params': model_info[model]['params'],
                        'prompt_type': prompt,
                        'phase': 'Base (Non Fine-Tuned)',
                        'f1_score': score
                    })
                
                phase3_scores = prompt_data[prompt_data['phase'] == 'phase3']['f1_score']
                for score in phase3_scores:
                    performance_scores.append({
                        'experiment_type': exp_type,
                        'model': model,
                        'model_size': model_info[model]['size'],
                        'model_category': model_info[model]['category'],
                        'model_params': model_info[model]['params'],
                        'prompt_type': prompt,
                        'phase': 'LoRA Fine-Tuned',
                        'f1_score': score
                    })
    
    scores_df = pd.DataFrame(performance_scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define orders
    category_order = ['Small', 'Medium', 'Large']
    phase_order = ['phase1', 'phase3']
    
    # In-Target Performance Scores
    in_target_scores = scores_df[scores_df['experiment_type'] == 'SD']
    
    sns.violinplot(data=in_target_scores, x='model_category', y='f1_score', hue='phase',
                   ax=ax1, inner='quart', palette=['lightcoral', 'skyblue'], 
                   order=category_order)
    
    # Add connecting lines for each phase
    for phase, color in [('Base (Non Fine-Tuned)', 'darkred'), ('LoRA Fine-Tuned', 'darkblue')]:
        phase_data = in_target_scores[in_target_scores['phase'] == phase]
        if not phase_data.empty:
            means = phase_data.groupby('model_category')['f1_score'].mean()
            means = means.reindex(category_order)
            x_positions = range(len(means))
            y_values = means.values
            ax1.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                    color=color, alpha=0.6, zorder=10)
    
    ax1.set_title('SD', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xlabel('Model Category', fontsize=24, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=24, fontweight='bold')
    ax1.set_ylim(-0.09, 1.09)
    ax1.tick_params(axis='x', labelsize=26, which='major')
    ax1.tick_params(axis='y', labelsize=26, which='major')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(title='Model Type', title_fontsize=18, fontsize=16, loc='best', prop={'weight': 'bold'})
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Cross-Target Performance Scores
    cross_target_scores = scores_df[scores_df['experiment_type'] == '$SD_{CT}$ + $SD_{CD}$']
    
    sns.violinplot(data=cross_target_scores, x='model_category', y='f1_score', hue='phase',
                   ax=ax2, inner='quart', palette=['lightcoral', 'skyblue'], 
                   order=category_order)
    
    # Add connecting lines for each phase
    for phase, color in [('Base (Non Fine-Tuned)', 'darkred'), ('LoRA Fine-Tuned', 'darkblue')]:
        phase_data = cross_target_scores[cross_target_scores['phase'] == phase]
        if not phase_data.empty:
            means = phase_data.groupby('model_category')['f1_score'].mean()
            means = means.reindex(category_order)
            x_positions = range(len(means))
            y_values = means.values
            ax2.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                    color=color, alpha=0.6, zorder=10)
    
    ax2.set_title('$SD_{CT}$ + $SD_{CD}$', fontsize=26, fontweight='bold', pad=20)
    ax2.set_xlabel('Model Category', fontsize=24, fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_ylim(-0.09, 1.09)
    ax2.tick_params(axis='x', labelsize=26, which='major')
    ax2.tick_params(axis='y', labelsize=0, which='major')
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(title='Model Type', title_fontsize=18, fontsize=16, loc='best', prop={'weight': 'bold'})
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'violin_4_model_size_performance_scores.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'violin_4_model_size_performance_scores.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.show()
    
    scores_df.to_csv(str(OUT_DIR / 'model_size_performance_scores_data.csv'), index=False)
    print("📊 Performance scores data saved to CSV")

def create_model_size_performance_scores_base_only(df, in_target_df, cross_target_df):
    """
    5. Comparison between each model size performance scores (actual F1 scores) for BASE ONLY, 
    across in-target and cross-target experiments.
    """
    performance_scores = []
    
    for exp_type, exp_df in [('SD', in_target_df), ('$SD_{CT}$ + $SD_{CD}$', cross_target_df)]:
        for model in exp_df['model'].unique():
            model_data = exp_df[exp_df['model'] == model]
            
            for prompt in model_data['prompt_type'].unique():
                prompt_data = model_data[model_data['prompt_type'] == prompt]
                
                model_info = {
                    'phi': {'size': '3B', 'params': 3, 'category': 'Small'},
                    'mistral_7b': {'size': '7B', 'params': 7, 'category': 'Medium'},
                    'llama3_8b': {'size': '8B', 'params': 8, 'category': 'Medium'},
                    'mistral_24b': {'size': '24B', 'params': 24, 'category': 'Large'}
                }
                
                phase1_scores = prompt_data[prompt_data['phase'] == 'phase1']['f1_score']
                for score in phase1_scores:
                    performance_scores.append({
                        'experiment_type': exp_type,
                        'model': model,
                        'model_size': model_info[model]['size'],
                        'model_category': model_info[model]['category'],
                        'model_params': model_info[model]['params'],
                        'prompt_type': prompt,
                        'phase': 'Base (Non Fine-Tuned)',
                        'f1_score': score
                    })
    
    scores_df = pd.DataFrame(performance_scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Define category order
    category_order = ['Small', 'Medium', 'Large']
    
    # In-Target Performance Scores (Base Only)
    in_target_scores = scores_df[scores_df['experiment_type'] == 'SD']
    
    sns.violinplot(data=in_target_scores, x='model_category', y='f1_score',
                   ax=ax1, inner='quart', palette=['lightcoral'], 
                   order=category_order)
    
    # Add connecting line for in-target
    if not in_target_scores.empty:
        means = in_target_scores.groupby('model_category')['f1_score'].mean()
        means = means.reindex(category_order)
        x_positions = range(len(means))
        y_values = means.values
        ax1.plot(x_positions, y_values, 'o-', linewidth=3, markersize=8, 
                color='darkred', alpha=0.8, label='Mean', zorder=10)
    
    ax1.set_title('SD', fontsize=26, fontweight='bold', pad=20)
    ax1.set_xlabel('Model Category', fontsize=24, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=24, fontweight='bold')
    ax1.set_ylim(-0.09, 1.09)
    ax1.tick_params(axis='x', labelsize=26, which='major')
    ax1.tick_params(axis='y', labelsize=26, which='major')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(title='Trend Line', title_fontsize=16, fontsize=14, loc='best', prop={'weight': 'bold'})
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Cross-Target Performance Scores (Base Only)
    cross_target_scores = scores_df[scores_df['experiment_type'] == '$SD_{CT}$ + $SD_{CD}$']
    
    sns.violinplot(data=cross_target_scores, x='model_category', y='f1_score',
                   ax=ax2, inner='quart', palette=['skyblue'], 
                   order=category_order)
    
    # Add connecting line for cross-target
    if not cross_target_scores.empty:
        means = cross_target_scores.groupby('model_category')['f1_score'].mean()
        means = means.reindex(category_order)
        x_positions = range(len(means))
        y_values = means.values
        ax2.plot(x_positions, y_values, 'o-', linewidth=3, markersize=8, 
                color='darkblue', alpha=0.8, label='Mean', zorder=10)
    
    ax2.set_title('$SD_{CT}$ + $SD_{CD}$', fontsize=26, fontweight='bold', pad=20)
    ax2.set_xlabel('Model Category', fontsize=24, fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_ylim(-0.09, 1.09)
    ax2.tick_params(axis='x', labelsize=26, which='major')
    ax2.tick_params(axis='y', labelsize=0, which='major')
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(title='Trend Line', title_fontsize=16, fontsize=14, loc='best', prop={'weight': 'bold'})
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'violin_5_model_size_performance_scores_base_only.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'violin_5_model_size_performance_scores_base_only.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.show()
    
    scores_df.to_csv(str(OUT_DIR / 'model_size_performance_scores_base_only_data.csv'), index=False)
    print("📊 Base-only performance scores data saved to CSV")

def create_prompt_comparison_base_only(df, in_target_df, cross_target_df):
    """
    6. Comparison between each prompt, Base (Non Fine Tuned) f1 scores ONLY
    across all models, for In target, and cross target.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    prompt_order = [
        'vanilla',
        'few_shot',
        'knowledge_infused',
        'cot',
        'cot_knowledge',
        'cot_knowledge_few_shot'
    ]
    
    prompt_titles = {
        'vanilla': 'Zero-Shot',
        'few_shot': 'Few-Shot',
        'knowledge_infused': 'Knowledge Infused',
        'cot': 'CoT',
        'cot_knowledge': 'CoT + Knowledge',
        'cot_knowledge_few_shot': 'CoT + Knowledge + Few-Shot'
    }
    
    # Define experiment type order
    exp_type_order = ['in_target', 'cross_target']
    
    for i, prompt in enumerate(prompt_order):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if prompt not in df['prompt_type'].unique():
            ax.set_visible(False)
            continue
        
        prompt_data = df[df['prompt_type'] == prompt].copy()
        prompt_data = prompt_data[prompt_data['phase'] == 'phase1']
        
        prompt_melted = prompt_data.melt(
            id_vars=['phase', 'experiment_type'], 
            value_vars=['f1_score'], 
            var_name='metric', 
            value_name='score'
        )
        
        sns.violinplot(data=prompt_melted, x='experiment_type', y='score', 
                       ax=ax, inner='quart', palette=['lightcoral', 'skyblue'],
                       order=exp_type_order)

        # Add connecting line for this prompt
        if not prompt_melted.empty:
            means = prompt_melted.groupby('experiment_type')['score'].mean()
            means = means.reindex(exp_type_order)
            x_positions = range(len(means))
            y_values = means.values
            ax.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                    color='darkgreen', alpha=0.8, zorder=10)

        ax.set_xticklabels(['SD', '$SD_{CT}$ + $SD_{CD}$'], fontweight='bold')
        
        title = prompt_titles.get(prompt, prompt.replace("_", " ").title())
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('')
        
        if col == 0:
            ax.set_ylabel('F1 Score', fontsize=24, fontweight='bold')
            ax.tick_params(axis='y', labelsize=26, which='major')
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelsize=0, which='major')
            ax.set_yticklabels([])
        
        ax.tick_params(axis='x', labelsize=26, which='major')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'violin_6_prompt_comparison_base_only.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'violin_6_prompt_comparison_base_only.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.show()

def create_additional_insights_plots(df, in_target_df, cross_target_df):
    """Create additional insightful violin plots."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define orders
    exp_type_order = ['in_target', 'cross_target']
    
    # Overall distribution comparison
    df_melted = df.melt(
        id_vars=['phase', 'experiment_type'], 
        value_vars=['f1_score'], 
        var_name='metric', 
        value_name='score'
    )
    df_melted['phase_label'] = df_melted['phase'].map({
        'phase1': 'Base (Non Fine-Tuned)', 
        'phase3': 'LoRA Fine-Tuned'
    })
    
    sns.violinplot(data=df_melted, x='experiment_type', y='score', hue='phase_label', 
                   ax=ax1, inner='quart', palette=['lightcoral', 'skyblue'],
                   order=exp_type_order)

    # Add connecting lines for each phase
    for phase, color in [('phase1', 'darkred'), ('phase3', 'darkblue')]:
        phase_data = df_melted[df_melted['phase'] == phase]
        if not phase_data.empty:
            means = phase_data.groupby('experiment_type')['score'].mean()
            means = means.reindex(exp_type_order)
            x_positions = range(len(means))
            y_values = means.values
            ax1.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                    color=color, alpha=0.6, zorder=10)

    ax1.set_xticklabels(['SD', '$SD_{CT}$ + $SD_{CD}$'], fontweight='bold')
    ax1.set_xlabel('', fontsize=24, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=24, fontweight='bold')
    ax1.set_ylim(-0.09, 1.09)
    ax1.tick_params(axis='x', labelsize=26, which='major')
    ax1.tick_params(axis='y', labelsize=26, which='major')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(title='Model Type', title_fontsize=18, fontsize=16, loc='best', prop={'weight': 'bold'})
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Prompt complexity vs performance
    prompt_complexity_data = df.copy()
    prompt_complexity_data = prompt_complexity_data.dropna(subset=['prompt_complexity'])
    
    # Get unique complexity levels and sort them
    complexity_order = sorted(prompt_complexity_data['prompt_complexity'].unique())
    
    sns.violinplot(data=prompt_complexity_data, x='prompt_complexity', y='f1_score', hue='phase', 
                   ax=ax2, inner='quart', palette=['lightcoral', 'skyblue'],
                   order=complexity_order)
    
    # Add connecting lines for each phase
    for phase, color in [('phase1', 'darkred'), ('phase3', 'darkblue')]:
        phase_data = prompt_complexity_data[prompt_complexity_data['phase'] == phase]
        if not phase_data.empty:
            means = phase_data.groupby('prompt_complexity')['f1_score'].mean()
            means = means.reindex(complexity_order)
            x_positions = range(len(means))
            y_values = means.values
            ax2.plot(x_positions, y_values, 'o-', linewidth=2, markersize=6, 
                    color=color, alpha=0.6, zorder=10)
    
    ax2.set_xlabel('Prompt Complexity Level', fontsize=24, fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_ylim(-0.09, 1.09)
    ax2.tick_params(axis='x', labelsize=26, which='major')
    ax2.tick_params(axis='y', labelsize=0, which='major')
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(title='Model-Type', labels=['Base', 'LoRA Fine-Tuned'], 
              title_fontsize=18, fontsize=16, loc='best', prop={'weight': 'bold'})
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'violin_bonus_insights.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(str(OUT_DIR / 'violin_bonus_insights.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all violin plots."""
    print("🎨 Starting Violin Plot Analysis...")
    print("="*50)
    
    # Create output directory
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load data
    print("📊 Loading and preparing data...")
    df, in_target_df, cross_target_df = load_and_prepare_data()
    
    print(f"✅ Data loaded: {len(df)} total records")
    print(f"   - In-Target: {len(in_target_df)} records")
    print(f"   - Cross-Target: {len(cross_target_df)} records")
    print()
    
    # Generate plots
    print("🎻 Generating Violin Plot 1: Base vs LoRA Comparison...")
    create_base_vs_lora_comparison(df, in_target_df, cross_target_df)
    
    print("🎻 Generating Violin Plot 2: Prompt-wise Comparison...")
    create_prompt_comparison(df, in_target_df, cross_target_df)
    
    print("🎻 Generating Violin Plot 3: Model Size Performance Gains...")
    create_model_size_performance_gains(df, in_target_df, cross_target_df)
    
    print("🎻 Generating Violin Plot 4: Model Size Performance Scores...")
    create_model_size_performance_scores(df, in_target_df, cross_target_df)
    
    print("🎻 Generating Violin Plot 5: Model Size Performance Scores (Base Only)...")
    create_model_size_performance_scores_base_only(df, in_target_df, cross_target_df)
    
    print("🎻 Generating Violin Plot 6: Prompt Comparison (Base Only)...")
    create_prompt_comparison_base_only(df, in_target_df, cross_target_df)
    
    print("🎻 Generating Bonus Insights Plots...")
    create_additional_insights_plots(df, in_target_df, cross_target_df)
    
    print("\n✨ All violin plots generated successfully!")
    print(f"📁 Saved to {OUT_DIR}/")
    print("\n📊 Generated plots:")
    print("   1. violin_1_base_vs_lora_comparison.png")
    print("   2. violin_2_prompt_comparison.png")
    print("   3. violin_3_model_size_performance_gains.png")
    print("   4. violin_4_model_size_performance_scores.png")
    print("   5. violin_5_model_size_performance_scores_base_only.png")
    print("   6. violin_6_prompt_comparison_base_only.png")
    print("   7. violin_bonus_insights.png")
    print("   8. model_size_performance_gains_data.csv (supporting data)")
    print("   9. model_size_performance_scores_data.csv (supporting data)")
    print("  10. model_size_performance_scores_base_only_data.csv (supporting data)")

if __name__ == "__main__":
    main()