import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Load the F1 scores data and display basic information."""
    df = pd.read_csv('results/f1_scores/combined_f1_scores.csv')
    
    print("📊 Data Overview:")
    print(f"Total records: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Phases: {sorted(df['phase'].unique())}")
    print(f"Experiment types: {sorted(df['experiment_type'].unique())}")
    print(f"Prompt types: {sorted(df['prompt_type'].unique())}")
    print()
    
    return df

def create_in_target_plots(df):
    """Create one plot per target for in-target experiments, with subplots for each model."""
    
    # Create plots directory
    os.makedirs('plots/in_target', exist_ok=True)
    
    in_target_data = df[df['experiment_type'] == 'in_target'].copy()
    
    if in_target_data.empty:
        print("No in-target data found!")
        return
    
    targets = sorted(in_target_data['target'].unique())
    models = sorted(in_target_data['model'].unique())
    prompt_types = sorted(in_target_data['prompt_type'].unique())
    
    for target in targets:
        target_data = in_target_data[in_target_data['target'] == target]
        
        # Create subplot for each model
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'In-Target: {target.upper()} - Phase 1 vs Phase 3 Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for i, model in enumerate(models):
            if i >= 4:  # Only handle up to 4 models
                break
                
            ax = axes_flat[i]
            model_data = target_data[target_data['model'] == model]
            
            if model_data.empty:
                ax.text(0.5, 0.5, f'No data for {model}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model.upper()}')
                continue
            
            # Prepare data for plotting
            x = np.arange(len(prompt_types))
            width = 0.35
            
            phase1_scores = []
            phase3_scores = []
            
            for prompt_type in prompt_types:
                prompt_data = model_data[model_data['prompt_type'] == prompt_type]
                
                p1_score = prompt_data[prompt_data['phase'] == 'phase1']['f1_score'].mean()
                p3_score = prompt_data[prompt_data['phase'] == 'phase3']['f1_score'].mean()
                
                phase1_scores.append(p1_score if not pd.isna(p1_score) else 0)
                phase3_scores.append(p3_score if not pd.isna(p3_score) else 0)
            
            # Create bars
            bars1 = ax.bar(x - width/2, phase1_scores, width, label='Phase 1', 
                          alpha=0.8, color='#1f77b4')
            bars2 = ax.bar(x + width/2, phase3_scores, width, label='Phase 3', 
                          alpha=0.8, color='#ff7f0e')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Prompt Type')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'{model.upper()}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(prompt_types, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(max(phase1_scores + phase3_scores) * 1.1, 0.1))
        
        # Hide empty subplots
        for i in range(len(models), 4):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'plots/in_target/target_{target}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: plots/in_target/target_{target}.png")

def create_cross_target_plots(df):
    """Create one plot per source->destination pair for cross-target experiments."""
    
    # Create plots directory
    os.makedirs('plots/cross_target', exist_ok=True)
    
    cross_target_data = df[df['experiment_type'] == 'cross_target'].copy()
    
    if cross_target_data.empty:
        print("No cross-target data found!")
        return
    
    # Create source->dest pairs
    cross_target_data['src_dst_pair'] = cross_target_data['source_target'] + '_to_' + cross_target_data['dest_target']
    
    pairs = sorted(cross_target_data['src_dst_pair'].unique())
    models = sorted(cross_target_data['model'].unique())
    prompt_types = sorted(cross_target_data['prompt_type'].unique())
    
    for pair in pairs:
        pair_data = cross_target_data[cross_target_data['src_dst_pair'] == pair]
        
        # Get source and destination for display
        src_target = pair_data['source_target'].iloc[0]
        dst_target = pair_data['dest_target'].iloc[0]
        
        # Create subplot for each model
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Cross-Target: {src_target.upper()} → {dst_target.upper()} - Phase 1 vs Phase 3 Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for i, model in enumerate(models):
            if i >= 4:  # Only handle up to 4 models
                break
                
            ax = axes_flat[i]
            model_data = pair_data[pair_data['model'] == model]
            
            if model_data.empty:
                ax.text(0.5, 0.5, f'No data for {model}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model.upper()}')
                continue
            
            # Prepare data for plotting
            x = np.arange(len(prompt_types))
            width = 0.35
            
            phase1_scores = []
            phase3_scores = []
            
            for prompt_type in prompt_types:
                prompt_data = model_data[model_data['prompt_type'] == prompt_type]
                
                p1_score = prompt_data[prompt_data['phase'] == 'phase1']['f1_score'].mean()
                p3_score = prompt_data[prompt_data['phase'] == 'phase3']['f1_score'].mean()
                
                phase1_scores.append(p1_score if not pd.isna(p1_score) else 0)
                phase3_scores.append(p3_score if not pd.isna(p3_score) else 0)
            
            # Create bars
            bars1 = ax.bar(x - width/2, phase1_scores, width, label='Phase 1', 
                          alpha=0.8, color='#2ca02c')
            bars2 = ax.bar(x + width/2, phase3_scores, width, label='Phase 3', 
                          alpha=0.8, color='#d62728')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Prompt Type')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'{model.upper()}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(prompt_types, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(max(phase1_scores + phase3_scores) * 1.1, 0.1))
        
        # Hide empty subplots
        for i in range(len(models), 4):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'plots/cross_target/{pair}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: plots/cross_target/{pair}.png")

def create_summary_plots(df):
    """Create summary plots showing overall trends."""
    
    os.makedirs('plots/summary', exist_ok=True)
    
    # 1. Overall model performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Model comparison
    model_avg = df.groupby(['model', 'phase'])['f1_score'].mean().unstack()
    x = np.arange(len(model_avg.index))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, model_avg['phase1'], width, label='Phase 1', alpha=0.8)
    bars2 = ax1.bar(x + width/2, model_avg['phase3'], width, label='Phase 3', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average F1 Score')
    ax1.set_title('Average F1 Score by Model', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_avg.index)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prompt type comparison
    prompt_avg = df.groupby(['prompt_type', 'phase'])['f1_score'].mean().unstack()
    x = np.arange(len(prompt_avg.index))
    
    bars1 = ax2.bar(x - width/2, prompt_avg['phase1'], width, label='Phase 1', alpha=0.8)
    bars2 = ax2.bar(x + width/2, prompt_avg['phase3'], width, label='Phase 3', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Prompt Type')
    ax2.set_ylabel('Average F1 Score')
    ax2.set_title('Average F1 Score by Prompt Type', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(prompt_avg.index, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/summary/overall_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: plots/summary/overall_performance_comparison.png")

def print_key_insights(df):
    """Print key insights from the data analysis"""
    
    print("\n" + "="*70)
    print("🔍 KEY INSIGHTS")
    print("="*70)

    # Overall improvement
    phase1_avg = df[df['phase'] == 'phase1']['f1_score'].mean()
    phase3_avg = df[df['phase'] == 'phase3']['f1_score'].mean()
    overall_improvement = phase3_avg - phase1_avg

    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Phase 1 average F1: {phase1_avg:.4f}")
    print(f"   Phase 3 average F1: {phase3_avg:.4f}")
    print(f"   Overall improvement: {overall_improvement:+.4f}")
    print()

    # Model-wise performance
    print("🤖 MODEL-WISE PERFORMANCE:")
    model_performance = df.groupby(['model', 'phase'])['f1_score'].mean().unstack()
    for model in model_performance.index:
        phase1_score = model_performance.loc[model, 'phase1']
        phase3_score = model_performance.loc[model, 'phase3']
        improvement = phase3_score - phase1_score
        print(f"   {model}: {phase1_score:.4f} → {phase3_score:.4f} ({improvement:+.4f})")
    print()

    print("="*70)

def main():
    """Main function to run all visualizations"""
    
    print("🚀 Starting Clean F1 Score Visualization Analysis...")
    print("="*50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Create in-target plots (one per target)
    print("🎯 Creating in-target plots (one per target)...")
    create_in_target_plots(df)
    
    # Create cross-target plots (one per source→destination pair)
    print("🔀 Creating cross-target plots (one per source→destination pair)...")
    create_cross_target_plots(df)
    
    # Create summary plots
    print("📊 Creating summary plots...")
    create_summary_plots(df)
    
    # Print key insights
    print_key_insights(df)
    
    print("\n✨ Clean visualization analysis complete!")
    print("📁 Plots organized in:")
    print("   plots/in_target/ - One plot per target")
    print("   plots/cross_target/ - One plot per source→destination pair") 
    print("   plots/summary/ - Overall performance summary")

if __name__ == "__main__":
    main()
