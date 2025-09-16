import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
import re
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Base directories
PARSED_PHASE1_DIR = "results/parsed_phase1_results"
PARSED_PHASE3_DIR = "results/parsed_phase3_results"
OUTPUT_DIR = "results/f1_scores"

# Dataset label mappings
DATASET_LABELS = {
    'wtwt': ['SUPPORT', 'REFUTE', 'COMMENT', 'UNRELATED'],
    'semeval': ['FAVOR', 'AGAINST', 'NONE'],
    'pstance': ['FAVOR', 'AGAINST'],
    'covid': ['FAVOR', 'AGAINST', 'NONE']
}

def create_output_directories():
    """Create the output directory structure."""
    directories = [
        f"{OUTPUT_DIR}/phase1/in_target",
        f"{OUTPUT_DIR}/phase1/cross_target",
        f"{OUTPUT_DIR}/phase3/in_target", 
        f"{OUTPUT_DIR}/phase3/cross_target"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Created output directory structure")

def validate_file_data(df: pd.DataFrame, dataset: str, filename: str) -> Tuple[bool, str]:
    """Validate that file has correct columns and expected label values."""
    
    # Check if predicted_stance column exists
    if 'predicted_stance' not in df.columns:
        return False, f"Missing 'predicted_stance' column"
    
    if 'parsed_output' not in df.columns:
        return False, f"Missing 'parsed_output' column"
    
    # Check unique values in parsed_output (this is what we use for F1 calculation)
    unique_predicted = set(df['parsed_output'].unique())
    expected_labels = set(DATASET_LABELS[dataset])
    expected_count = len(expected_labels)
    
    # Check if we have the right number of unique labels (allowing for subset)
    if len(unique_predicted) > expected_count:
        unexpected = unique_predicted - expected_labels
        return False, f"Unexpected labels found: {unexpected}. Expected only: {expected_labels}"
    
    # Check if all predicted labels are valid
    invalid_labels = unique_predicted - expected_labels
    if invalid_labels:
        return False, f"Invalid labels found: {invalid_labels}. Expected: {expected_labels}"
    
    # Check true_stance as well
    if 'true_stance' in df.columns:
        unique_true = set(df['true_stance'].unique())
        invalid_true = unique_true - expected_labels
        if invalid_true:
            return False, f"Invalid true labels found: {invalid_true}. Expected: {expected_labels}"
    
    return True, f"Valid - Found {len(unique_predicted)} unique predicted labels, {len(set(df['true_stance'].unique()))} unique true labels"

def calculate_f1_score(y_true: List[str], y_pred: List[str], dataset: str) -> float:
    """Calculate overall F1 score (macro average across all labels in dataset)."""
    all_labels = DATASET_LABELS[dataset]
    return round(f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0), 4)

def extract_file_info(filename: str, experiment_type: str) -> Dict[str, str]:
    """Extract information from filename."""
    info = {}
    
    if experiment_type == 'in_target':
        # Format: {model}_{prompt_type}_target-{target}_dataset-{dataset}_type-{type}_results.csv
        # Model names can be: mistral_24b, mistral_7b, llama3_8b, phi
        pattern = r'((?:mistral_(?:24b|7b)|llama3_8b|phi))_([^_]+(?:_[^_]*)*?)_target-([^_]+)_dataset-([^_]+)_type-([^_]+)_results\.csv'
        match = re.match(pattern, filename)
        
        if match:
            info['model'] = match.group(1)
            info['prompt_type'] = match.group(2)
            info['target'] = match.group(3)
            info['dataset'] = match.group(4)
            info['type'] = match.group(5)
        else:
            raise ValueError(f"Could not parse in_target filename: {filename}")
            
    elif experiment_type == 'cross_target':
        # Format: {model}_{prompt_type}_src-{source}_dst-{dest}_results.csv
        # Model names can be: mistral_24b, mistral_7b, llama3_8b, phi
        pattern = r'((?:mistral_(?:24b|7b)|llama3_8b|phi))_([^_]+(?:_[^_]*)*?)_src-([^_]+)_dst-([^_]+)_results\.csv'
        match = re.match(pattern, filename)
        
        if match:
            info['model'] = match.group(1)
            info['prompt_type'] = match.group(2)
            info['source_target'] = match.group(3)
            info['dest_target'] = match.group(4)
        else:
            raise ValueError(f"Could not parse cross_target filename: {filename}")
    
    return info

def process_single_file(filepath: str, experiment_type: str) -> Dict:
    """Process a single CSV file and calculate metrics."""
    try:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Extract file info
        file_info = extract_file_info(filename, experiment_type)
        
        # Determine dataset
        if experiment_type == 'in_target':
            dataset = file_info['dataset']
        else:  # cross_target
            # Get dataset from the actual data
            if 'dest_dataset' in df.columns:
                dataset = df['dest_dataset'].iloc[0]
            else:
                print(f"❌ ERROR: Could not determine dataset for {filename}")
                return None
        
        # Validate dataset
        if dataset not in DATASET_LABELS:
            print(f"❌ ERROR: Unknown dataset '{dataset}' in {filename}")
            return None
        
        # Validate file data
        is_valid, validation_msg = validate_file_data(df, dataset, filename)
        if not is_valid:
            print(f"❌ VALIDATION ERROR in {filename}: {validation_msg}")
            return None
        else:
            print(f"✅ {validation_msg}")
        
        # Calculate F1 score
        y_true = df['true_stance'].tolist()
        y_pred = df['parsed_output'].tolist()
        
        f1_overall = calculate_f1_score(y_true, y_pred, dataset)
        
        # Calculate additional metrics
        metrics = {
            'filename': filename,
            'model': file_info['model'],
            'prompt_type': file_info['prompt_type'],
            'dataset': dataset,
            'experiment_type': experiment_type,
            'total_samples': len(df),
            'f1_score': f1_overall,
            'fallback_used_count': df['fallback_used'].sum() if 'fallback_used' in df.columns else 0,
            'prompted_again_count': df['prompted_again'].sum() if 'prompted_again' in df.columns else 0,
            'number_tries_mean': df['number_tries'].mean() if 'number_tries' in df.columns else 1.0,
            'number_tries_median': df['number_tries'].median() if 'number_tries' in df.columns else 1.0,
        }
        
        # Add target information
        if experiment_type == 'in_target':
            metrics['target'] = file_info['target']
        else:  # cross_target
            metrics['source_target'] = file_info['source_target']
            metrics['dest_target'] = file_info['dest_target']
        
        return metrics
        
    except Exception as e:
        print(f"❌ ERROR processing {filepath}: {e}")
        return None

def validate_directory(input_dir: str, experiment_type: str, phase: str) -> Tuple[int, int]:
    """Pre-validate all files in a directory and return counts."""
    total_files = 0
    valid_files = 0
    invalid_files = []
    
    print(f"\n🔍 Pre-validating {phase} {experiment_type} files...")
    
    for root, dirs, files in os.walk(input_dir):
        if root == input_dir:
            continue
            
        for file in files:
            if file.endswith('.csv'):
                total_files += 1
                filepath = os.path.join(root, file)
                
                try:
                    # Quick validation without full processing
                    filename = os.path.basename(filepath)
                    file_info = extract_file_info(filename, experiment_type)
                    
                    df = pd.read_csv(filepath)
                    
                    # Determine dataset
                    if experiment_type == 'in_target':
                        dataset = file_info['dataset']
                    else:
                        if 'dest_dataset' in df.columns:
                            dataset = df['dest_dataset'].iloc[0]
                        else:
                            invalid_files.append((filename, "No dest_dataset column"))
                            continue
                    
                    if dataset not in DATASET_LABELS:
                        invalid_files.append((filename, f"Unknown dataset: {dataset}"))
                        continue
                    
                    # Validate data
                    is_valid, validation_msg = validate_file_data(df, dataset, filename)
                    if is_valid:
                        valid_files += 1
                        print(f"✅ {filename}: {validation_msg}")
                    else:
                        invalid_files.append((filename, validation_msg))
                        print(f"❌ {filename}: {validation_msg}")
                        
                except Exception as e:
                    invalid_files.append((filename, str(e)))
                    print(f"❌ {filename}: Error - {e}")
    
    print(f"\n📊 Validation Summary for {phase} {experiment_type}:")
    print(f"  Total files: {total_files}")
    print(f"  Valid files: {valid_files}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print(f"\n❌ Invalid files:")
        for filename, reason in invalid_files:
            print(f"  - {filename}: {reason}")
    
    return valid_files, total_files

def process_directory(input_dir: str, experiment_type: str, phase: str) -> pd.DataFrame:
    """Process all files in a directory."""
    # First validate all files
    valid_count, total_count = validate_directory(input_dir, experiment_type, phase)
    
    if valid_count == 0:
        print(f"⚠️ No valid files found in {input_dir}")
        return pd.DataFrame()
    
    print(f"\n📈 Processing {valid_count}/{total_count} valid files...")
    
    all_metrics = []
    
    # Walk through model directories
    for root, dirs, files in os.walk(input_dir):
        # Skip non-model directories
        if root == input_dir:
            continue
            
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                metrics = process_single_file(filepath, experiment_type)
                if metrics:
                    metrics['phase'] = phase
                    all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

def main():
    """Main function to process all results and generate F1 score summaries."""
    print("🚀 Starting F1 score calculation with validation...")
    
    # Create output directories
    create_output_directories()
    
    # Process each combination
    configurations = [
        (PARSED_PHASE1_DIR, "phase1"),
        (PARSED_PHASE3_DIR, "phase3")
    ]
    
    for base_dir, phase in configurations:
        if not os.path.exists(base_dir):
            print(f"⚠️ Directory not found: {base_dir}")
            continue
            
        print(f"\n📊 Processing {phase} results...")
        
        # Process in_target
        in_target_dir = os.path.join(base_dir, "in_target")
        if os.path.exists(in_target_dir):
            print(f"\n🎯 Processing {phase} in_target...")
            df_in_target = process_directory(in_target_dir, "in_target", phase)
            
            if not df_in_target.empty:
                output_path = f"{OUTPUT_DIR}/{phase}/in_target/f1_scores_summary.csv"
                df_in_target.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")
                print(f"   📈 Successfully processed {len(df_in_target)} files")
            else:
                print(f"⚠️ No valid data found for {phase} in_target")
        
        # Process cross_target
        cross_target_dir = os.path.join(base_dir, "cross_target")
        if os.path.exists(cross_target_dir):
            print(f"\n🔀 Processing {phase} cross_target...")
            df_cross_target = process_directory(cross_target_dir, "cross_target", phase)
            
            if not df_cross_target.empty:
                output_path = f"{OUTPUT_DIR}/{phase}/cross_target/f1_scores_summary.csv"
                df_cross_target.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")
                print(f"   📈 Successfully processed {len(df_cross_target)} files")
            else:
                print(f"⚠️ No valid data found for {phase} cross_target")
    
    # Generate combined summaries
    print("\n📋 Generating combined summaries...")
    
    # Combine all results for easier analysis
    all_dfs = []
    for phase in ['phase1', 'phase3']:
        for exp_type in ['in_target', 'cross_target']:
            summary_path = f"{OUTPUT_DIR}/{phase}/{exp_type}/f1_scores_summary.csv"
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_output_path = f"{OUTPUT_DIR}/combined_f1_scores.csv"
        combined_df.to_csv(combined_output_path, index=False)
        print(f"✅ Saved combined results: {combined_output_path}")
        
        # Print summary statistics
        print("\n📊 Final Summary Statistics:")
        print(f"Total files processed: {len(combined_df)}")
        print(f"Models: {sorted(combined_df['model'].unique())}")
        print(f"Prompt types: {sorted(combined_df['prompt_type'].unique())}")
        print(f"Datasets: {sorted(combined_df['dataset'].unique())}")
        
        # F1 score statistics
        print(f"\n🎯 F1 Score Statistics:")
        print(f"F1 Score - Mean: {combined_df['f1_score'].mean():.3f}, Std: {combined_df['f1_score'].std():.3f}")
        print(f"F1 Score - Min: {combined_df['f1_score'].min():.3f}, Max: {combined_df['f1_score'].max():.3f}")
        
        # Fallback and retry statistics
        print(f"\n🔄 Retry Statistics:")
        print(f"Average fallback usage: {combined_df['fallback_used_count'].mean():.2f}")
        print(f"Average prompted again: {combined_df['prompted_again_count'].mean():.2f}")
        print(f"Average number of tries: {combined_df['number_tries_mean'].mean():.2f}")
        
        # Dataset breakdown
        print(f"\n📊 Dataset Breakdown:")
        dataset_summary = combined_df.groupby(['dataset', 'phase']).agg({
            'f1_score': ['count', 'mean', 'std']
        }).round(3)
        print(dataset_summary)
        
        # Top performing configurations
        print(f"\n🏆 Top 5 F1 Scores:")
        top_configs = combined_df.nlargest(5, 'f1_score')[['model', 'prompt_type', 'dataset', 'phase', 'experiment_type', 'f1_score']]
        print(top_configs.to_string(index=False))
    
    print("\n✨ F1 score calculation complete!")

if __name__ == "__main__":
    main()
