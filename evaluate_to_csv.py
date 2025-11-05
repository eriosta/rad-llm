"""
CSV-Based Trajectory Evaluation

Generates detailed CSV files for manual review and analysis:
1. trajectory_comparison.csv - Side-by-side comparison of each trajectory
2. measurement_errors.csv - Measurement-by-measurement error analysis
3. metrics_summary.csv - High-level metrics summary

Usage:
    python evaluate_to_csv.py

Output:
    evaluation_results/trajectory_comparison.csv
    evaluation_results/measurement_errors.csv
    evaluation_results/metrics_summary.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re


def parse_ground_truth_simple(gt_file: str) -> pd.DataFrame:
    """
    Parse ground truth into a simple DataFrame.
    
    Returns:
        DataFrame with columns: lesion_id, location, type, report_id, size_mm, notes
    """
    with open(gt_file, 'r') as f:
        content = f.read()
    
    records = []
    
    # Manual extraction of key lesions from ground truth
    # L001 - RUL Primary
    records.extend([
        {'lesion_id': 'L001', 'location': 'Right upper lobe, anterior segment', 
         'type': 'Primary adenocarcinoma', 'report_id': 'RPT-001', 'size_mm': 32.0, 
         'notes': 'Spiculated with pleural retraction'},
        {'lesion_id': 'L001', 'location': 'Right upper lobe, anterior segment', 
         'type': 'Primary adenocarcinoma', 'report_id': 'RPT-002', 'size_mm': 24.0, 
         'notes': 'Decreased 25% (Partial Response)'},
        {'lesion_id': 'L001', 'location': 'Right upper lobe, anterior segment', 
         'type': 'Primary adenocarcinoma', 'report_id': 'RPT-003', 'size_mm': 18.0, 
         'notes': 'Decreased 44% from baseline (Major PR)'},
        {'lesion_id': 'L001', 'location': 'Right upper lobe, anterior segment', 
         'type': 'Primary adenocarcinoma', 'report_id': 'RPT-004', 'size_mm': 18.0, 
         'notes': 'Stable (Sustained response)'},
        {'lesion_id': 'L001', 'location': 'Right upper lobe, anterior segment', 
         'type': 'Primary adenocarcinoma', 'report_id': 'RPT-005', 'size_mm': 26.0, 
         'notes': 'Increased 44% from prior (Progressive Disease)'},
    ])
    
    # L002 - LLL Nodule
    records.extend([
        {'lesion_id': 'L002', 'location': 'Left lower lobe, lateral basal segment', 
         'type': 'Pulmonary metastasis', 'report_id': 'RPT-001', 'size_mm': 6.0, 'notes': 'Solid nodule'},
        {'lesion_id': 'L002', 'location': 'Left lower lobe, lateral basal segment', 
         'type': 'Pulmonary metastasis', 'report_id': 'RPT-002', 'size_mm': 4.0, 'notes': 'Decreased 33%'},
        {'lesion_id': 'L002', 'location': 'Left lower lobe, lateral basal segment', 
         'type': 'Pulmonary metastasis', 'report_id': 'RPT-003', 'size_mm': 3.0, 'notes': 'Decreased 50%'},
        {'lesion_id': 'L002', 'location': 'Left lower lobe, lateral basal segment', 
         'type': 'Pulmonary metastasis', 'report_id': 'RPT-004', 'size_mm': 3.0, 'notes': 'Stable'},
        {'lesion_id': 'L002', 'location': 'Left lower lobe, lateral basal segment', 
         'type': 'Pulmonary metastasis', 'report_id': 'RPT-005', 'size_mm': 7.0, 'notes': 'Increased 133%'},
    ])
    
    # L005 - Paratracheal LN
    records.extend([
        {'lesion_id': 'L005', 'location': 'Right paratracheal (Station 4R)', 
         'type': 'Lymph node metastasis', 'report_id': 'RPT-001', 'size_mm': 18.0, 'notes': 'ENLARGED'},
        {'lesion_id': 'L005', 'location': 'Right paratracheal (Station 4R)', 
         'type': 'Lymph node metastasis', 'report_id': 'RPT-002', 'size_mm': 13.0, 'notes': 'decreased 28%'},
        {'lesion_id': 'L005', 'location': 'Right paratracheal (Station 4R)', 
         'type': 'Lymph node metastasis', 'report_id': 'RPT-003', 'size_mm': 10.0, 'notes': 'decreased 44%'},
        {'lesion_id': 'L005', 'location': 'Right paratracheal (Station 4R)', 
         'type': 'Lymph node metastasis', 'report_id': 'RPT-004', 'size_mm': 10.0, 'notes': 'stable'},
        {'lesion_id': 'L005', 'location': 'Right paratracheal (Station 4R)', 
         'type': 'Lymph node metastasis', 'report_id': 'RPT-005', 'size_mm': 21.0, 'notes': 'increased 110%'},
    ])
    
    # L006 - Subcarinal LN
    records.extend([
        {'lesion_id': 'L006', 'location': 'Subcarinal (Station 7)', 
         'type': 'Lymph node', 'report_id': 'RPT-001', 'size_mm': 15.0, 'notes': 'borderline enlarged'},
        {'lesion_id': 'L006', 'location': 'Subcarinal (Station 7)', 
         'type': 'Lymph node', 'report_id': 'RPT-002', 'size_mm': 12.0, 'notes': 'decreased'},
        {'lesion_id': 'L006', 'location': 'Subcarinal (Station 7)', 
         'type': 'Lymph node', 'report_id': 'RPT-003', 'size_mm': 9.0, 'notes': 'normalized'},
        {'lesion_id': 'L006', 'location': 'Subcarinal (Station 7)', 
         'type': 'Lymph node', 'report_id': 'RPT-004', 'size_mm': 9.0, 'notes': 'stable'},
        {'lesion_id': 'L006', 'location': 'Subcarinal (Station 7)', 
         'type': 'Lymph node', 'report_id': 'RPT-005', 'size_mm': 18.0, 'notes': 'increased 100%'},
    ])
    
    # L007 - Right Adrenal
    records.extend([
        {'lesion_id': 'L007', 'location': 'Right adrenal gland', 
         'type': 'Adrenal metastasis', 'report_id': 'RPT-001', 'size_mm': 12.0, 'notes': 'indeterminate'},
        {'lesion_id': 'L007', 'location': 'Right adrenal gland', 
         'type': 'Adrenal metastasis', 'report_id': 'RPT-002', 'size_mm': 10.0, 'notes': 'slight decrease'},
        {'lesion_id': 'L007', 'location': 'Right adrenal gland', 
         'type': 'Adrenal metastasis', 'report_id': 'RPT-003', 'size_mm': 8.0, 'notes': 'decreased'},
        {'lesion_id': 'L007', 'location': 'Right adrenal gland', 
         'type': 'Adrenal metastasis', 'report_id': 'RPT-004', 'size_mm': 8.0, 'notes': 'stable'},
        {'lesion_id': 'L007', 'location': 'Right adrenal gland', 
         'type': 'Adrenal metastasis', 'report_id': 'RPT-005', 'size_mm': 18.0, 'notes': 'increased 125%'},
    ])
    
    return pd.DataFrame(records)


def load_generated_trajectories(json_file: str) -> pd.DataFrame:
    """Load generated trajectories into DataFrame."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    patient_id = list(data['trajectories'].keys())[0]
    trajectories = data['trajectories'][patient_id]
    
    # Flatten to one row per trajectory
    records = []
    for traj in trajectories:
        # Get size progression and compute trend programmatically
        size_progression = traj.get('size_progression', [])
        sizes_valid = [s for s in size_progression if s is not None]
        
        # Compute trend from sizes (not LLM's trend field)
        if len(sizes_valid) >= 2:
            computed_trend = _compute_trend(sizes_valid)
        else:
            computed_trend = 'insufficient_data'
        
        record = {
            'trajectory_id': traj['trajectory_id'],
            'lesion_ids': '|'.join(traj['lesion_ids']),
            'n_lesions': len(traj['lesion_ids']),
            'anatomy': traj.get('anatomy', ''),
            'status': traj.get('status', ''),
            'trend': computed_trend,  # Use computed trend, not LLM's
            'confidence': traj.get('confidence', 0),
            'sizes': '|'.join([str(s) if s is not None else 'None' for s in size_progression]),
            'reasoning': traj.get('reasoning', '')[:200]  # Truncate
        }
        records.append(record)
    
    return pd.DataFrame(records)


def compare_trajectories(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Create side-by-side comparison of trajectories."""
    
    # For each ground truth lesion, find best matching generated trajectory
    comparisons = []
    
    for lesion_id in gt_df['lesion_id'].unique():
        lesion_gt = gt_df[gt_df['lesion_id'] == lesion_id]
        
        # Ground truth summary
        gt_location = lesion_gt.iloc[0]['location']
        gt_type = lesion_gt.iloc[0]['type']
        gt_sizes = lesion_gt['size_mm'].tolist()
        gt_trend = _compute_trend(gt_sizes)
        
        # Find best match in generated
        best_match = None
        best_score = 0
        
        for _, gen_row in gen_df.iterrows():
            score = _compute_match_score_simple(gt_location, gen_row['anatomy'])
            if score > best_score:
                best_score = score
                best_match = gen_row
        
        comparison = {
            'gt_lesion_id': lesion_id,
            'gt_location': gt_location,
            'gt_type': gt_type,
            'gt_sizes': '|'.join([f'{s:.1f}' for s in gt_sizes]),
            'gt_trend': gt_trend,
            'matched_trajectory': best_match['trajectory_id'] if best_match is not None else 'NONE',
            'gen_location': best_match['anatomy'] if best_match is not None else '',
            'gen_sizes': best_match['sizes'] if best_match is not None else '',
            'gen_trend': best_match['trend'] if best_match is not None else '',
            'match_score': best_score,
            'location_match': 'YES' if best_score > 0.7 else 'PARTIAL' if best_score > 0.3 else 'NO',
            'trend_match': 'YES' if (best_match is not None and best_match['trend'] == gt_trend) else 'NO',
        }
        comparisons.append(comparison)
    
    return pd.DataFrame(comparisons)


def compute_measurement_errors(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Compute measurement-by-measurement errors."""
    
    errors = []
    
    for lesion_id in gt_df['lesion_id'].unique():
        lesion_gt = gt_df[gt_df['lesion_id'] == lesion_id]
        gt_location = lesion_gt.iloc[0]['location']
        
        # Find matching generated trajectory
        best_match = None
        best_score = 0
        for _, gen_row in gen_df.iterrows():
            score = _compute_match_score_simple(gt_location, gen_row['anatomy'])
            if score > best_score:
                best_score = score
                best_match = gen_row
        
        if best_match is None:
            continue
        
        # Parse sizes
        gt_sizes = lesion_gt['size_mm'].tolist()
        gen_sizes_str = best_match['sizes'].split('|')
        gen_sizes = []
        for s in gen_sizes_str:
            try:
                gen_sizes.append(float(s) if s != 'None' else None)
            except:
                gen_sizes.append(None)
        
        # Compare timepoint by timepoint
        for i, (gt_size, report_id) in enumerate(zip(gt_sizes, lesion_gt['report_id'])):
            if i < len(gen_sizes) and gen_sizes[i] is not None:
                gen_size = gen_sizes[i]
                error = abs(gen_size - gt_size)
                pct_error = (error / gt_size) * 100 if gt_size > 0 else 0
                
                errors.append({
                    'lesion_id': lesion_id,
                    'report_id': report_id,
                    'gt_size_mm': gt_size,
                    'gen_size_mm': gen_size,
                    'absolute_error_mm': error,
                    'percent_error': pct_error,
                    'within_1mm': 'YES' if error <= 1.0 else 'NO',
                    'within_10pct': 'YES' if pct_error <= 10 else 'NO',
                })
    
    return pd.DataFrame(errors)


def compute_metrics_summary(comparison_df: pd.DataFrame, errors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute high-level metrics summary."""
    
    metrics = {}
    
    # Linking metrics
    total_gt = len(comparison_df)
    matched = len(comparison_df[comparison_df['match_score'] > 0.5])
    perfect_match = len(comparison_df[comparison_df['location_match'] == 'YES'])
    
    metrics['Total Ground Truth Lesions'] = total_gt
    metrics['Matched Trajectories'] = matched
    metrics['Perfect Location Match'] = perfect_match
    metrics['Recall'] = f"{matched / total_gt:.2%}" if total_gt > 0 else "0%"
    metrics['Location Accuracy'] = f"{perfect_match / total_gt:.2%}" if total_gt > 0 else "0%"
    
    # Trend metrics
    trend_correct = len(comparison_df[comparison_df['trend_match'] == 'YES'])
    metrics['Trend Accuracy'] = f"{trend_correct / matched:.2%}" if matched > 0 else "0%"
    
    # Measurement metrics
    if len(errors_df) > 0:
        metrics['Mean Absolute Error (mm)'] = f"{errors_df['absolute_error_mm'].mean():.2f}"
        metrics['Median Absolute Error (mm)'] = f"{errors_df['absolute_error_mm'].median():.2f}"
        metrics['Mean Percent Error'] = f"{errors_df['percent_error'].mean():.1f}%"
        metrics['Within 1mm'] = f"{(errors_df['within_1mm'] == 'YES').sum() / len(errors_df):.2%}"
        metrics['Within 10%'] = f"{(errors_df['within_10pct'] == 'YES').sum() / len(errors_df):.2%}"
    
    # Overall quality score
    linking_score = matched / total_gt if total_gt > 0 else 0
    location_score = perfect_match / total_gt if total_gt > 0 else 0
    trend_score = trend_correct / matched if matched > 0 else 0
    
    overall = 0.4 * linking_score + 0.3 * location_score + 0.3 * trend_score
    metrics['Overall Quality Score'] = f"{overall:.2%}"
    
    return pd.DataFrame([metrics]).T.reset_index()


def _compute_trend(sizes: List[float]) -> str:
    """Compute trend from size measurements."""
    if len(sizes) < 2:
        return 'insufficient_data'
    
    baseline = sizes[0]
    latest = sizes[-1]
    
    # Handle zero baseline
    if baseline == 0:
        return 'new' if latest > 0 else 'stable'
    
    pct_change = ((latest - baseline) / baseline) * 100
    
    if pct_change < -30:
        return 'decreasing'
    elif pct_change > 20:
        return 'increasing'
    else:
        return 'stable'


def _compute_match_score_simple(gt_location: str, gen_location: str) -> float:
    """Simple matching score based on term overlap."""
    gt_terms = set(gt_location.lower().replace(',', '').split())
    gen_terms = set(gen_location.lower().replace(',', '').split())
    
    if not gt_terms:
        return 0.0
    
    overlap = len(gt_terms & gen_terms)
    return overlap / len(gt_terms)


def main():
    """Generate CSV evaluation files."""
    
    print("="*70)
    print("CSV-BASED TRAJECTORY EVALUATION")
    print("="*70)
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n📁 Loading data...")
    gt_df = parse_ground_truth_simple('demo/LESION_TRAJECTORY_SUMMARY.txt')
    gen_df = load_generated_trajectories('outputs/direct_trajectories.json')
    
    print(f"   Ground truth lesions: {len(gt_df['lesion_id'].unique())}")
    print(f"   Generated trajectories: {len(gen_df)}")
    
    # Generate comparisons
    print("\n📊 Computing comparisons...")
    comparison_df = compare_trajectories(gt_df, gen_df)
    errors_df = compute_measurement_errors(gt_df, gen_df)
    metrics_df = compute_metrics_summary(comparison_df, errors_df)
    
    # Save CSVs
    print("\n💾 Saving results...")
    comparison_df.to_csv(output_dir / 'trajectory_comparison.csv', index=False)
    errors_df.to_csv(output_dir / 'measurement_errors.csv', index=False)
    metrics_df.to_csv(output_dir / 'metrics_summary.csv', index=False, header=False)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   - trajectory_comparison.csv")
    print(f"   - measurement_errors.csv")
    print(f"   - metrics_summary.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)
    print(metrics_df.to_string(index=False, header=False))
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

