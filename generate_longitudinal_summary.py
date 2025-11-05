"""
Modal GPU Script: Generate Longitudinal Impression Summary

Reads direct_trajectories.json and generates a 50-word summary of the 
longitudinal impression using Modal GPU with Qwen 2.5-7B-Instruct.

Usage:
    modal run generate_longitudinal_summary.py
"""

import modal
import json
from pathlib import Path

# Create Modal app
app = modal.App("longitudinal-summary")

# GPU image with necessary dependencies
summary_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "accelerate==0.33.0",
    )
)


@app.function(
    gpu="A100",  # A100 GPU for fast inference
    image=summary_image,
    timeout=600,
    cpu=4.0
)
def generate_summary(trajectories_data: dict) -> str:
    """
    Generate a 50-word longitudinal impression summary from trajectories.
    
    Args:
        trajectories_data: Dictionary containing trajectory information
    
    Returns:
        50-word summary string
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print("[Modal] Loading Qwen 2.5-7B-Instruct...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    
    print("[Modal] Model loaded successfully")
    
    # Extract key information from trajectories
    metadata = trajectories_data.get("metadata", {})
    trajectories = trajectories_data.get("trajectories", {})
    
    # Prepare trajectory summary for the prompt
    trajectory_summary = []
    for patient_id, patient_trajectories in trajectories.items():
        for traj in patient_trajectories:
            traj_id = traj.get("trajectory_id", "unknown")
            anatomy = traj.get("anatomy", "unknown location")
            status = traj.get("status", "unknown")
            trend = traj.get("trend", "unknown")
            size_prog = traj.get("size_progression", [])
            
            trajectory_summary.append(
                f"- {traj_id}: {anatomy} | Status: {status} | Trend: {trend} | "
                f"Sizes: {size_prog}"
            )
    
    trajectory_text = "\n".join(trajectory_summary)
    
    # Build prompt
    prompt = f"""<|im_start|>system
You are an expert radiologist summarizing longitudinal imaging findings.
Your task is to write a concise 50-word longitudinal impression that captures the overall disease trajectory across all timepoints.<|im_end|>
<|im_start|>user
Based on the following lesion trajectories across {metadata.get('n_reports', 'multiple')} timepoints for a patient with {metadata.get('n_trajectories', 'multiple')} tracked lesions, write a 50-word longitudinal impression:

TRAJECTORIES:
{trajectory_text}

Write a clinical longitudinal impression (exactly 50 words) summarizing the overall disease course, response patterns, and current status.<|im_end|>
<|im_start|>assistant
Longitudinal Impression:
"""
    
    print("[Modal] Generating summary...")
    
    # Generate summary
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Allow some buffer for 50 words
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract the summary
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "Longitudinal Impression:" in full_response:
        summary = full_response.split("Longitudinal Impression:")[-1].strip()
    else:
        summary = full_response.split("<|im_start|>assistant")[-1].strip()
    
    print(f"[Modal] ✅ Generated summary ({len(summary.split())} words)")
    
    return summary


@app.local_entrypoint()
def main():
    """Main entry point - reads trajectories and generates summary."""
    
    # Path to trajectories file
    trajectories_path = Path(__file__).parent / "outputs" / "direct_trajectories.json"
    
    print(f"Reading trajectories from: {trajectories_path}")
    
    if not trajectories_path.exists():
        print(f"❌ Error: File not found: {trajectories_path}")
        return
    
    # Load trajectories data
    with open(trajectories_path, 'r') as f:
        trajectories_data = json.load(f)
    
    print(f"✓ Loaded {trajectories_data['metadata']['n_trajectories']} trajectories")
    print(f"✓ Spanning {trajectories_data['metadata']['n_reports']} reports")
    print("\n" + "="*80)
    print("Generating 50-word longitudinal impression summary using Modal GPU...")
    print("="*80 + "\n")
    
    # Generate summary using Modal GPU
    summary = generate_summary.remote(trajectories_data)
    
    # Display result
    print("\n" + "="*80)
    print("LONGITUDINAL IMPRESSION (50-word summary):")
    print("="*80)
    print(summary)
    print("\n" + "="*80)
    
    # Save to file
    output_path = Path(__file__).parent / "outputs" / "longitudinal_summary.txt"
    with open(output_path, 'w') as f:
        f.write(f"Longitudinal Impression Summary\n")
        f.write(f"Generated from {trajectories_data['metadata']['n_trajectories']} trajectories\n")
        f.write(f"="*80 + "\n\n")
        f.write(summary)
        f.write(f"\n\n(Word count: {len(summary.split())} words)\n")
    
    print(f"\n✓ Summary saved to: {output_path}")

