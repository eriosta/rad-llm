"""
Modal Function: Direct Trajectory Generation from Reports

Generates trajectories directly from report text without RadGraph.
Uses Outlines for schema-compliant JSON generation.

Deploy with:
    modal deploy src/modal_direct_trajectories.py
"""

import modal
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json

# Create Modal app
app = modal.App("radiology-direct-outlinesv1")

# GPU image with Outlines for structured generation
# Using latest outlines which may have better dependency handling
outlines_image = (modal.Image.debian_slim(python_version="3.10")
         .env({"REBUILD_TIMESTAMP": "2024-11-05-21:45"})  # Force rebuild
         .pip_install(
             "numpy<2.0",
             "pydantic==2.8.2",
             "torch==2.4.0",
             "transformers==4.44.2",
             "accelerate==0.33.0",
             "outlines>=0.1.0"  # Latest version
         )
)


# ============================================================================
# Trajectory Schema (Pydantic)
# ============================================================================

class LesionTrajectory(BaseModel):
    """Schema for a single lesion trajectory across timepoints."""
    
    trajectory_id: str = Field(description="Unique trajectory ID (e.g., PATIENT_T001)")
    lesion_ids: List[str] = Field(description="Lesion IDs from each report (e.g., ['L001', 'L001', 'L001'])")
    timepoints: List[int] = Field(description="Timepoint indices (e.g., [0, 1, 2])")
    study_dates: List[str] = Field(description="Study dates (e.g., ['2024-01-15', '2024-03-18'])")
    anatomy: str = Field(description="Anatomical location (must be consistent across timepoints)")
    status: str = Field(description="Overall status: 'new', 'active', 'resolved'")
    trend: str = Field(description="Size trend description")
    size_progression: List[Optional[float]] = Field(description="Size in mm at each timepoint (use null if not measured)")
    reasoning: str = Field(description="Brief explanation of why these lesions were linked")


class TrajectoryResponse(BaseModel):
    """Response containing all trajectories for a patient."""
    
    patient_id: str
    trajectories: List[LesionTrajectory]


# ============================================================================
# Modal Function
# ============================================================================

@app.function(
    gpu="A100-40GB",  # High-quality A100 40GB for best performance
    image=outlines_image,
    timeout=1800,
    scaledown_window=300,
    cpu=4.0
)
def generate_trajectories_direct(
    patient_id: str,
    reports_text: str,
    rag_context: str
) -> Dict[str, Any]:
    """
    Generate lesion trajectories directly from report text using Outlines.
    
    Args:
        patient_id: Patient identifier
        reports_text: All reports formatted chronologically
        rag_context: Medical knowledge context from RAG
    
    Returns:
        Dictionary with patient_id and list of trajectories
    """
    import outlines
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"[Modal] Generating trajectories for {patient_id} (with Outlines)")
    print(f"[Modal] Context size: {len(rag_context)} chars")
    print(f"[Modal] Reports size: {len(reports_text)} chars")
    
    # Load model with Outlines (correct API)
    print("[Modal] Loading Qwen 2.5-7B with Outlines...")
    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto"
        ),
        AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True
        )
    )
    
    print("[Modal] Model loaded successfully with Outlines")
    
    # Build prompt (shorter, no JSON example needed - Outlines handles that)
    prompt = f"""<|im_start|>system
You are an expert radiologist AI specialized in longitudinal lesion tracking across serial CT scans.

TASK: Create trajectories by linking the SAME physical lesion across multiple timepoints.

LESION LINKING CRITERIA (in order of importance):
1. **Anatomical Location**: Same organ/lobe/segment (e.g., "Right upper lobe" = "Right upper lobe")
2. **Lesion Type**: Same pathology (mass, nodule, lymph node, etc.)
3. **Spatial Position**: Same relative position within the anatomy
4. **Size Evolution**: Gradual changes expected (sudden jumps indicate different lesions)

IMPORTANT RULES:
- Each unique anatomical location = ONE trajectory
- ALL lesions from ALL timepoints MUST be included
- If a lesion disappears/appears, note it (but still track what's present)
- Lymph nodes: Different stations (4R, 7) = DIFFERENT trajectories
- Convert measurements: cm → mm (multiply by 10)

{rag_context}<|im_end|>
<|im_start|>user
{reports_text}

STEP-BY-STEP INSTRUCTIONS:

Step 1: EXTRACT all lesions from EACH timepoint
- List every lesion mentioned in each report
- Note: lesion ID, anatomy, type, size

Step 2: LINK lesions across timepoints
- Group lesions by anatomical location
- Same location across timepoints = same trajectory
- Verify the lesion type is consistent

Step 3: CREATE trajectories
- For each group, create one trajectory
- List lesion IDs chronologically
- Extract size_progression in mm
- For trend: describe what you observe (e.g., "initially decreasing then increasing")

Step 4: VERIFY completeness
- Count total lesions extracted
- Count total trajectories created
- Ensure no lesions are missing

Generate the trajectory data now (patient_id: {patient_id}).<|im_end|>
<|im_start|>assistant
"""
    
    # Generate with structured output (guaranteed valid JSON)
    print("[Modal] Generating structured trajectories...")
    try:
        # Call model directly with prompt and schema (new Outlines API)
        result_json = model(prompt, TrajectoryResponse, max_new_tokens=4096)
        
        # Parse the JSON result
        result = TrajectoryResponse.model_validate_json(result_json)
        response_dict = result.model_dump()
        
        n_traj = len(response_dict.get('trajectories', []))
        print(f"[Modal] ✅ Generated {n_traj} trajectories")
        
        return response_dict
        
    except Exception as e:
        print(f"[Modal] ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'patient_id': patient_id,
            'trajectories': [],
            'error': str(e)
        }


@app.local_entrypoint()
def test():
    """Test the function locally."""
    
    sample_reports = """
### RPT-001 - 2024-01-15 (Timepoint 0)
- L001: Primary mass in Right upper lobe, measuring 3.2 cm. Status: New
- L002: Pulmonary nodule in Left lower lobe, measuring 0.6 cm. Status: New

### RPT-002 - 2024-03-18 (Timepoint 1)
- L001: Primary mass in Right upper lobe, measuring 2.4 cm. Status: Decreased
- L002: Pulmonary nodule in Left lower lobe, measuring 0.4 cm. Status: Decreased
"""
    
    rag_context = """
### Anatomical Terminology:
- Right upper lobe: anterior, posterior, apical segments
- Lesion tracking requires consistent anatomical location
"""
    
    result = generate_trajectories_direct.remote(
        patient_id="TEST_PATIENT",
        reports_text=sample_reports,
        rag_context=rag_context
    )
    
    print("\n" + "="*80)
    print("TEST RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2))
