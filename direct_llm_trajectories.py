#!/usr/bin/env python3
"""
Direct LLM Trajectory Generation (No RadGraph)

Skip RadGraph extraction entirely. Feed raw report text directly to LLM
with RAG (RadLex, LOINC) and use Outlines to generate structured JSON.

Architecture:
    1. Load report text from demo/*.json or demo/*.txt
    2. Retrieve medical knowledge via RAG (RadLex, LOINC)
    3. LLM with Outlines generates trajectory JSON directly

Benefits:
    - Simpler pipeline (no RadGraph)
    - Leverages LLM's NLP strengths
    - No entity linking errors
    - Guaranteed schema compliance (Outlines)

Usage:
    python direct_llm_trajectories.py

Output:
    direct_trajectories.json

Author: Simplified approach
Date: 2025-11-05
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import sys

# Add src to path for modal import
sys.path.insert(0, str(Path(__file__).parent))


class ReportLoader:
    """Load radiology reports from demo files."""
    
    def __init__(self, demo_dir: str = 'demo'):
        self.demo_dir = Path(demo_dir)
    
    def load_all_reports(self) -> List[Dict]:
        """
        Load all reports with metadata.
        
        Returns:
            List of report dictionaries with text and metadata
        """
        reports = []
        
        # Try JSON files first
        json_files = sorted(self.demo_dir.glob('*.json'))
        
        if json_files:
            print(f"Loading {len(json_files)} JSON reports...")
            for i, json_file in enumerate(json_files):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get('report_metadata', {})
                
                # Build full report text from structured data
                report_text = self._build_report_text(data)
                
                reports.append({
                    'report_id': metadata.get('report_id', f'RPT-{i+1:03d}'),
                    'patient_id': 'DEMO_PATIENT',
                    'timepoint': i,
                    'study_date': metadata.get('exam_date', ''),
                    'exam_type': metadata.get('exam_type', ''),
                    'findings_text': report_text,
                    'raw_data': data
                })
        else:
            # Fallback to text files
            txt_files = sorted(self.demo_dir.glob('*.txt'))
            print(f"Loading {len(txt_files)} text reports...")
            
            for i, txt_file in enumerate(txt_files):
                with open(txt_file, 'r') as f:
                    text = f.read()
                
                reports.append({
                    'report_id': f'RPT-{i+1:03d}',
                    'patient_id': 'DEMO_PATIENT',
                    'timepoint': i,
                    'study_date': '',
                    'exam_type': 'CT Chest',
                    'findings_text': text,
                    'raw_data': None
                })
        
        print(f"✅ Loaded {len(reports)} reports")
        return reports
    
    def _build_report_text(self, data: Dict) -> str:
        """Build full report text from structured JSON data."""
        
        text_parts = []
        
        # Patient info
        patient_info = data.get('patient_info', {})
        text_parts.append(f"Patient: {patient_info.get('age', 'Unknown')} year old {patient_info.get('sex', 'Unknown')}")
        text_parts.append(f"Clinical History: {patient_info.get('clinical_history', 'Unknown')}\n")
        
        # Findings
        text_parts.append("FINDINGS:")
        
        for lesion in data.get('lesions', []):
            lesion_id = lesion.get('lesion_id', '')
            location = lesion.get('location', '')
            lesion_type = lesion.get('type', '')
            measurements = lesion.get('measurements', {})
            characteristics = lesion.get('characteristics', '')
            status = lesion.get('status', '')
            
            # Build lesion description
            desc = f"- {lesion_id}: {lesion_type} in {location}"
            
            # Add measurements
            if 'max_diameter_cm' in measurements:
                desc += f", measuring {measurements['max_diameter_cm']} cm"
            elif 'long_axis_cm' in measurements:
                desc += f", measuring {measurements['long_axis_cm']} x {measurements.get('short_axis_cm', 'N/A')} cm"
            
            if characteristics:
                desc += f". {characteristics}"
            
            if status:
                desc += f". Status: {status}"
            
            text_parts.append(desc)
        
        # Staging
        staging = data.get('staging_summary', {})
        if staging:
            text_parts.append(f"\nProvisional Stage: {staging.get('provisional_stage', 'Unknown')}")
        
        return "\n".join(text_parts)


class MedicalKnowledgeRAG:
    """Retrieve medical knowledge from indexed collections."""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = QdrantClient(qdrant_url)
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        print("✅ RAG retriever initialized")
    
    def retrieve_knowledge(self, reports: List[Dict], n_docs: int = 5) -> str:
        """
        Retrieve medical knowledge relevant to the reports.
        
        Args:
            reports: List of report dictionaries
            n_docs: Number of documents to retrieve per source
        
        Returns:
            Formatted medical knowledge string
        """
        # Build query from all reports
        all_text = " ".join([r['findings_text'][:500] for r in reports])
        
        query = f"""
        Longitudinal tracking of lung lesions and lymph nodes.
        Tumor measurement standards and response assessment.
        Anatomical terminology for thoracic imaging.
        {all_text[:200]}
        """
        
        # Generate embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        knowledge_sections = []
        
        # Retrieve from RadLex
        try:
            radlex_results = self.client.search(
                collection_name="radlex",
                query_vector=query_embedding,
                limit=n_docs
            )
            
            if radlex_results:
                knowledge_sections.append("### Anatomical Terminology (RadLex):")
                for i, hit in enumerate(radlex_results, 1):
                    text = hit.payload.get('text', '')
                    knowledge_sections.append(f"{i}. {text[:200]}")
        except Exception as e:
            print(f"   ⚠️  RadLex retrieval failed: {e}")
        
        # Retrieve from LOINC
        try:
            loinc_results = self.client.search(
                collection_name="loinc_procedures",
                query_vector=query_embedding,
                limit=3
            )
            
            if loinc_results:
                knowledge_sections.append("\n### Imaging Standards (LOINC):")
                for i, hit in enumerate(loinc_results, 1):
                    text = hit.payload.get('text', '')
                    knowledge_sections.append(f"{i}. {text[:200]}")
        except Exception as e:
            print(f"   ⚠️  LOINC retrieval failed: {e}")
        
        if not knowledge_sections:
            return "No external medical knowledge retrieved."
        
        return "\n".join(knowledge_sections)


class DirectLLMGenerator:
    """Generate trajectories directly from reports using Modal LLM."""
    
    def __init__(self, app_name: str = 'radiology-direct-outlinesv1'):
        self.app_name = app_name
        self.function_name = 'generate_trajectories_direct'
        self.modal_function = None
    
    def connect(self):
        """Connect to Modal function."""
        import modal
        
        try:
            self.modal_function = modal.Function.from_name(
                self.app_name,
                self.function_name
            )
            print(f"✅ Connected to Modal: {self.app_name}.{self.function_name}")
        except Exception as e:
            print(f"⚠️  Modal connection failed: {e}")
            print(f"\n💡 Deploy the function first:")
            print(f"   modal deploy src/modal_direct_trajectories.py")
            raise
    
    def generate(self, patient_id: str, reports: List[Dict], rag_context: str) -> Dict:
        """
        Generate trajectories from reports.
        
        Args:
            patient_id: Patient identifier
            reports: List of report dictionaries
            rag_context: Medical knowledge context
        
        Returns:
            Dictionary with trajectories
        """
        if not self.modal_function:
            self.connect()
        
        print(f"\n🤖 Calling Modal LLM to generate trajectories...")
        print(f"   Patient: {patient_id}")
        print(f"   Reports: {len(reports)}")
        print(f"   Context size: {len(rag_context)} chars")
        
        # Format all reports into a single prompt
        reports_text = self._format_reports_for_llm(reports)
        
        print(f"   Reports text size: {len(reports_text)} chars")
        
        try:
            # Call Modal with new signature
            result = self.modal_function.remote(
                patient_id=patient_id,
                reports_text=reports_text,
                rag_context=rag_context
            )
            
            n_trajectories = len(result.get('trajectories', []))
            print(f"   ✅ Generated {n_trajectories} trajectories")
            return result
            
        except Exception as e:
            print(f"   ❌ Modal call failed: {e}")
            import traceback
            traceback.print_exc()
            return {'patient_id': patient_id, 'trajectories': [], 'error': str(e)}
    
    def _format_reports_for_llm(self, reports: List[Dict]) -> str:
        """Format reports chronologically for LLM."""
        
        formatted = []
        
        for report in reports:
            formatted.append(f"""
### {report['report_id']} - {report['study_date']} (Timepoint {report['timepoint']})

{report['findings_text']}
""")
        
        return "\n".join(formatted)


def main():
    """Main execution."""
    
    print("="*80)
    print("DIRECT LLM TRAJECTORY GENERATION (No RadGraph)")
    print("="*80)
    print("\nApproach: Reports → LLM + RAG → JSON (via Outlines)")
    
    # Step 1: Load reports
    print("\n[1/4] Loading reports from demo/...")
    loader = ReportLoader('demo')
    reports = loader.load_all_reports()
    
    # Step 2: Retrieve medical knowledge
    print("\n[2/4] Retrieving medical knowledge via RAG...")
    rag = MedicalKnowledgeRAG()
    rag_context = rag.retrieve_knowledge(reports, n_docs=5)
    print(f"✅ Retrieved {len(rag_context)} chars of medical knowledge")
    
    # Step 3: Generate trajectories with LLM
    print("\n[3/4] Generating trajectories with LLM (Outline-guided)...")
    generator = DirectLLMGenerator()
    result = generator.generate(
        patient_id='DEMO_PATIENT',
        reports=reports,
        rag_context=rag_context
    )
    
    # Step 4: Save results
    print("\n[4/4] Saving trajectories...")
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'direct_trajectories.json'
    
    output_data = {
        'metadata': {
            'approach': 'direct_llm',
            'radgraph_used': False,
            'n_patients': 1,
            'n_reports': len(reports),
            'n_trajectories': len(result.get('trajectories', []))
        },
        'trajectories': {
            result['patient_id']: result.get('trajectories', [])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\nGenerated: {len(result.get('trajectories', []))} trajectories")
    print(f"Method: Direct LLM (no RadGraph)")
    print(f"Output: {output_file}")
    
    print("\n💡 Next steps:")
    print("  1. Evaluate: python evaluate_to_csv.py (update to use direct_trajectories.json)")
    print("  2. Compare performance vs RadGraph-based approach")
    print("  3. Generate Figure 1 & Figure 2")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

