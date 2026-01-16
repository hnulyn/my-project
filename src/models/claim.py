"""
Claim data model
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .evidence import EvidenceSet


@dataclass
class Claim:
    """Claim data model"""
    
    claim_id: int
    claim: str
    speaker: Optional[str] = None
    label: Optional[str] = None
    pred_label: Optional[str] = None
    claim_types: Optional[List[str]] = None
    evidence_set: Optional[EvidenceSet] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        data = {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "speaker": self.speaker,
            "label": self.label,
            "pred_label": self.pred_label,
            "claim_types": self.claim_types
        }
        
        if self.evidence_set:
            data["evidence"] = self.evidence_set.to_list()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Claim':
        """Create Claim instance from dictionary"""
        evidence_set = None
        
        if "evidence" in data:
            evidence_set = EvidenceSet.from_list(data["evidence"])
        elif "questions" in data:
            evidences = []
            for question_data in data["questions"]:
                answers = question_data.get("answers", [])
                if not isinstance(answers, list):
                    answers = [answers]
                
                for answer in answers:
                    evidence = {
                        "question": question_data["question"],
                        "answer": answer.get("answer", "No answer could be found."),
                        "url": answer.get("source_url", "None"),
                        "answer_type": answer.get("answer_type"),
                        "boolean_explanation": answer.get("boolean_explanation"),
                        "source_medium": answer.get("source_medium"),
                        "cached_source_url": answer.get("cached_source_url")
                    }
                    evidences.append(evidence)
                    
                if not answers:
                    evidences.append({
                        "question": question_data["question"],
                        "answer": "No answer could be found.",
                        "url": "None"
                    })
            
            evidence_set = EvidenceSet.from_list(evidences)
        
        return cls(
            claim_id=data["claim_id"],
            claim=data["claim"],
            speaker=data.get("speaker"),
            label=data.get("label"),
            pred_label=data.get("pred_label"),
            claim_types=data.get("claim_types"),
            evidence_set=evidence_set
        )
    
    def get_claim_context(self) -> str:
        """Get claim context text"""
        context = f"Claim: {self.claim}"
        if self.speaker:
            context += f"\nSpeaker: {self.speaker}"
        return context
    
    def get_evidence_text(self) -> str:
        """Get evidence text"""
        if self.evidence_set:
            return self.evidence_set.to_text()
        return ""
    
    def has_evidence(self) -> bool:
        """Check if has evidence"""
        return self.evidence_set is not None and len(self.evidence_set) > 0


VERDICT_LABELS = [
    "Supported",
    "Refuted", 
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking"
]


def normalize_label(label: Any) -> str:
    """Normalize label format"""
    try:
        if isinstance(label, list):
            label = label[0] if label else ""
        label = str(label).lower().strip()
    except:
        return "Not Enough Evidence"
    if any(x in label for x in ["not enough evidence", "insufficient evidence"]):
        return "Not Enough Evidence"
    elif any(x in label for x in ["conflicting evidence", "cherrypicking", "cherry-picking", "cherry picking", "conflicting"]):
        return "Conflicting Evidence"
    elif "support" in label:
        return "Supported"
    elif "refut" in label:
        return "Refuted"
    else:
        return label if label in VERDICT_LABELS else "Not Enough Evidence"
