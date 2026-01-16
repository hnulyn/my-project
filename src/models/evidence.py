"""
Evidence data model
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Evidence:
    """Evidence data model"""
    
    question: str
    answer: str
    url: str
    answer_type: Optional[str] = None
    boolean_explanation: Optional[str] = None
    source_medium: Optional[str] = None
    cached_source_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "question": self.question,
            "answer": self.answer,
            "url": self.url,
            "answer_type": self.answer_type,
            "boolean_explanation": self.boolean_explanation,
            "source_medium": self.source_medium,
            "cached_source_url": self.cached_source_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evidence':
        """Create Evidence instance from dictionary"""
        return cls(
            question=data["question"],
            answer=data["answer"],
            url=data["url"],
            answer_type=data.get("answer_type"),
            boolean_explanation=data.get("boolean_explanation"),
            source_medium=data.get("source_medium"),
            cached_source_url=data.get("cached_source_url")
        )
    
    def get_full_text(self) -> str:
        """Get full evidence text (question + answer)"""
        text = f"{self.question} {self.answer}"
        if self.answer_type == "Boolean" and self.boolean_explanation:
            text += f". {self.boolean_explanation}"
        return text
    
    def format_for_display(self, evidence_id: int) -> str:
        """Format evidence text for display"""
        return (
            f"Evidence_ID: {evidence_id}\n"
            f"Content:{{Question: {self.question}\n"
            f"Answer: {self.answer}}}\n"
        )


class EvidenceSet:
    """Evidence set management class"""
    
    def __init__(self, evidences: List[Evidence] = None):
        self.evidences = evidences or []
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence"""
        self.evidences.append(evidence)
    
    def remove_duplicates(self):
        """Remove duplicate evidence (based on answer content)"""
        seen_answers = set()
        unique_evidences = []
        
        for evidence in self.evidences:
            if evidence.answer not in seen_answers:
                seen_answers.add(evidence.answer)
                unique_evidences.append(evidence)
        
        self.evidences = unique_evidences
    
    def to_text(self) -> str:
        """Convert to text format for prompts"""
        return "\n".join([
            evidence.format_for_display(i) 
            for i, evidence in enumerate(self.evidences)
        ])
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to dictionary list"""
        return [evidence.to_dict() for evidence in self.evidences]
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> 'EvidenceSet':
        """Create EvidenceSet from dictionary list"""
        evidences = [Evidence.from_dict(item) for item in data]
        return cls(evidences)
    
    def __len__(self):
        return len(self.evidences)
    
    def __iter__(self):
        return iter(self.evidences)
    
    def __getitem__(self, index):
        return self.evidences[index]
