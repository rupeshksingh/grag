# config/constants.py
from typing import Dict, Set
from models.enums import DocumentType

SIMILARITY_THRESHOLD = 0.75
BATCH_SIZE = 5

DOCUMENT_FEATURES = {
    "technical_terms": {
        "specification", "technical", "system", "architecture", "interface",
        "implementation", "configuration", "infrastructure", "integration",
        "requirements", "performance", "scalability", "reliability"
    },
    "financial_terms": {
        "budget", "cost", "price", "payment", "financial", "expense",
        "ROI", "investment", "funding", "monetary", "billing", "invoice"
    },
    "legal_terms": {
        "agreement", "contract", "clause", "party", "liability", "terms",
        "conditions", "compliance", "regulation", "statutory", "legal"
    },
    "requirement_indicators": {
        "must", "shall", "should", "required", "mandatory", "essential",
        "necessary", "needed", "crucial", "critical"
    },
    "evaluation_indicators": {
        "criteria", "evaluation", "assessment", "scoring", "rating",
        "benchmark", "threshold", "minimum", "maximum", "acceptable"
    },
    "compliance_indicators": {
        "compliance", "regulatory", "standard", "regulation", "guideline",
        "policy", "procedure", "protocol", "requirement", "certification"
    },
    "timeline_indicators": {
        "schedule", "timeline", "deadline", "milestone", "phase",
        "duration", "period", "date", "delivery", "implementation"
    }
}

RELATIONSHIP_PATTERNS = {
    "REQUIRES": {
        "patterns": ["requires", "depends on", "needs", "prerequisite"],
        "bidirectional": False
    },
    "REFERENCES": {
        "patterns": ["refers to", "references", "related to", "see also"],
        "bidirectional": True
    },
    "SPECIFIES": {
        "patterns": ["specifies", "defines", "details", "elaborates"],
        "bidirectional": False
    },
    "CONSTRAINS": {
        "patterns": ["limits", "constrains", "restricts", "bounds"],
        "bidirectional": False
    },
    "IMPLEMENTS": {
        "patterns": ["implements", "fulfills", "satisfies", "meets"],
        "bidirectional": False
    },
    "CONFLICTS": {
        "patterns": ["conflicts with", "contradicts", "incompatible"],
        "bidirectional": True
    }
}

type_mapping = {
            "technical specification": DocumentType.TECHNICAL_SPECIFICATION,
            "requirement": DocumentType.REQUIREMENTS,
            "proposal": DocumentType.PROPOSAL,
            "contract": DocumentType.CONTRACT,
            "financial": DocumentType.FINANCIAL,
            "legal": DocumentType.LEGAL,
            "compliance": DocumentType.COMPLIANCE,
            "evaluation criteria": DocumentType.EVALUATION_CRITERIA,
            "timeline": DocumentType.TIMELINE
        }