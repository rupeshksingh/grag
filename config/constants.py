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

type_descriptions = {
            DocumentType.TECHNICAL_SPECIFICATION: """
                Technical specifications and detailed system requirements.
                Contains technical details, system architecture, performance requirements,
                functionality specifications, integration requirements, and technical standards.
            """,
            DocumentType.REQUIREMENTS: """
                Requirements documentation detailing project needs and expectations.
                Includes functional requirements, business requirements, user requirements,
                system requirements, and acceptance criteria.
            """,
            DocumentType.PROPOSAL: """
                Project or solution proposals and bidding documents.
                Contains proposed solutions, methodologies, implementation approaches,
                team structure, pricing proposals, and company capabilities.
            """,
            DocumentType.CONTRACT: """
                Legal contracts and agreements between parties.
                Includes terms and conditions, service level agreements, 
                contractual obligations, payment terms, and legal requirements.
            """,
            DocumentType.FINANCIAL: """
                Financial documentation and pricing information.
                Contains budgets, cost breakdowns, pricing schedules, payment terms,
                financial projections, and cost estimates.
            """,
            DocumentType.LEGAL: """
                Legal documentation and compliance requirements.
                Includes legal terms, regulatory requirements, liability clauses,
                intellectual property rights, and legal obligations.
            """,
            DocumentType.COMPLIANCE: """
                Compliance documentation and regulatory requirements.
                Contains compliance standards, regulatory frameworks, certification requirements,
                audit requirements, and compliance procedures.
            """,
            DocumentType.EVALUATION_CRITERIA: """
                Evaluation and assessment criteria documentation.
                Includes scoring criteria, evaluation methodologies, assessment frameworks,
                selection criteria, and performance metrics.
            """,
            DocumentType.TIMELINE: """
                Project timeline and schedule documentation.
                Contains project schedules, milestones, delivery timelines,
                implementation phases, and project roadmaps.
            """
        }