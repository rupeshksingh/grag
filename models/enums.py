# models/enums.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class DocumentType(Enum):
    """Enumeration of possible document types in tender"""
    TECHNICAL_SPECIFICATION = "TechnicalSpec"
    REQUIREMENTS = "Requirements"
    PROPOSAL = "Proposal"
    CONTRACT = "Contract"
    FINANCIAL = "Financial"
    LEGAL = "Legal"
    COMPLIANCE = "Compliance"
    EVALUATION_CRITERIA = "EvaluationCriteria"
    TIMELINE = "Timeline"
    GENERAL = "General"

class RelationType(str, Enum):
    BUDGETS = "BUDGETS"
    DEPENDS_ON = "DEPENDS_ON"
    EVALUATES = "EVALUATES"
    SCHEDULES = "SCHEDULES"
    COMPLEMENTS = "COMPLEMENTS"
    AUTHORIZES = "AUTHORIZES"
    FULFILLS = "FULFILLS"
    RELATES_TO = "RELATES_TO"

class Direction(str, Enum):
    UNIDIRECTIONAL = "unidirectional"
    BIDIRECTIONAL = "bidirectional"

class RelationshipOutput(BaseModel):
    rel_type: str
    direction: str
    confidence_score: float

class Properties(BaseModel):
    entities: List[str] = Field(default_factory=list)
    technical_terms: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    monetary_values: List[str] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)
    key_phrases: List[str] = Field(default_factory=list)