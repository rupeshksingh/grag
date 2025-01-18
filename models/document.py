# models/document.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
from pathlib import Path

from .enums import DocumentType

@dataclass
class Table:
    """Represents a table extracted from a document"""
    content: List[List[str]]
    page_number: int
    coordinates: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary representation"""
        return {
            'rows': self.rows,
            'page_number': self.page_number,
            'coordinates': self.coordinates
        }

@dataclass
class Page:
    """Represents a single page in a document"""
    page_num: int
    text: str
    tables: List[Table]
    # words: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def get_text_without_tables(self) -> str:
        """Return page text excluding table content"""
        # Implementation would remove table text from page text
        return self.text

@dataclass
class DocumentMetadata:
    """Metadata for a document"""
    filename: str
    page_count: int
    language: str
    doc_type: DocumentType
    processed_date: datetime = field(default_factory=datetime.now)
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation"""
        return {
            'filename': self.filename,
            'file_path': str(self.file_path),
            'page_count': self.page_count,
            'language': self.language,
            'doc_type': self.doc_type.value,
            'file_size': self.file_size,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'processed_date': self.processed_date.isoformat(),
            'confidence_score': self.confidence_score
        }

@dataclass
class Document:
    """Represents a processed document"""
    metadata: DocumentMetadata
    pages: List[Page]
    raw_content: Optional[bytes] = None
    
    def get_text(self) -> str:
        """Get full document text"""
        return '\n'.join(page.text for page in self.pages)
    
    def get_page(self, page_num: int) -> Optional[Page]:
        """Get specific page by number"""
        try:
            return next(page for page in self.pages if page.page_num == page_num)
        except StopIteration:
            return None

@dataclass
class DocumentFeatures:
    """Features extracted from document for type detection"""
    technical_terms: Set[str] = field(default_factory=set)
    financial_terms: Set[str] = field(default_factory=set)
    legal_terms: Set[str] = field(default_factory=set)
    requirement_indicators: Set[str] = field(default_factory=set)
    evaluation_indicators: Set[str] = field(default_factory=set)
    compliance_indicators: Set[str] = field(default_factory=set)
    timeline_indicators: Set[str] = field(default_factory=set)
    
    def merge(self, other: 'DocumentFeatures') -> 'DocumentFeatures':
        """Merge features with another DocumentFeatures instance"""
        return DocumentFeatures(
            technical_terms=self.technical_terms.union(other.technical_terms),
            financial_terms=self.financial_terms.union(other.financial_terms),
            legal_terms=self.legal_terms.union(other.legal_terms),
            requirement_indicators=self.requirement_indicators.union(other.requirement_indicators),
            evaluation_indicators=self.evaluation_indicators.union(other.evaluation_indicators),
            compliance_indicators=self.compliance_indicators.union(other.compliance_indicators),
            timeline_indicators=self.timeline_indicators.union(other.timeline_indicators)
        )

@dataclass
class Chunk:
    """Represents a semantic chunk of text with its properties"""
    text: str
    properties: Dict[str, Any]
    page_num: int
    doc_metadata: DocumentMetadata
    chunk_id: str = field(default_factory=lambda: f"chunk_{datetime.now().timestamp()}")
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'properties': self.properties,
            'page_num': self.page_num,
            'doc_metadata': self.doc_metadata.to_dict(),
            'confidence': self.confidence
        }