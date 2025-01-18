# exceptions/custom_exceptions.py
class TenderKGException(Exception):
    """Base exception for TenderKG"""
    pass

class DocumentProcessingError(TenderKGException):
    """Raised when there's an error processing a document"""
    pass

class GraphBuildingError(TenderKGException):
    """Raised when there's an error building the knowledge graph"""
    pass

class AzureClientError(TenderKGException):
    """Raised when there's an error with Azure Form Recognizer"""
    pass

class Neo4jError(TenderKGException):
    """Raised when there's an error with Neo4j operations"""
    pass

class PropertyExtractionError(TenderKGException):
    """Raised when there's an error extracting properties"""
    pass