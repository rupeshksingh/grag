# utils/azure_helpers.py
from typing import Dict, Any, List
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import AzureError

from config.logging_config import logger
from exceptions.custom_exceptions import AzureClientError
from models.document import Table

class AzureDocumentProcessor:
    """Helper class for Azure Form Recognizer operations"""
    
    def __init__(self, endpoint: str, key: str):
        """Initialize Azure Form Recognizer client"""
        try:
            self.client = DocumentAnalysisClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key)
            )
            logger.info("Azure Document Processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client: {str(e)}")
            raise AzureClientError(f"Azure client initialization failed: {str(e)}")

    def analyze_document(self, doc_file) -> Any:
        """Analyze a document using Form Recognizer
        
        Args:
            doc_file: File object of the document
            
        Returns:
            Azure analysis result
        """
        try:
            poller = self.client.begin_analyze_document(
                "prebuilt-document",
                doc_file
            )
            return poller.result()
        except AzureError as e:
            logger.error(f"Azure service error analyzing document: {str(e)}")
            raise AzureClientError(f"Azure service error: {str(e)}")
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            raise AzureClientError(f"Document analysis failed: {str(e)}")

    def process_tables(self, result: Any, page_num: int) -> List[Table]:
        """Process tables from Azure analysis result for a specific page"""
        tables = []
        for table in result.tables:
            if table.bounding_regions[0].page_number == page_num:
                # Reconstruct the table structure from cells
                table_content = []
                rows = set([cell.row_index for cell in table.cells])
                cols = set([cell.column_index for cell in table.cells])

                for row_index in sorted(list(rows)):
                    row_content = []
                    for col_index in sorted(list(cols)):
                        cell_content = next(
                            (cell.content for cell in table.cells 
                             if cell.row_index == row_index and 
                             cell.column_index == col_index), 
                            ""
                        )
                        row_content.append(cell_content)
                    table_content.append(row_content)

                tables.append(Table(
                    content=table_content,
                    page_number=page_num,
                    coordinates={
                        'x': table.bounding_regions[0].polygon[0].x,
                        'y': table.bounding_regions[0].polygon[0].y,
                        'width': table.bounding_regions[0].polygon[2].x - table.bounding_regions[0].polygon[0].x,
                        'height': table.bounding_regions[0].polygon[2].y - table.bounding_regions[0].polygon[0].y
                    }
                ))
        
        return tables

    def get_page_text(self, page: Any) -> str:
        """Extract text content from a page"""
        return " ".join(line.content for line in page.lines)

    def get_document_language(self, result: Any) -> str:
        """Get primary language from analysis result"""
        return result.languages[0] if result.languages else "unknown"

    def get_confidence_score(self, result: Any) -> float:
        """Get confidence score from analysis result"""
        return result.confidence if hasattr(result, 'confidence') else 1.0