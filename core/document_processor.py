# core/document_processor.py
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config.logging_config import logger
from models.document import Document, DocumentMetadata, Page, Chunk
from models.enums import DocumentType
from exceptions.custom_exceptions import DocumentProcessingError, AzureClientError
from config.constants import type_mapping, type_descriptions
from utils.azure_helpers import AzureDocumentProcessor

class DocumentProcessor:
    """Handles document processing and feature extraction"""
    
    def __init__(self, 
                 azure_endpoint: str,
                 azure_key: str,
                 openai_api_key: str):
        try:
            self.azure_processor = AzureDocumentProcessor(
                endpoint=azure_endpoint,
                key=azure_key
            )
            self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            self.text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_amount=95
            )
            logger.info("DocumentProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {str(e)}")
            raise AzureClientError(f"Failed to initialize Azure client: {str(e)}")

    def _cache_type_embeddings(self) -> None:
        """Pre-compute and cache embeddings for all document type descriptions"""
        logger.info("Caching document type embeddings")
        self.type_embeddings = {}
        
        try:
            for doc_type, description in self.type_descriptions.items():
                embedding = self.embeddings.embed_query(description)
                self.type_embeddings[doc_type] = np.array(embedding)
            logger.debug("Successfully cached document type embeddings")
        except Exception as e:
            logger.error(f"Failed to cache type embeddings: {str(e)}")
            raise

    def _detect_document_type(self, text: str) -> Tuple[DocumentType, float]:
        """
        Detect document type using semantic similarity with type descriptions
        Returns tuple of (DocumentType, confidence_score)
        """
        logger.debug("Detecting document type using semantic similarity")
        
        try:
            # Get embedding for input text
            text_embedding = np.array(self.embeddings.embed_query(text))
            
            # Calculate similarities with all type descriptions
            similarities = {}
            for doc_type, type_embedding in self.type_embeddings.items():
                similarity = float(cosine_similarity(
                    text_embedding.reshape(1, -1),
                    type_embedding.reshape(1, -1)
                )[0][0])
                similarities[doc_type] = similarity

            # Find best matching type and confidence score
            best_type = max(similarities.items(), key=lambda x: x[1])
            doc_type, confidence = best_type

            # If confidence is too low, fall back to GENERAL type
            if confidence < 0.3:  # Adjustable threshold
                logger.debug(f"Low confidence ({confidence:.2f}) for document type detection, using GENERAL")
                return DocumentType.GENERAL, confidence
                
            logger.debug(f"Detected document type {doc_type} with confidence {confidence:.2f}")
            return doc_type, confidence

        except Exception as e:
            logger.error(f"Error in document type detection: {str(e)}")
            return DocumentType.GENERAL, 0.0

    async def process_document(self, file_path: str) -> Document:
        """Process a single document and return structured data"""
        logger.info(f"Processing document: {file_path}")
        try:
            with open(file_path, 'rb') as doc_file:
                result = self.azure_processor.analyze_document(doc_file)

                pages = []
                for page_num, page in enumerate(tqdm(
                    result.pages,
                    desc="Processing pages",
                    leave=False
                ), 1):
                    # Process tables using Azure helper
                    tables = self.azure_processor.process_tables(result, page_num)
                    
                    # Create page with extracted content
                    page_text = self.azure_processor.get_page_text(page)
                    
                    pages.append(Page(
                        page_num=page_num,
                        text=page_text,
                        tables=tables,
                        confidence=page.confidence if hasattr(page, 'confidence') else 1.0
                    ))
                    logger.debug(f"Processed page {page_num}")

                # Create document metadata with enhanced type detection
                file_path_obj = Path(file_path)
                combined_text = " ".join(page.text for page in pages)
                doc_type, type_confidence = self._detect_document_type(combined_text)
                
                metadata = DocumentMetadata(
                    filename=file_path_obj.name,
                    page_count=len(result.pages),
                    language=self.azure_processor.get_document_language(result),
                    doc_type=doc_type,
                    doc_type_confidence=type_confidence,  # Added confidence score
                    processed_date=datetime.now(),
                    confidence_score=self.azure_processor.get_confidence_score(result)
                )

                return Document(
                    metadata=metadata,
                    pages=pages,
                )

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document {file_path}: {str(e)}")


    def create_chunks(self, document: Document) -> List[Chunk]:
        """Create chunks from document with validation"""
        logger.info(f"Creating chunks for document: {document.metadata.filename}")
        chunks = []
        
        # Get full document text
        doc_text = document.get_text()
        if not doc_text or not doc_text.strip():
            logger.warning(f"Empty document content detected: {document.metadata.filename}")
            return chunks

        try:
            # Split content into chunks
            raw_chunks = self.text_splitter.split_text(doc_text)
            
            # Create chunk objects
            for i, chunk_text in enumerate(raw_chunks):
                if chunk_text and chunk_text.strip():
                    chunk = Chunk(
                        text=chunk_text,
                        properties={},  # Will be populated by RelationshipManager
                        page_num=self._estimate_page_num(chunk_text, document.pages),
                        doc_metadata=document.metadata,
                        confidence=1.0
                    )
                    chunks.append(chunk)
                    
            logger.info(f"Created {len(chunks)} valid chunks for {document.metadata.filename}")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk creation failed for {document.metadata.filename}: {str(e)}")
            raise

    def _estimate_page_num(self, chunk_text: str, pages: List[Page]) -> int:
        """Estimate which page a chunk came from based on content similarity"""
        max_similarity = 0
        best_page = 1
        
        chunk_embedding = self.embeddings.embed_query(chunk_text)
        chunk_embedding = np.array(chunk_embedding)
        
        for page in pages:
            page_embedding = self.embeddings.embed_query(page.text)
            page_embedding = np.array(page_embedding)

            similarity = float(cosine_similarity(
                chunk_embedding.reshape(1, -1),
                page_embedding.reshape(1, -1)
            )[0][0])
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_page = page.page_num
                
        return best_page

    async def process_folder(self, folder_path: str) -> List[Document]:
        """Process all documents in a folder"""
        logger.info(f"Processing documents in folder: {folder_path}")
        documents = []
        
        # Get all document files
        files = [f for f in os.listdir(folder_path)
                if f.lower().endswith(('.pdf', '.docx', '.doc'))]
        
        # Process each file with progress bar
        for filename in tqdm(files, desc="Processing documents"):
            try:
                file_path = os.path.join(folder_path, filename)
                document = await self.process_document(file_path)
                documents.append(document)
            except DocumentProcessingError as e:
                logger.error(f"Skipping document {filename} due to error: {str(e)}")
                continue
                
        return documents