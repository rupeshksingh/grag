# core/graph_builder.py
from typing import List
from tqdm import tqdm
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.logging_config import logger
from models.document import Document, Chunk
from core.document_processor import DocumentProcessor
from core.relationship_manager import RelationshipManager
from utils.neo4j_helpers import Neo4jManager
from exceptions.custom_exceptions import GraphBuildingError

class GraphBuilder:
    """Manages the complete process of building the knowledge graph"""

    def __init__(self,
                 doc_processor: DocumentProcessor,
                 relationship_manager: RelationshipManager,
                 neo4j_manager: Neo4jManager,
                 max_concurrent_files: int = 3):
        self.doc_processor = doc_processor
        self.relationship_manager = relationship_manager
        self.neo4j_manager = neo4j_manager
        self.max_concurrent_files = max_concurrent_files
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_files)
        logger.info("GraphBuilder initialized")

    async def process_single_document(self, file_path: str) -> Document:
        """Process a single document"""
        try:
            document = await self.doc_processor.process_document(file_path)
            logger.info(f"Processed document: {file_path}")
            return document
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            raise

    async def process_documents(self, folder_path: str) -> List[Document]:
        """Process all documents in the folder with concurrency control"""
        try:
            files = [f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.pdf', '.docx', '.doc'))]
            
            documents = []
            semaphore = asyncio.Semaphore(self.max_concurrent_files)
            
            async def process_with_semaphore(filename):
                async with semaphore:
                    file_path = os.path.join(folder_path, filename)
                    return await self.process_single_document(file_path)
            
            # Process documents concurrently with progress bar
            tasks = [process_with_semaphore(filename) for filename in files]
            for task in tqdm(asyncio.as_completed(tasks), 
                           total=len(tasks), 
                           desc="Processing documents"):
                document = await task
                documents.append(document)
                
            logger.info(f"Processed {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise GraphBuildingError(f"Failed to process documents: {str(e)}")

    async def create_chunks(self, documents: List[Document]) -> List[Chunk]:
        """Create chunks from all documents with batched processing"""
        logger.info("Creating chunks from documents")
        all_chunks = []
        
        # Process documents in batches to prevent memory issues
        batch_size = 5
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in tqdm(batch, desc=f"Creating chunks (batch {i//batch_size + 1})"):
                chunks = self.doc_processor.create_chunks(doc)
                all_chunks.extend(chunks)
                
            # Optional: Add a small delay between batches to prevent resource exhaustion
            await asyncio.sleep(0.1)

        logger.info(f"Created {len(all_chunks)} chunks total")
        return all_chunks

    def enhance_relationships(self, relationships):
        """Enhance relationship tuples to match Neo4j manager expectations"""
        enhanced_relationships = []
        for chunk1, chunk2, rel_type, direction, similarity, confidence in relationships:
            if similarity > 0.7:
                strength = "strong"
            elif similarity > 0.4:
                strength = "moderate"
            else:
                strength = "weak"
                
            enhanced_rel = (
                chunk1,
                chunk2,
                rel_type,
                direction,
                similarity,
                strength,
                confidence
            )
            enhanced_relationships.append(enhanced_rel)
        return enhanced_relationships

    async def process_chunks(self, chunks: List[Chunk]):
        """Process all chunks and create graph with batched processing"""
        logger.info("Processing chunks and building graph")

        try:
            # Process properties in batches
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Extract properties for current batch
                for chunk in tqdm(batch, 
                                desc=f"Extracting properties (batch {i//batch_size + 1})"):
                    await self.relationship_manager.extract_properties(chunk)
                
                # Optional: Add a small delay between batches
                await asyncio.sleep(0.1)

            # Create nodes in Neo4j in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.neo4j_manager.bulk_create_nodes(batch)
                await asyncio.sleep(0.1)

            # Process relationships
            relationships = await self.relationship_manager.process_chunk_relationships(
                chunks
            )

            # Enhance relationships with strength before creating in Neo4j
            enhanced_relationships = self.enhance_relationships(relationships)

            # Create relationships in Neo4j in batches
            batch_size = 100  # Larger batch size for relationships
            for i in range(0, len(enhanced_relationships), batch_size):
                batch = enhanced_relationships[i:i + batch_size]
                self.neo4j_manager.bulk_create_relationships(batch)
                await asyncio.sleep(0.1)

            logger.info("Graph building completed successfully")

        except Exception as e:
            logger.error(f"Chunk processing failed: {str(e)}")
            raise GraphBuildingError(f"Failed to process chunks: {str(e)}")

    async def build_graph(self, folder_path: str):
        """Main method to build the complete knowledge graph"""
        logger.info(f"Starting graph building process for {folder_path}")

        try:
            # Process documents
            documents = await self.process_documents(folder_path)

            # Create and process chunks
            all_chunks = []
            batch_size = 5
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                for doc in batch:
                    chunks = self.doc_processor.create_chunks(doc)
                    
                    # Extract properties for chunks in current batch
                    for chunk in chunks:
                        properties = await self.relationship_manager.extract_properties(chunk)
                        all_chunks.append(chunk)
                
                # Optional: Add a small delay between batches
                await asyncio.sleep(0.1)

            # Create nodes in batches
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                self.neo4j_manager.bulk_create_nodes(batch)
                await asyncio.sleep(0.1)

            # Process relationships
            relationships = await self.relationship_manager.process_chunk_relationships(
                all_chunks
            )

            # Enhance relationships with strength before creating in Neo4j
            enhanced_relationships = self.enhance_relationships(relationships)

            # Create relationships in batches
            rel_batch_size = 100
            for i in range(0, len(enhanced_relationships), rel_batch_size):
                batch = enhanced_relationships[i:i + rel_batch_size]
                self.neo4j_manager.bulk_create_relationships(batch)
                await asyncio.sleep(0.1)

            logger.info("Knowledge graph built successfully")

        except Exception as e:
            logger.error(f"Graph building failed: {str(e)}")
            raise GraphBuildingError(f"Failed to build graph: {str(e)}")

    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        self.neo4j_manager.close()