from typing import List, Tuple, Dict, Any
from neo4j import GraphDatabase, Session, Transaction
from neo4j.exceptions import Neo4jError as BaseNeo4jError
from tqdm import tqdm
import json
from datetime import datetime

from config.logging_config import logger
from models.document import Chunk
from exceptions.custom_exceptions import Neo4jError

class Neo4jManager:
    """Manages Neo4j database operations with robust relationship handling"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50
            )
            self.database = database
            self._setup_database()
            logger.info("Neo4j connection established and verified")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise Neo4jError(f"Neo4j connection failed: {str(e)}")

    def _setup_database(self):
        """Set up database schema and constraints"""
        with self.driver.session(database=self.database) as session:
            try:
                # Create constraints
                session.run("""
                    CREATE CONSTRAINT chunk_id IF NOT EXISTS
                    FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE
                """)

                # Create composite index for faster relationship queries
                session.run("""
                    CREATE INDEX chunk_composite IF NOT EXISTS
                    FOR (n:Chunk)
                    ON (n.filename, n.doc_type, n.page_num)
                """)

                logger.info("Database schema setup completed")
            except Exception as e:
                logger.error(f"Schema setup failed: {str(e)}")
                raise Neo4jError(f"Database setup failed: {str(e)}")

    def _create_node_batch(self, tx: Transaction, chunks: List[Chunk]):
        """Create a batch of nodes"""
        query = """
        UNWIND $chunks AS chunk
        MERGE (n:Chunk {chunk_id: chunk.chunk_id})
        ON CREATE SET
            n.text = chunk.text,
            n.doc_type = chunk.doc_type,
            n.page_num = chunk.page_num,
            n.filename = chunk.filename,
            n.properties = chunk.properties,
            n.confidence = chunk.confidence,
            n.created_at = chunk.created_at
        """

        chunk_params = [{
            'chunk_id': chunk.chunk_id,
            'text': chunk.text,
            'doc_type': chunk.doc_metadata.doc_type.value,
            'page_num': chunk.page_num,
            'filename': chunk.doc_metadata.filename,
            'properties': json.dumps(chunk.properties),
            'confidence': float(chunk.confidence),
            'created_at': datetime.now().isoformat()
        } for chunk in chunks]

        return tx.run(query, chunks=chunk_params)

    def bulk_create_nodes(self, chunks: List[Chunk], batch_size: int = 100):
        """Create nodes in batches with progress tracking"""
        logger.info(f"Starting bulk node creation for {len(chunks)} chunks")

        if not chunks:
            logger.warning("No chunks provided for node creation")
            return

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="Creating nodes") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                try:
                    with self.driver.session(database=self.database) as session:
                        session.execute_write(self._create_node_batch, batch)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error creating node batch {i//batch_size}: {str(e)}")
                    continue

    def _create_relationship_batch(self, tx: Transaction, relationships: List[Dict]):
        """Create a batch of relationships"""
        query = """
        UNWIND $rels AS rel
        MATCH (source:Chunk {chunk_id: rel.source_id})
        MATCH (target:Chunk {chunk_id: rel.target_id})
        MERGE (source)-[r:RELATES {type: rel.rel_type}]->(target)
        ON CREATE SET
            r.similarity = rel.similarity,
            r.created_at = rel.created_at
        """

        result = tx.run(query, rels=relationships)
        return result.single()[0]

    def bulk_create_relationships(self, relationships: List[Tuple], batch_size: int = 500):
        """Bulk create relationships with strength and confidence scores"""
        logger.info("Starting bulk relationship creation")

        # Prepare relationship parameters
        rel_params = []
        for chunk1, chunk2, rel_type, direction, similarity, strength, confidence in relationships:
            try:
                params = {
                    "source_id": chunk1.chunk_id,
                    "target_id": chunk2.chunk_id,
                    "rel_type": rel_type,
                    "similarity": similarity,
                    "strength": strength,
                    "confidence": confidence,
                    "created_at": datetime.now().isoformat()
                }
                rel_params.append(params)

                if direction == "bidirectional":
                    reverse_params = params.copy()
                    reverse_params.update({
                        "source_id": chunk2.chunk_id,
                        "target_id": chunk1.chunk_id
                    })
                    rel_params.append(reverse_params)

            except Exception as e:
                logger.error(f"Error preparing relationship parameters: {str(e)}")
                continue

        # Process relationships in batches
        total_batches = (len(rel_params) + batch_size - 1) // batch_size
        with tqdm(total=total_batches, desc="Creating relationships") as pbar:
            for i in range(0, len(rel_params), batch_size):
                batch = rel_params[i:i + batch_size]

                query = """
                UNWIND $batch AS row
                MATCH (source:Chunk {chunk_id: row.source_id})
                MATCH (target:Chunk {chunk_id: row.target_id})
                MERGE (source)-[r:RELATES {type: row.rel_type}]->(target)
                ON CREATE SET
                    r.similarity = row.similarity,
                    r.strength = row.strength,
                    r.confidence = row.confidence,
                    r.created_at = row.created_at
                """

                try:
                    with self.driver.session(database=self.database) as session:
                         session.execute_write(lambda tx: tx.run(query, batch=batch))
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error creating batch of relationships: {str(e)}")
                    continue

        logger.info(f"Completed creating {len(rel_params)} relationships")

    def close(self):
        """Close database connection"""
        try:
            self.driver.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")