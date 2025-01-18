from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from sklearn.metrics.pairwise import cosine_similarity

from config.logging_config import logger
from models.document import Chunk
from exceptions.custom_exceptions import PropertyExtractionError
from models.enums import RelationType, Direction, RelationshipOutput, Properties

class RelationshipManager:
    """Manages relationship detection between tender document chunks"""
    
    def __init__(self,
                 openai_api_key: str,
                 anthropic_api_key: str,
                 batch_size: int = 5):
        """Initialize relationship manager"""
        self.chat_model = ChatOpenAI(
            api_key=openai_api_key,
            temperature=0,
            model="chatgpt-4o-latest"
        )
        self.batch_size = batch_size
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        logger.info("TenderRelationshipManager initialized successfully")

    async def extract_properties(self, chunk: Chunk) -> Dict[str, Any]:
        """Extract properties from a tender document chunk using LLM"""
        logger.debug(f"Extracting properties for chunk {chunk.chunk_id}")

        system_template = """Analyze the following tender document text and extract key information into these categories. Consider the document type context when extracting information.

        Categories to extract:
        - entities: Organizations, companies, departments, or individuals mentioned
        - technical_terms: Industry-specific terminology, standards, or technical specifications
        - requirements: 
            * Mandatory requirements or specifications
            * Minimum qualifications
            * Compliance requirements
            * Technical requirements
            * Service level agreements
        - dependencies: 
            * Prerequisites or conditions
            * Referenced documents or sections
            * Sequential processes
            * Approval dependencies
        - dates: 
            * Submission deadlines
            * Project milestones
            * Contract periods
            * Implementation timelines
        - monetary_values: 
            * Budget allocations
            * Cost estimates
            * Financial requirements
            * Pricing details
        - stakeholders: 
            * Decision makers
            * Evaluators
            * Required personnel
            * Contact points
        - key_phrases: 
            * Critical success factors
            * Evaluation criteria
            * Strategic objectives
            * Core deliverables

        Important guidelines:
        1. Extract ONLY explicitly mentioned items
        2. Maintain the exact terminology used in the text
        3. For each item, include any associated qualifiers or conditions
        4. If a category has no explicit mentions, return an empty list
        5. Pay special attention to mandatory requirements and dependencies

        Consider how this chunk might relate to other tender documents when identifying key elements."""

        try:
            parser = PydanticOutputParser(pydantic_object=Properties)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template + "\n{format_instructions}"),
                ("human", "{text}")
            ]).partial(format_instructions=parser.get_format_instructions())

            chain = prompt | self.chat_model | parser
            response = await chain.ainvoke({"text": chunk.text})
            
            properties = response.dict()
            chunk.properties = properties
            return properties
            
        except Exception as e:
            logger.error(f"Property extraction failed for chunk {chunk.chunk_id}: {str(e)}")
            raise PropertyExtractionError(f"Failed to extract properties: {str(e)}")

    async def determine_relationship(self,
                                  chunk1: Chunk,
                                  chunk2: Chunk,
                                  similarity: float) -> Tuple[RelationType, Direction, float]:
        """Determine relationship between tender document chunks"""
        logger.debug(f"Determining relationship between chunks {chunk1.chunk_id} and {chunk2.chunk_id}")

        system_template = """Analyze the relationship between these tender document chunks and determine their connection. Consider the document types and context carefully.

        Relationship Types (Choose the most appropriate):
        1. BUDGETS:
           - Financial dependencies or implications
           - Budget allocations and constraints
           - Cost-related dependencies

        2. DEPENDS_ON:
           - Sequential process requirements
           - Prerequisite conditions
           - Technical dependencies
           - Approval workflows

        3. EVALUATES:
           - Assessment criteria relationships
           - Scoring or rating connections
           - Performance measurement links
           - Quality assurance relationships

        4. SCHEDULES:
           - Timeline dependencies
           - Milestone relationships
           - Deadline connections
           - Project phase linkages

        5. COMPLEMENTS:
           - Supporting information
           - Additional details or clarifications
           - Related specifications
           - Supplementary requirements

        6. AUTHORIZES:
           - Approval relationships
           - Sign-off requirements
           - Permission dependencies
           - Certification needs

        7. FULFILLS:
           - Requirement satisfaction
           - Compliance demonstration
           - Specification matching
           - Deliverable completion

        8. RELATES_TO:
           - General topical connections
           - Context relationships
           - Reference links
           - Informal associations

        Direction Analysis:
        - UNIDIRECTIONAL: One chunk directly influences or affects the other
        - BIDIRECTIONAL: Both chunks have mutual influence or dependency

        Confidence Scoring Factors:
        - Explicit references between chunks
        - Shared key terms or concepts
        - Temporal or logical sequence
        - Common stakeholders or requirements
        - Similarity score context
        - Document type relationship strength

        Document Properties Context:
        Text 1 Properties: {chunk1_properties}
        Text 1: {text1}

        Text 2 Properties: {chunk2_properties}
        Text 2: {text2}
        
        Similarity Score: {similarity}

        Return:
        1. Most appropriate relationship type
        2. Direction of influence
        3. Confidence score (0-1) based on evidence strength"""

        try:
            parser = PydanticOutputParser(pydantic_object=RelationshipOutput)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_template + "\n{format_instructions}"),
                ("human", "{text1}\n{text2}")
            ]).partial(format_instructions=parser.get_format_instructions())

            chain = prompt | self.chat_model | parser
            response = await chain.ainvoke({
                "chunk1_properties": chunk1.properties,
                "text1": chunk1.text,
                "chunk2_properties": chunk2.properties,
                "text2": chunk2.text,
                "similarity": similarity
            })

            rel_type = RelationType[response.rel_type.strip().upper()]
            direction = Direction[response.direction.strip().upper()]
            confidence = float(response.confidence_score)

            return rel_type, direction, confidence

        except Exception as e:
            logger.error(f"Relationship determination failed: {str(e)}")
            return RelationType.RELATES_TO, Direction.UNIDIRECTIONAL, 0.5

    def calculate_similarity(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """Calculate semantic similarity between chunks"""
        try:
            embedding1 = np.array(self.embeddings.embed_query(chunk1.text))
            embedding2 = np.array(self.embeddings.embed_query(chunk2.text))

            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)

            similarity = float(cosine_similarity(embedding1, embedding2)[0][0])
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0

    async def process_chunk_relationships(
        self,
        chunks: List[Chunk]
    ) -> List[Tuple[Chunk, Chunk, str, str, float, float]]:
        """Process relationships between all chunks"""
        logger.info("Processing chunk relationships")
        relationships = []
        
        total_comparisons = len(chunks) * (len(chunks) - 1) // 2
        pbar = tqdm(total=total_comparisons, desc="Processing relationships")

        try:
            for i, chunk1 in enumerate(chunks):
                for chunk2 in chunks[i+1:]:
                    try:
                        # Calculate similarity
                        similarity = self.calculate_similarity(chunk1, chunk2)
                        
                        # Determine relationship with similarity context
                        rel_type, direction, confidence = await self.determine_relationship(
                            chunk1, chunk2, similarity
                        )
                        
                        # Create relationship tuple
                        relationship = (
                            chunk1,
                            chunk2,
                            rel_type.value,
                            direction.value,
                            similarity,
                            confidence
                        )
                        relationships.append(relationship)
                        
                        logger.debug(
                            f"Created relationship: {rel_type} between "
                            f"{chunk1.chunk_id} and {chunk2.chunk_id} "
                            f"(similarity: {similarity:.2f}, confidence: {confidence:.2f})"
                        )
                        
                    except Exception as e:
                        logger.error(
                            f"Error processing relationship between chunks "
                            f"{chunk1.chunk_id} and {chunk2.chunk_id}: {str(e)}"
                        )
                        continue
                        
                    finally:
                        pbar.update(1)
                        
        except Exception as e:
            logger.error(f"Error in relationship processing: {str(e)}")
            raise
            
        finally:
            pbar.close()
            
        logger.info(f"Processed {len(relationships)} relationships")
        return relationships