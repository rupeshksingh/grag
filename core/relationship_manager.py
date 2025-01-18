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
    """Manages relationship detection between financial document chunks"""
    
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
        logger.info("RelationshipManager initialized successfully")

    async def extract_properties(self, chunk: Chunk) -> Dict[str, Any]:
        """Extract properties from a chunk using LLM"""
        logger.debug(f"Extracting properties for chunk {chunk.chunk_id}")

        system_template = """Analyze the following financial document text and extract key information into these categories:
        - entities: Named entities (organizations, people, products)
        - technical_terms: Technical or domain-specific terminology
        - requirements: Explicit requirements or specifications
        - dependencies: References to other components or systems
        - dates: Any dates or time references
        - monetary_values: Any financial figures or costs
        - stakeholders: People or groups involved
        - key_phrases: Important phrases or concepts

        Return ONLY items that are explicitly mentioned in the text. If a category has no items, return an empty list."""

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
        """Determine relationship between financial document chunks"""
        logger.debug(f"Determining relationship between chunks {chunk1.chunk_id} and {chunk2.chunk_id}")

        system_template = """Analyze the relationship between these financial document chunks and determine:

        - Relationship Type: Choose the most appropriate from the list below:
            BUDGETS: Budget-related dependencies
            DEPENDS_ON: Process/approval dependencies
            EVALUATES: Evaluation criteria relationships
            SCHEDULES: Timeline/scheduling relationships
            COMPLEMENTS: Complementary information
            AUTHORIZES: Authorization/approval relationships
            FULFILLS: Requirement fulfillment
            RELATES_TO: General relation

        - Direction: Choose one:
            unidirectional: One chunk affects/influences the other
            bidirectional: Both chunks affect/influence each other

        - Confidence: Provide a value between 0 and 1 indicating your confidence in this relationship.

        Consider the provided similarity score and properties when determining the relationship.

        Text 1 Properties: {chunk1_properties}
        Text 1: {text1}

        Text 2 Properties: {chunk2_properties}
        Text 2: {text2}
        
        Similarity Score: {similarity}"""

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

            # Convert string responses to enum values
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