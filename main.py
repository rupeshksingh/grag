# main.py
import os
import asyncio
from dotenv import load_dotenv
import argparse
from pathlib import Path

from config.logging_config import logger
from core.document_processor import DocumentProcessor
from core.relationship_manager import RelationshipManager
from core.graph_builder import GraphBuilder
from utils.neo4j_helpers import Neo4jManager
from exceptions.custom_exceptions import TenderKGException

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Build knowledge graph from tender documents'
    )
    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help='Path to folder containing tender documents'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Batch size for processing (default: 5)'
    )
    return parser.parse_args()

async def main():
    """Main application entry point"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load environment variables
        load_dotenv()
        
        # Validate input folder
        input_folder = Path(args.input_folder)
        if not input_folder.exists() or not input_folder.is_dir():
            raise ValueError(f"Input folder does not exist: {input_folder}")
            
        # Initialize components
        logger.info("Initializing components")
        
        doc_processor = DocumentProcessor(
            azure_endpoint=os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"],
            azure_key=os.environ["AZURE_FORM_RECOGNIZER_KEY"],
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
        
        relationship_manager = RelationshipManager(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            batch_size=args.batch_size
        )
        
        neo4j_manager = Neo4jManager(
            uri=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"]
        )
        
        # Initialize graph builder
        graph_builder = GraphBuilder(
            doc_processor=doc_processor,
            relationship_manager=relationship_manager,
            neo4j_manager=neo4j_manager
        )
        
        logger.info("Starting knowledge graph construction")
        
        # Build knowledge graph
        await graph_builder.build_graph(str(input_folder))
        
        logger.info("Knowledge graph construction completed successfully")
        
    except TenderKGException as e:
        logger.error(f"Application error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        # Clean up resources
        logger.info("Cleaning up resources")
        try:
            graph_builder.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        exit(1)