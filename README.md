# Tender Knowledge Graph Pipeline

A sophisticated pipeline for creating knowledge graphs from tender documents using Azure Document Intelligence, LangChain, Neo4j, and various LLMs (OpenAI GPT and Anthropic Claude).

## Overview

This pipeline processes tender documents to create a rich knowledge graph that captures relationships between different sections, requirements, specifications, and dependencies. It uses semantic chunking and advanced NLP techniques to understand document context and create meaningful connections.

## Features

### Document Processing

- Multi-format support (PDF, DOCX, DOC)

- Page-level extraction and analysis

- Automatic document type detection

- Semantic text chunking

- Document feature extraction

### Knowledge Graph Creation

- Rich node properties with metadata

- Sophisticated relationship detection

- Confidence scoring for relationships

- Bidirectional and unidirectional relationships

- Batch processing for performance

### Relationship Types

- REQUIRES: Dependencies and prerequisites

- REFERENCES: Cross-references and mentions

- ELABORATES: Detailed explanations

- CONTRADICTS: Conflicting information

- SUPERSEDES: Updates and replacements

## Prerequisites

```bash

# Python packages

pip install azure-ai-formrecognizer

pip install langchain-experimental

pip install langchain-openai

pip install langchain-anthropic

pip install neo4j

pip install spacy

pip install python-dotenv

# Download spaCy model

python -m spacy download en_core_web_sm

```

## Environment Setup

Create a `.env` file with the following credentials:

```env

NEO4J_URI=neo4j://localhost:7687

NEO4J_USERNAME=your_username

NEO4J_PASSWORD=your_password

AZURE_FORM_RECOGNIZER_ENDPOINT=your_azure_endpoint

AZURE_FORM_RECOGNIZER_KEY=your_azure_key

OPENAI_API_KEY=your_openai_key

ANTHROPIC_API_KEY=your_anthropic_key

```

## Usage

### Basic Usage

```python

from tender_knowledge_graph import TenderKnowledgeGraph

# Initialize the pipeline

kg = TenderKnowledgeGraph(

    neo4j_uri=os.environ["NEO4J_URI"],

    neo4j_username=os.environ["NEO4J_USERNAME"],

    neo4j_password=os.environ["NEO4J_PASSWORD"],

    azure_endpoint=os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"],

    azure_key=os.environ["AZURE_FORM_RECOGNIZER_KEY"],

    openai_api_key=os.environ["OPENAI_API_KEY"],

    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],

)

# Process documents and build graph

await kg.build_graph("path/to/tender/documents")

# Close connections

kg.close()

```

### Document Types

The pipeline automatically detects various document types:

- Technical Specifications

- Requirements Documents

- Proposals

- Contracts

- Financial Documents

- Legal Documents

- Compliance Documents

- Evaluation Criteria

- Timelines

- General Documents

## Architecture

### Components

1\. **Document Processing Layer**

   - Azure Document Intelligence for text extraction

   - Page-level content processing

   - Document type detection

2\. **Text Processing Layer**

   - Semantic chunking using LangChain

   - Property extraction using Claude

   - Summary generation using GPT

3\. **Graph Creation Layer**

   - Node creation with metadata

   - Relationship detection

   - Confidence scoring

   - Batch processing

### Data Flow

1\. Document Intake → Azure Document Processing

2\. Text Extraction → Semantic Chunking

3\. Chunk Analysis → Property Extraction

4\. Node Creation → Relationship Analysis

5\. Graph Population → Neo4j Database

## Performance Optimization

- Batch processing for document chunks

- Async operations for API calls

- Confidence thresholds for relationships

- Optimized Neo4j operations

## Query Examples

```cypher

// Find all requirements in technical specifications

MATCH (n:TenderNode)

WHERE n.doc_type = 'TechnicalSpec'

RETURN n.text, n.properties;

// Find related documents with high confidence

MATCH (a:TenderNode)-[r]->(b:TenderNode)

WHERE r.confidence > 0.8

RETURN a.text, type(r), r.confidence, b.text;

```

## Best Practices

1\. **Document Organization**

   - Keep related documents in the same folder

   - Use consistent naming conventions

   - Maintain document hierarchy

2\. **Performance**

   - Adjust batch sizes based on document complexity

   - Monitor memory usage for large document sets

   - Use appropriate similarity thresholds

3\. **Maintenance**

   - Regularly update the graph with new documents

   - Monitor relationship confidence scores

   - Clean up obsolete relationships

## Limitations

- Maximum document size: 500MB per file

- Supported formats: PDF, DOCX, DOC

- Language support: Primary focus on English documents

## Future Enhancements

- [ ] Multi-language support

- [ ] Additional document formats

- [ ] Enhanced relationship patterns

- [ ] Improved confidence scoring

- [ ] Document version tracking

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
