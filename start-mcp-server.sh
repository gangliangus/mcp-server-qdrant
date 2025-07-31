#!/bin/bash
#Step 1: install the package from source code
#under the containing folder run:
#> pip install -e .

#Step 2: start the server
OPENAI_API_KEY="..." \
QDRANT_URL="http://work.laptop:6333" \
COLLECTION_NAME="main2-openai-text3l-3072" \
VECTOR_FIELD_NAME="dense_vector" \
EMBEDDING_PROVIDER="openai" \
EMBEDDING_MODEL="text-embedding-3-large" \
TOOL_FIND_DESCRIPTION="Use this tool to Search company's knowledge repository which is a vector database of past proposal answers, capabilities & past projects to retrieve relevant context." \
mcp-server-qdrant --transport sse


# start to connect to main-cleaned collection
#OPENAI_API_KEY="..." \
#QDRANT_URL="http://work.laptop:6333" \
#COLLECTION_NAME="main-cleaned" \
#EMBEDDING_PROVIDER="openai" \
#EMBEDDING_MODEL="text-embedding-ada-002" \
#TOOL_FIND_DESCRIPTION="Use this tool to Search company's knowledge repository which is a vector database of past proposal answers, capabilities & past projects to retrieve relevant context." \
#mcp-server-qdrant --transport sse
