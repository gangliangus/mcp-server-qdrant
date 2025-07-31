import asyncio
from enum import Enum

from fastembed.common.model_description import DenseModelDescription

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from langchain_openai import OpenAIEmbeddings

class ModelName(Enum):
    # openai dense models
    TEXT_EMBEDDING_ADA_2 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class OpenAIEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    """

    def __init__(self, model_name: str):
        if model_name not in [ModelName.TEXT_EMBEDDING_3_LARGE.value, ModelName.TEXT_EMBEDDING_ADA_2.value]:
            raise ValueError(f"Unsupported openai embedding model: {model_name}")

        self.model_name = model_name
        self.embedding_model = OpenAIEmbeddings(
            model=model_name,
            dimensions=(
                3072 if model_name == ModelName.TEXT_EMBEDDING_3_LARGE.value else None
            ),
            max_retries=20,
        )
        self.vector_size = 3072 if model_name == ModelName.TEXT_EMBEDDING_3_LARGE.value else 1536

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.embedding_model.embed_documents(documents)
        )
        return embeddings
        #return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.embedding_model.embed_documents([query])
        )
        return embeddings[0]

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.6.0.
        """
        return "dense_vector"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.vector_size
