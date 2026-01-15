from pydoc import doc
from tkinter import SE
from typing import Any


from enum import StrEnum
from pickle import TRUE

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
        )

    # TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB...
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)
    def _process_text_file(
        self,
        file_name: str,
        chunk_size: int,
        overlap: int,
        dimensions: int,
        truncate_table: bool = TRUE,
    ):
        if truncate_table:
            self._truncate_table()

        with open(file=file_name, mode="r", encoding="utf-8") as file:
            content = file.read()

        print("Processing text...")
        chunks = chunk_text(text=content, chunk_size=chunk_size, overlap=overlap)

        embeddings = self.embeddings_client.get_embeddings(
            text=chunks, dimensions=dimensions
        )
        for chunk, index in len(chunks):
            self._save_chunk(
                chunk=chunk, embedding=embeddings[index], document_name=file_name
            )

    def _save_chunk(self, chunk: str, embedding: list[float], document_name: str):
        print("Saving chunk to database\n")
        vector_string = f"[{','.join(map[str](str,embedding))}]"
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector);",
                    (document_name, chunk, embedding),
                )
            conn.commit()
        finally:
            conn.close()
        print("Chunk is saved to database\n")

    def _truncate_table(self):
        print("Truncating vectors table\n")
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors;")
            conn.commit()
        finally:
            conn.close()
        print("Truncating vectors table is completed\n")

    # TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

    def search(
        self,
        search_mode: SearchMode,
        user_request: str,
        top_k: int,
        min_score: float,
        threshold: float,
        dimensions: int,
    ) -> list[str]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if min_score < 0 or min_score > 1:
            raise ValueError("min_score must be between 0 and 1")
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")

        print("Generating embeddings from user request\n")
        user_request_embeddings = self.embeddings_client.get_embeddings(
            text=user_request, dimensions=dimensions
        )
        print("Embeddings from user request are generated\n")
        vector_string = f"[{','.join(map[str](str,user_request_embeddings))}]"
        print("Vector string is generated\n")

        print("Searching in database...\n")

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                conn.execute(
                    self._get_query(search_mode), (vector_string, min_score, top_k)
                )
            result = cursor.fetchall()
            retrieved_chunks = []
            for row in result:
                retrieved_chunks.append(row["text"])

            print("Retrieved chunks: ", retrieved_chunks, "\n")
            return retrieved_chunks

        except Exception as e:
            print("Error searching in database: ", e)
            raise e

        finally:
            conn.close()
        print("Searching in database is completed\n")

    def _get_query(self, search_mode: SearchMode):
        return """
        SELECT text, embedding {mode} %s::vector as distance
        FROM vectors
        WHERE embedding {mode} %s::vector < %s
        ORDER BY distance ASC
        LIMIT %s;
        """.format(
            mode="<->" if search_mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"
        )
