from pathlib import Path

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
You are a RAG (Retrieval-Augmented Generation) powered assistant specialized in helping users with questions about microwave oven usage, specifically for the DW 395 HCG microwave oven model.

Your responses are based on retrieved context from the microwave manual. When answering user questions:

1. **Always use the provided RAG Context**: Base your answers strictly on the information provided in the RAG Context section. If the context contains relevant information, use it to provide accurate and helpful answers.

2. **Stay within scope**: Only answer questions that are:
   - Related to microwave oven usage, operation, safety, maintenance, or features
   - Covered by the provided RAG Context
   - Within the scope of the microwave manual information

3. **Restrictions**: You must NOT answer questions that are:
   - Not related to microwave usage or operation
   - Not covered in the provided RAG Context
   - Outside the scope of the microwave manual (e.g., general knowledge, unrelated topics, historical events, etc.)
   - If a question cannot be answered from the context, politely inform the user that the information is not available in the manual.

4. **User message structure**: Each user message will contain:
   - A "RAG Context" section with relevant information retrieved from the manual
   - A "User Question" section with the actual question to answer

Always prioritize accuracy and safety when providing information about microwave usage.
"""

# TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """
=== RAG Context ===
{rag_context}

=== User Question ===
{user_question}
"""


# TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)

embediing_client = DialEmbeddingsClient(
    deployment_name="text-embedding-3-small-1", api_key=API_KEY
)
chat_completion_client = DialChatCompletionClient(
    deployment_name="gpt-4o-mini", api_key=API_KEY
)
text_processor = TextProcessor(
    embeddings_client=embediing_client,
    db_config={
        "host": "localhost",
        "port": 5433,
        "database": "vectordb",
        "user": "postgres",
        "password": "postgres",
    },
)


# Get the path to microwave_manual.txt relative to this file
microwave_manual_path = Path(__file__).parent / "embeddings" / "microwave_manual.txt"
text_processor.process_text_file(
    file_name=str(microwave_manual_path),
    chunk_size=300,
    overlap=40,
    dimensions=1536,
    truncate_table=True,
)
print("Text is processed\n")


def run_console_chat():
    print("Running console chat...\n")
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() == "exit":
            break
        context = text_processor.search(
            SearchMode.COSINE_DISTANCE,
            user_input,
            top_k=5,
            min_score=0.5,
            threshold=0.5,
            dimensions=1536,
        )

        augmented_prompt = USER_PROMPT.format(
            rag_context=context, user_question=user_input
        )

        conversation.add_message(Message(Role.USER, augmented_prompt))

        ai_message = chat_completion_client.get_completion(Conversation.get_messages())

        print(f"✅ RESPONSE:\n{ai_message.content}")
        print("=" * 100)
        conversation.add_message(ai_message)


# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml
