import os
import re
import json
import uuid
import time
from typing import List, Dict, Any

import boto3
import streamlit as st
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

# ======================================================================================
# 1. CONFIGURATION & AWS CLIENTS
# ======================================================================================

# --- Environment Variables ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "9DB3QUTETF")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")

# --- Boto3 Clients ---
try:
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
except ClientError as e:
    st.error(f"Error initializing AWS clients in region {AWS_REGION}: {e}. Please check your credentials and region configuration.")
    st.stop()


# ======================================================================================
# 2. DYNAMODB CHAT HISTORY (FIXED)
# ======================================================================================

class DynamoDBChatHistory(BaseChatMessageHistory):
    """
    Stores chat history in a DynamoDB table with string-based message_timestamp.
    """
    def __init__(self, table_name: str, session_id: str):
        self.table = dynamodb.Table(table_name)
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id)
            )
            # Sort by timestamp (which is a string, but lexicographical sort works for ISO timestamps)
            items = sorted(response.get("Items", []), key=lambda x: x['message_timestamp'])
            messages = []
            for item in items:
                content = item.get("content", "")
                if item.get("message_type") == 'human':
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            return messages
        except ClientError as e:
            st.warning(f"Could not load chat history from DynamoDB: {e}")
            return []

    def add_message(self, message: BaseMessage) -> None:
        message_type = "human" if isinstance(message, HumanMessage) else "ai"
        try:
            self.table.put_item(
                Item={
                    "session_id": self.session_id,
                    # Convert timestamp to a string to match the table schema
                    "message_timestamp": str(int(time.time() * 1000)),
                    "message_type": message_type,
                    "content": message.content,
                }
            )
        except ClientError as e:
            st.error(f"Failed to save message to DynamoDB: {e}")

    def clear(self) -> None:
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id),
                ProjectionExpression='session_id, message_timestamp'
            )
            with self.table.batch_writer() as batch:
                for item in response.get("Items", []):
                    # Ensure both key attributes are of the correct type (string)
                    batch.delete_item(Key={"session_id": item["session_id"], "message_timestamp": item["message_timestamp"]})
        except ClientError as e:
            st.error(f"Failed to clear history from DynamoDB: {e}")


# ======================================================================================
# 3. CORE LOGIC & PROMPTS
# ======================================================================================

# --- Entity Extraction Prompt (Refactored to be a system prompt) ---
ENTITY_EXTRACTION_PROMPT = """
Analyze the user's input and the conversation history to extract key information.
The user is asking about charging guidelines at Deriva Energy.

You must extract the following entities if they are present:
- **team**: The specific department (e.g., "Operations", "Finance", "IT", "Engineering").
- **asset_type**: Only relevant for the Operations team. Can be "Wind", "Solar", or "Battery".

Return ONLY a JSON object with:
{{
  "intent": "clarify" | "answer",
  "questions": [ "q1", "q2" ],
  "known": {{"team": "...", "asset_type": "...", "site": "..."}},
  "notes": ""
}}

Return ONLY the JSON object. Do not include any additional text or markdown formatting.
"""

# --- Answer Generation ---
ANSWER_PROMPT = """
You are Diva, an internal Deriva Energy assistant for charging guidelines.
Based on the retrieved context, conversation history, and user query, provide a detailed answer.

**Instructions:**
1.  If the user is just greeting you (e.g., 'hi', 'hello'), respond with a simple, friendly greeting and ask how you can help with charging guidelines. Do not use bullet points.
2.  For all other questions, use the provided context to answer. Format the response as a markdown bulleted list with these exact fields:
    - **Description:**
    - **Account number:**
    - **Location:**
    - **Company ID:**
    - **Project:**
    - **Department:**
3.  Use "N/A" if a field is not available in the context.
4.  You can add a short "Notes" section at the end for any important extra details.
5.  Always state your confidence level at the very end (e.g., "Confidence: High").

**Context from Knowledge Base:**
{context}
"""

def get_llm():
    """Initializes and returns the ChatBedrock model."""
    return ChatBedrock(
        client=bedrock_runtime,
        model_id=BEDROCK_MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": 0.1},
    )

def retrieve_from_kb(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Retrieves context and confidence scores from the Bedrock Knowledge Base."""
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": max_results}}
        )
        chunks = []
        scores = []
        for result in response.get("retrievalResults", []):
            chunks.append(result["content"]["text"])
            scores.append(result.get("score", 0.0))
            
        # Get the highest confidence score from the retrieved chunks
        max_score = max(scores) if scores else 0.0

        return {"context": "\n\n---\n\n".join(chunks), "confidence": max_score}
    except ClientError as e:
        st.error(f"Error retrieving from Knowledge Base: {e}")
        return {"context": "Error retrieving context.", "confidence": 0.0}

def extract_entities(user_input: str) -> Dict[str, Any]:
    """
    Uses the LLM to extract 'team' and 'asset_type' from user input.
    Refactored to use ChatPromptTemplate correctly.
    """
    history = st.session_state.chat_history.messages
    llm = get_llm()

    # The template is now correctly defined with variables and the fixed JSON example
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACTION_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({
            "chat_history": history,
            "input": user_input
        })
        content = response.content.strip()
        
        # Robust JSON parsing
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return {"known": {}, "intent": "clarify", "questions": ["which team or department you're with"], "notes": "No valid JSON returned from LLM."}
    except (json.JSONDecodeError, ClientError) as e:
        st.warning(f"Could not extract entities due to an error: {e}")
        return {"known": {}, "intent": "clarify", "questions": ["which team or department you're with"], "notes": "JSON parsing failed."}


def generate_final_answer(user_input: str) -> str:
    """Generates the final answer or a clarifying question based on confidence."""
    known_context = st.session_state.known_context

    # New: Check for a specific 'description' in the user's input
    # This is a heuristic to help the search. You can refine this.
    specific_description_pattern = r"(for|about) (.+?)(?:$|[\.\?])"
    match = re.search(specific_description_pattern, user_input, re.IGNORECASE)
    
    specific_description = None
    if match:
        specific_description = match.group(2).strip()

    # Construct the full search query
    full_query = f"Charging guidelines for {known_context.get('team')}"
    if known_context.get('asset_type'):
        full_query += f" related to {known_context.get('asset_type')} assets."
    
    # New: Add the extracted description to the query for better search
    if specific_description:
        full_query += f" on the topic of {specific_description}."

    full_query += f" Original question: {user_input}"
    
    retrieval = retrieve_from_kb(full_query)
    context = retrieval["context"]
    confidence = retrieval["confidence"]
    
    # Lower the confidence threshold to a more realistic value (e.g., 0.5)
    # The confidence score is a measure of the semantic similarity of the query to the retrieved chunk.
    # Scores are often lower in real-world scenarios than in simple examples.
    if confidence < 0.5:
        return "I'm not completely confident I have the right information. Could you please provide a more specific 'Description' for what you're trying to charge? This will help me find the correct guideline."

    # If confidence is high enough, proceed with generating the full answer
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt | get_llm()
    response = chain.invoke({
        "context": context,
        "chat_history": st.session_state.chat_history.messages,
        "input": user_input
    })
    return response.content
# ======================================================================================
# 4. STREAMLIT UI & APPLICATION FLOW
# ======================================================================================

def setup_page():
    """Sets up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Diva the Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.markdown("<h1 style='text-align: center;'>⚡ Meet Diva!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your AI assistant for Deriva's Charging Guidelines.</p>", unsafe_allow_html=True)

def setup_sidebar():
    """Sets up the Streamlit sidebar."""
    with st.sidebar:
        st.title("⚙️ Settings")
        if st.button("🗑️ Clear & Reset Chat", use_container_width=True):
            st.session_state.chat_history.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        st.markdown(
            "Diva is an internal tool and may sometimes provide incorrect information. "
            "Always verify critical details."
        )

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = DynamoDBChatHistory(
            table_name=DDB_TABLE_NAME,
            session_id=st.session_state.session_id,
        )
    
    if "known_context" not in st.session_state:
        st.session_state.known_context = {
            "team": None,
            "asset_type": None,
        }

def display_chat_history():
    """Displays the existing chat messages."""
    for msg in st.session_state.chat_history.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

# ======================================================================================
# 4. STREAMLIT UI & APPLICATION FLOW (FIXED)
# ======================================================================================

def handle_user_input(user_input: str):
    """
    Main logic flow for handling a user's turn.
    """
    st.session_state.chat_history.add_message(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Vague topics that require clarification
    VAGUE_TOPICS = {"travel", "expenses", "software", "hardware"}
    user_input_lower = user_input.lower()
    
    ai_response = ""

    with st.chat_message("assistant"):
        # GREETING CHECK
        greetings = {"hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening"}
        if user_input_lower.strip() in greetings or any(user_input_lower.strip().startswith(g) for g in greetings):
            ai_response = "Hello! Welcome to Deriva Energy's charging guidelines assistance. How can I help you with information about charging codes, projects, or departments today?"
            st.markdown(ai_response)
        
        # VAGUE TOPIC CHECK
        elif any(topic in user_input_lower for topic in VAGUE_TOPICS):
            # If a vague topic is detected, ask for specifics.
            ai_response = "I can help with that! To give you the right charging guidelines, could you tell me more specifically about the type of travel you're asking about (e.g., flights, lodging, or meals)?"
            st.markdown(ai_response)

        # ENTITY EXTRACTION & DECISION TREE
        else:
            # Only proceed to entity extraction if the input is not a greeting or a vague topic
            entities = extract_entities(user_input)
            if entities.get("known", {}).get("team"):
                st.session_state.known_context["team"] = entities["known"]["team"]
            if entities.get("known", {}).get("asset_type"):
                st.session_state.known_context["asset_type"] = entities["known"]["asset_type"]
            
            known_context = st.session_state.known_context

            if not known_context.get("team"):
                ai_response = "I can help with that! To give you the most accurate information, could you please tell me which team or department you're with?"
            
            elif known_context.get("team") == "Operations" and not known_context.get("asset_type"):
                ai_response = "Thanks! For Operations, I'll also need to know the asset type. Is this for Wind, Solar, or Battery?"
                
            else:
                with st.spinner("Finding the answer..."):
                    ai_response = generate_final_answer(user_input)

            st.markdown(ai_response)

    st.session_state.chat_history.add_message(AIMessage(content=ai_response))

def create_ddb_table_if_not_exists():
    """Checks if the DynamoDB table exists and creates it if not."""
    try:
        dynamodb.meta.client.describe_table(TableName=DDB_TABLE_NAME)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            st.warning(f"DynamoDB table '{DDB_TABLE_NAME}' not found. Creating it now...")
            try:
                table = dynamodb.create_table(
                    TableName=DDB_TABLE_NAME,
                    KeySchema=[
                        {'AttributeName': 'session_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'message_timestamp', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'session_id', 'AttributeType': 'S'},
                        # Define message_timestamp as a string (S) type
                        {'AttributeName': 'message_timestamp', 'AttributeType': 'S'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                table.wait_until_exists()
                st.success(f"✅ Created DynamoDB table '{DDB_TABLE_NAME}'")
                return True
            except ClientError as create_error:
                st.error(f"❌ Failed to create DynamoDB table: {create_error}")
                return False
        else:
            st.error(f"❌ An error occurred checking the DynamoDB table: {e}")
            return False

# ======================================================================================
# 5. MAIN APPLICATION EXECUTION
# ======================================================================================

def main():
    """Main function to run the Streamlit app."""
    if not create_ddb_table_if_not_exists():
        st.stop()
        
    setup_page()
    setup_sidebar()
    initialize_session_state()
    display_chat_history()
    
    if user_input := st.chat_input("Ask about codes, departments, projects, etc."):
        handle_user_input(user_input)

if __name__ == "__main__":

    main()
