import os
import json
import uuid
import boto3
import streamlit as st
from typing import List, Dict

# --- Config ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "YBW1J8NMTI")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")

# --- LangChain (custom implementation for existing table) ---
from langchain_aws.chat_models import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
from typing import List

# Custom DynamoDB Chat History for your existing table schema
class CustomDynamoDBChatHistory(BaseChatMessageHistory):
    def __init__(self, table_name: str, session_id: str):
        super().__init__()
        self.table_name = table_name
        self.session_id = session_id
        self.dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        self.table = self.dynamodb.Table(table_name)
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from DynamoDB"""
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id),
                ScanIndexForward=True  # Sort by timestamp ascending
            )
            
            messages = []
            for item in response.get('Items', []):
                msg_type = item.get('message_type', 'human')
                content = item.get('content', '')
                
                if msg_type == 'human':
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            
            return messages
        except Exception as e:
            st.warning(f"Could not load chat history: {e}")
            return []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to DynamoDB"""
        try:
            msg_type = 'human' if isinstance(message, HumanMessage) else 'ai'
            self.table.put_item(
                Item={
                    'session_id': self.session_id,
                    'message_timestamp': str(int(time.time() * 1000)),
                    'message_type': msg_type,
                    'content': message.content
                }
            )
        except Exception as e:
            st.error(f"Failed to save message: {e}")
    
    def add_user_message(self, message: str):
        """Add user message to DynamoDB"""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
        """Add AI message to DynamoDB"""
        self.add_message(AIMessage(content=message))
    
    def clear(self):
        """Clear all messages for this session"""
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id)
            )
            
            for item in response.get('Items', []):
                self.table.delete_item(
                    Key={
                        'session_id': item['session_id'],
                        'message_timestamp': str(item['message_timestamp'])
                    }
                )
        except Exception as e:
            st.error(f"Failed to clear history: {e}")

# --- AWS Clients ---
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

# --- Create DynamoDB table if it doesn't exist ---
def create_dynamodb_table_if_not_exists():
    try:
        table = dynamodb.Table(DDB_TABLE_NAME)
        table.load()
    except dynamodb.meta.client.exceptions.ResourceNotFoundException:
        st.warning(f"Creating DynamoDB table '{DDB_TABLE_NAME}'...")
        try:
            table = dynamodb.create_table(
                TableName=DDB_TABLE_NAME,
                KeySchema=[
                    {
                        'AttributeName': 'SessionId',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'SessionId',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            table.wait_until_exists()
            st.success(f"✅ Created DynamoDB table '{DDB_TABLE_NAME}'")
        except Exception as e:
            st.error(f"❌ Failed to create table: {e}")
            return False
    except Exception as e:
        st.error(f"❌ Error checking table: {e}")
        return False
    return True

if not create_dynamodb_table_if_not_exists():
    st.stop()

# --- Streamlit Page ---
st.set_page_config(page_icon="", page_title="Diva the Chatbot", layout="centered", initial_sidebar_state="expanded")

st.sidebar.title("⚙️ Settings")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

def reset_history():
    try:
        hist = CustomDynamoDBChatHistory(
            table_name=DDB_TABLE_NAME, 
            session_id=st.session_state["session_id"]
        )
        hist.clear()
    except Exception as e:
        st.warning(f"Could not clear history: {e}")

with st.sidebar.expander(" Tools", expanded=True):
    if st.button("Clear Chat"):
        reset_history()
        st.rerun()

with st.sidebar.expander("�� Support"):
    st.markdown("[Report an issue](mailto:joe.cheng@derivaenergy.com)")

st.sidebar.divider()
st.sidebar.caption("Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")

st.markdown("<h1 style='text-align: center;'>⚡Meet Diva!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deriva's AI Chatbot for Charging Guidelines.</p>", unsafe_allow_html=True)

# --- LLM + Memory ---
chat_model = ChatBedrock(
    client=bedrock_runtime,
    model_id=BEDROCK_MODEL_ID,
    region_name=AWS_REGION,
)

chat_history_store = CustomDynamoDBChatHistory(
    table_name=DDB_TABLE_NAME,
    session_id=st.session_state["session_id"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=chat_history_store,
    return_messages=True
)

# --- Retrieval from Bedrock KB ---
def retrieve_from_kb(query: str, max_results: int = 6) -> Dict:
    resp = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": max_results}}
    )
    chunks, sources = [], []
    for item in resp.get("retrievalResults", []):
        txt = item.get("content", {}).get("text", "")
        loc = item.get("location", {})
        score = item.get("score", None)
        if txt:
            chunks.append(txt.strip())
        if loc:
            sources.append({"location": loc, "score": score})
    return {"context": "\n\n---\n\n".join(chunks), "sources": sources, "relevance_score": max([s.get("score", 0) for s in sources]) if sources else 0}

# --- Detect if query is about charging guidelines ---
def is_charging_related(query: str) -> bool:
    """Detect if the query is related to charging guidelines"""
    charging_keywords = [
        "charge", "charging", "expense", "expenses", "code", "codes", "account", "budget",
        "department", "project", "cost", "billing", "guidelines", "policy", "travel",
        "meal", "lodging", "flight", "hotel", "finance", "accounting", "reimbursement"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in charging_keywords)

# --- Improved Query Classification ---
def classify_and_respond(user_input: str) -> Dict:
    """
    Classify the query and determine response strategy:
    1. Greeting -> Simple greeting response
    2. Charging-related -> Try KB first, then clarify if needed
    3. General knowledge -> Use LLM directly
    """
    
    # Check for greetings
    text_lower = user_input.lower().strip()
    greetings = {"hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"}
    if text_lower in greetings or any(text_lower.startswith(g) for g in greetings):
        return {
            "type": "greeting",
            "response": "Hello! I'm Diva, your AI assistant for Deriva Energy's charging guidelines. I can help you find expense codes, department information, and charging policies. I can also answer general questions. How can I help you today?"
        }
    
    # Check if it's charging-related
    if is_charging_related(user_input):
        return {"type": "charging", "query": user_input}
    
    # General knowledge query
    return {"type": "general", "query": user_input}

# --- Enhanced Answer Generation ---
def generate_answer(user_input: str, query_type: str) -> Dict:
    """Generate answer based on query type"""
    
    if query_type == "charging":
        # Try knowledge base first
        retrieval = retrieve_from_kb(user_input)
        
        # If we have good relevance, provide answer
        if retrieval["relevance_score"] > 0.3:  # Lowered threshold
            # Check if we have enough context from conversation
            mv = memory.load_memory_variables({})
            history = mv.get("chat_history", [])
            
            # Extract any team/department info from history
            team_context = ""
            for msg in reversed(history[-10:]):  # Check last 10 messages
                content = msg.content.lower()
                if "operations" in content:
                    team_context = "Operations team"
                elif "finance" in content:
                    team_context = "Finance team"
                elif "engineering" in content:
                    team_context = "Engineering team"
                elif "it" in content:
                    team_context = "IT team"
                if team_context:
                    break
            
            # Build enhanced context
            enhanced_context = retrieval["context"]
            if team_context:
                enhanced_context = f"User context: {team_context}\n\n{enhanced_context}"
            
            # Generate structured answer
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are Diva, Deriva Energy's internal charging guidelines assistant.

Based on the context provided, answer the user's question about charging guidelines. If you have specific information, format it as:

- **Description:** [specific description]
- **Account number:** [number or N/A]
- **Location:** [location or N/A]  
- **Company ID:** [ID or N/A]
- **Project:** [project or N/A]
- **Department:** [department or N/A]

If the context doesn't have complete information but is relevant, provide what you can and mention what additional details might be helpful.

If you need more context to give accurate charging guidelines, ask for clarification about team/department.

Context: {context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            messages = answer_prompt.format_messages(
                context=enhanced_context,
                chat_history=mv["chat_history"], 
                input=user_input
            )
            llm_resp = chat_model.invoke(messages)
            
            return {
                "answer": llm_resp.content,
                "sources": retrieval["sources"],
                "type": "structured"
            }
        
        else:
            # Low relevance - ask for clarification
            return {
                "answer": "I can help you with charging guidelines! To give you the most accurate information, could you tell me which team you're with (Operations, Engineering, Finance, or IT)?",
                "sources": [],
                "type": "clarification"
            }
    
    else:  # General knowledge
        # Use LLM for general questions
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Diva, a helpful AI assistant for Deriva Energy. Answer the user's question naturally and conversationally. If it's not related to charging guidelines, feel free to use your general knowledge."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        mv = memory.load_memory_variables({})
        messages = general_prompt.format_messages(
            chat_history=mv["chat_history"],
            input=user_input
        )
        llm_resp = chat_model.invoke(messages)
        
        return {
            "answer": llm_resp.content,
            "sources": [],
            "type": "general"
        }

# --- Render existing history ---
for m in chat_history_store.messages:
    role = "assistant" if m.type in ("ai", "assistant") else "user"
    with st.chat_message(role):
        st.markdown(m.content)

# --- Chat loop ---
user_input = st.chat_input("Ask about codes, departments, projects, etc.")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Classify the query
    classification = classify_and_respond(user_input)
    
    if classification["type"] == "greeting":
        # Simple greeting response
        response = classification["response"]
        
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
        
        with st.chat_message("assistant"):
            st.markdown(response)
    
    else:
        # Generate answer based on type
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = generate_answer(user_input, classification["type"])
                    
                    memory.chat_memory.add_user_message(user_input)
                    memory.chat_memory.add_ai_message(result["answer"])
                    
                    st.markdown(result["answer"])
                    
                    # Show sources for charging-related queries if in debug mode
                    # if result.get("sources") and result["type"] == "structured":
                    #     with st.expander("Sources"):
                    #         for i, s in enumerate(result["sources"], 1):
                    #             loc = s.get("location", {})
                    #             score = s.get("score")
                    #             st.markdown(f"- {i}. `{json.dumps(loc)}`" + (f"  (score: {score:.3f})" if score is not None else ""))
                            
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# --- Footer ---
st.divider()
footer = """
<style>
a:link , a:visited{ color: blue; background-color: transparent; text-decoration: underline; }
a:hover, a:active { color: red; background-color: transparent; text-decoration: underline; }
.footer { position: fixed; left:0; bottom:0; width:100%; background-color:white; color:black; text-align:center; }
</style>
<div class="footer">
<p>Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
