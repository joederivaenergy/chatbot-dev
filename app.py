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
from langchain_core.chat_history import BaseChatMessageHistory  # Correct import path
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
                    'message_timestamp': str(int(time.time() * 1000)),  # Convert to string
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
            # Query all items for this session
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id)
            )
            
            # Delete each item
            for item in response.get('Items', []):
                self.table.delete_item(
                    Key={
                        'session_id': item['session_id'],
                        'message_timestamp': str(item['message_timestamp'])  # Ensure string format
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
        table.load()  # Check if table exists
        # st.success(f"✅ DynamoDB table '{DDB_TABLE_NAME}' exists")
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

# Check/create table on startup
if not create_dynamodb_table_if_not_exists():
    st.stop()

# --- Streamlit Page ---
st.set_page_config(page_icon="🤖", page_title="Diva the Chatbot", layout="centered", initial_sidebar_state="expanded")

st.sidebar.title("⚙️ Settings")

# Session / User identity (swap with SSO user if you have it)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

def reset_history():
    try:
        # Use custom history class
        hist = CustomDynamoDBChatHistory(
            table_name=DDB_TABLE_NAME, 
            session_id=st.session_state["session_id"]
        )
        hist.clear()
    except Exception as e:
        st.warning(f"Could not clear history: {e}")

with st.sidebar.expander("🧹 Tools", expanded=True):
    if st.button("🗑️ Clear Chat"):
        reset_history()
        st.rerun()

with st.sidebar.expander("📧 Support"):
    st.markdown("[Report an issue](mailto:joe.cheng@derivaenergy.com)")

st.sidebar.divider()
st.sidebar.caption("Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")

st.markdown("<h1 style='text-align: center;'>⚡Meet Diva!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deriva's AI Chatbot for Charging Guidelines.</p>", unsafe_allow_html=True)

# --- LLM + Memory (simplified approach) ---
chat_model = ChatBedrock(
    client=bedrock_runtime,
    model_id=BEDROCK_MODEL_ID,
    region_name=AWS_REGION,
    # model_kwargs={"temperature": 0.2}
)

# Use custom DynamoDB chat history that works with your table schema
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
    return {"context": "\n\n---\n\n".join(chunks), "sources": sources}

# --- Clarifying Router ---
ROUTER_POLICY = """
You are Diva, Deriva's internal charging-guidelines assistant.
 
DEFAULT: Clarify first for any query about policies, charging, codes, expenses, departments, projects, or sites. Only skip clarification for pure greetings or when the message + history already contains all critical fields.
 
Critical fields (generic):
- team/department (ask: “Which team or department do you work in?”)
- if the provided team is umbrella-level (e.g., an org name), ask for the specific sub-team/department within it
- site/plant if policy can vary by site
- any other column implied by the retrieved context that changes the code (category, activity, etc.)
 
Rules:
1) Greetings → intent: "answer".
2) Prefer intent: "clarify" and ask at most TWO concise, targeted questions for missing fields.
3) If prior chat history already contains what's needed, intent: "answer".
4) Never invent values; if unsure, ask.
5) Keep questions short and friendly.
 
Return ONLY JSON:
{
  "intent": "clarify" | "answer",
  "questions": ["q1","q2"],
  "known": {"team": "...", "department": "...", "site": "..."},
  "notes": "if IT, which IT department or team; if operations, is it wind, solar, or battery?"
}
"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_POLICY),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)


def route_turn(user_input: str) -> Dict:
    import re, json as _json

    # 0) Quick greeting bypass
    text_raw = (user_input or "").strip()
    text = text_raw.lower()
    GREETINGS = {"hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"}
    if text in GREETINGS or any(text.startswith(g) for g in GREETINGS):
        return {"intent": "answer", "questions": [], "known": {"reason": "greeting"}, "notes": ""}

    # 1) Heuristic: detect explicit team in the current input (so we don't rely solely on the LLM)
    TEAM_KEYWORDS = {
        "it": "IT",
        "finance": "Finance",
        "engineering": "Engineering",
        "ops": "Operations",
        "operations": "Operations",
        "data analytics": "IT",  # treat as IT umbrella; you can specialize later
    }
    team_guess = None
    for kw, norm in TEAM_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", text):
            team_guess = norm
            break

    # 2) Ask the LLM router (for broader detection like site or asset_type)
    try:
        mv = memory.load_memory_variables({})
        msgs = router_prompt.format_messages(chat_history=mv.get("chat_history", []), input=user_input)
        resp = chat_model.invoke(msgs)
        content = resp.content.strip()

        # strip code fences if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = _json.loads(m.group() if m else content)
    except Exception:
        data = {"intent": "clarify", "questions": [], "known": {}, "notes": "router_exception_fallback"}

    # 3) Consolidate knowns (prefer explicit user hint)
    known = data.get("known", {}) or {}
    if team_guess and not known.get("team"):
        known["team"] = team_guess

    team = (known.get("team") or "").strip().lower()
    asset_type = (known.get("asset_type") or "").strip().lower()
    site = (known.get("site") or "").strip()

    # 4) Compute what’s still missing
    missing_questions = []
    if not team:
        missing_questions.append("which team you're with (Operations, Engineering, Finance, or IT)")
    if team == "operations" and not asset_type:
        missing_questions.append("if it's for Wind, Solar, or Battery")
    if team == "operations" and not site:
        # only ask site for Ops to keep it short; include if we still have space
        missing_questions.append("the site/plant (if applicable)")

    # 5) Sufficiency rule
    sufficient = False
    if team:
        if team == "operations":
            sufficient = bool(asset_type)  # site can still be asked later in the answer if truly required
        else:
            sufficient = True

    # 6) Force **answer** when sufficient, even if the LLM said "clarify"
    if sufficient:
        return {
            "intent": "answer",
            "questions": [],
            "known": known,
            "notes": "forced_answer_minimum_context"
        }

    # 7) Otherwise clarify with only the missing pieces (max 2)
    qs = missing_questions[:2] or (data.get("questions", [])[:2] if isinstance(data.get("questions"), list) else [])
    return {
        "intent": "clarify",
        "questions": qs,
        "known": known,
        "notes": data.get("notes", "")
    }



# --- Answer Prompt (only for final answers) ---
SYSTEM_INSTRUCTIONS = (
    "You are Diva, an internal Deriva Energy assistant for charging guidelines. "
    "If the user is just greeting you (like 'hi', 'hello', 'hey', etc.), respond with a simple, friendly greeting and ask how you can help with charging guidelines. Do NOT use bullet points for greetings.\n\n"
    "For all other queries about guidelines, codes, departments, projects, etc., return a markdown bulleted list with exactly these fields:\n"
    "- **Description:**\n- **Account number:**\n- **Location:**\n- **Company ID:**\n- **Project:**\n- **Department:**\n"
    "Use 'N/A' when unavailable. Finish with 1-2 short notes if needed."
)

#"""
# If it's a greeting, greet back (your name is Diva. Made by Deriva Energy.) Else, based on the following retrieved information and the user's query, please provide the most relevant details about charging guidelines.\n\nExtract and present the following in a markdown bulleted list:\n\n- **Description:**\n- **Account number:**\n- **Location:**\n- **Company ID:**\n- **Project:**\n- **Department:**\n\nIf not available, return \"N/A\".\n\nFinish with 1-2 relevant notes if needed.
# """

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# --- Clarification Prompt (for asking questions) ---
CLARIFICATION_INSTRUCTIONS = (
    "You are Diva, an internal Deriva Energy assistant. "
    "Based on the conversation and the questions you need to ask, respond naturally and conversationally. "
    "Ask the clarifying questions in a friendly, helpful manner. Do NOT provide bullet points or structured data yet."
)

clarification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CLARIFICATION_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Please ask these clarifying questions: {questions}")
    ]
)


# def generate_clarification(user_input: str, questions: List[str]) -> str:
#     """Generate a natural clarifying response - simplified for speed"""
#     # Use a simple template instead of calling the LLM for speed
#     q_text = " ".join(questions[:2])  # Join questions naturally
#     return f"I'd be happy to help! To give you the right charging guidelines, could you tell me {q_text}?"

def generate_clarification(user_input: str, questions: List[str], notes: str = "") -> str:
    qs = [q.strip().rstrip("?") for q in (questions or []) if q and q.strip()]
    
    # Use notes to provide specific follow-up questions
    if "if IT, which IT department or team" in notes.lower() and "it" in user_input.lower():
        return "I can help with that! To give you the right charging guideline, could you tell me which specific IT department or team you're with?"
    
    if "if operations, is it wind, solar, or battery" in notes.lower() and "operations" in user_input.lower():
        return "I can help with that! Could you specify if this is for Wind, Solar, or Battery Operations?"
        
    if not qs:
        # Fallback if no specific questions are generated
        return "I'd be happy to help! To give you the right charging guidelines, could you tell me which team you're with (Operations, Engineering, Finance, or IT)?"
        
    if len(qs) == 1:
        q_text = qs[0] + "?"
    else:
        q_text = f"{qs[0]} and {qs[1]}?"
        
    return f"I can help with that. To point you to the right charging guideline, could you tell me {q_text}"

def generate_answer(user_input: str) -> Dict:
    retrieval = retrieve_from_kb(user_input)
    context = retrieval["context"]
    mv = memory.load_memory_variables({})
    messages = answer_prompt.format_messages(context=context, chat_history=mv["chat_history"], input=user_input)
    llm_resp = chat_model.invoke(messages)
    
    # Use the standard LangChain memory methods
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(llm_resp.content)
    
    return {"answer_md": llm_resp.content, "sources": retrieval["sources"]}

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

    # 1) Route: clarify or answer?
    try:
        decision = route_turn(user_input)
    except:
        # If router fails, default to answer for speed
        decision = {"intent": "answer", "questions": [], "known": {}, "notes": ""}

    if decision["intent"] == "clarify" and decision.get("questions"):
        # Generate a natural clarifying response
        # Pass the notes from the router's decision
        clarifier = generate_clarification(user_input, decision["questions"], decision["notes"]) 
    
        # Save to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(clarifier)
    
        with st.chat_message("assistant"):
            st.markdown(clarifier)
    else:
    # ... (rest of the code is unchanged)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = generate_answer(user_input)
                    st.markdown(result["answer_md"])
                    
                    # Comment out sources for production, uncomment for dev
                    # if result["sources"]:
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
