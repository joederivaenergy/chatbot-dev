import os
import json
import uuid
import boto3
import streamlit as st
from typing import List, Dict
import re

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
st.set_page_config(page_icon="🤖", page_title="Diva the Chatbot", layout="centered", initial_sidebar_state="expanded")

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
    return {"context": "\n\n---\n\n".join(chunks), "sources": sources}

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

# --- Clarifying Router (FROM FIRST VERSION) ---
ROUTER_POLICY = """
You are Diva, Deriva's internal charging-guidelines assistant.
 
DEFAULT: Clarify first for any query about policies, charging, codes, expenses, departments, projects, or sites. Only skip clarification for pure greetings or when the message + history already contains all critical fields.
 
Critical fields (generic):
- team/department (ask: "Which team or department do you work in?")
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
    """Route charging-related queries (FROM FIRST VERSION)"""
    import re, json as _json

    # 0) Quick greeting bypass
    text_raw = (user_input or "").strip()
    text = text_raw.lower()
    GREETINGS = {"hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"}
    if text in GREETINGS or any(text.startswith(g) for g in GREETINGS):
        return {"intent": "answer", "questions": [], "known": {"reason": "greeting"}, "notes": ""}

    # 1) Heuristic: detect explicit team in the current input
    TEAM_KEYWORDS = {
        "it": "IT",
        "finance": "Finance",
        "engineering": "Engineering",
        "ops": "Operations",
        "operations": "Operations",
        "data analytics": "IT",
    }
    team_guess = None
    for kw, norm in TEAM_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", text):
            team_guess = norm
            break

    # 2) Ask the LLM router
    try:
        mv = memory.load_memory_variables({})
        msgs = router_prompt.format_messages(chat_history=mv.get("chat_history", []), input=user_input)
        resp = chat_model.invoke(msgs)
        content = resp.content.strip()

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

    # 3) Consolidate knowns
    known = data.get("known", {}) or {}
    if team_guess and not known.get("team"):
        known["team"] = team_guess

    team = (known.get("team") or "").strip().lower()
    asset_type = (known.get("asset_type") or "").strip().lower()
    site = (known.get("site") or "").strip()

    # 4) Compute what's still missing
    missing_questions = []
    if not team:
        missing_questions.append("which team you're with (Operations, Engineering, Finance, or IT)")
    if team == "operations" and not asset_type:
        missing_questions.append("if it's for Wind, Solar, or Battery")
    if team == "operations" and not site:
        missing_questions.append("the site/plant (if applicable)")

    # 5) Sufficiency rule
    sufficient = False
    if team:
        if team == "operations":
            sufficient = bool(asset_type)
        else:
            sufficient = True

    # 6) Force answer when sufficient
    if sufficient:
        return {
            "intent": "answer",
            "questions": [],
            "known": known,
            "notes": "forced_answer_minimum_context"
        }

    # 7) Otherwise clarify
    qs = missing_questions[:2] or (data.get("questions", [])[:2] if isinstance(data.get("questions"), list) else [])
    return {
        "intent": "clarify",
        "questions": qs,
        "known": known,
        "notes": data.get("notes", "")
    }

# --- Answer Prompts ---
# CHARGING-RELATED SYSTEM INSTRUCTIONS (UPDATED FORMAT)
CHARGING_SYSTEM_INSTRUCTIONS = (
    "You are Diva, an internal Deriva Energy assistant for charging guidelines. "
    "If the user is just greeting you (like 'hi', 'hello', 'hey', etc.), respond with a simple, friendly greeting and ask how you can help with charging guidelines. Do NOT use bullet points for greetings.\n\n"
    "For ALL charging-related questions, you MUST use this EXACT format:\n\n"
    "**Description:** [specific description from context]\n"
    "**Account number:** [account number from context or N/A]\n"
    "**Location:** [DSOP, DWOP, DCS4, DWE1, STRG, or DSOL depending on the specific project or area]\n"
    "**Company ID:** [77079 (Deriva Energy Sub I), 75752 (Deriva Energy Wind, LLC), or 75969 (Deriva Energy Storage, LLC) based on context]\n"
    "**Project:** [DSOP25G001, DWOP25G001, DCS425G001, DWE125G001, STRG25G001, or DSOL25G001 - choose the appropriate one based on specific work]\n"
    "**Department:** [department from context or based on user's team]\n\n"
    "IMPORTANT MAPPING RULES:\n"
    "- Solar Operations → Location: DSOP, Company ID: 77079, Project: DSOP25G001\n"
    "- Wind Operations → Location: DWOP, Company ID: 75752, Project: DWOP25G001\n"
    "- Solar Development → Location: DCS4, Company ID: 77079, Project: DCS425G001\n"
    "- Wind Development → Location: DWE1, Company ID: 75752, Project: DWE125G001\n"
    "- Storage Development → Location: STRG, Company ID: 75969, Project: STRG25G001\n"
    "- General/Admin → Location: DSOL, Company ID: 75736, Project: DSOL25G001\n\n"
    "Use 'N/A' only when information is truly unavailable. Add 1-2 short notes at the end if helpful."
)

charging_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CHARGING_SYSTEM_INSTRUCTIONS + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# GENERAL KNOWLEDGE SYSTEM INSTRUCTIONS (FROM SECOND VERSION)
GENERAL_SYSTEM_INSTRUCTIONS = """You are Diva, a helpful AI assistant for Deriva Energy. Answer the user's question naturally and conversationally. If it's not related to charging guidelines, feel free to use your general knowledge.

Current facts (as of 2025):
- Donald Trump is the current President of the United States (inaugurated January 20, 2025)
- Trump won the 2024 presidential election against Kamala Harris"""

general_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERAL_SYSTEM_INSTRUCTIONS),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# --- Clarification Generation (FROM FIRST VERSION) ---
def generate_clarification(user_input: str, questions: List[str], notes: str = "") -> str:
    qs = [q.strip().rstrip("?") for q in (questions or []) if q and q.strip()]
    
    if "if IT, which IT department or team" in notes.lower() and "it" in user_input.lower():
        return "I can help with that! To give you the right charging guideline, could you tell me which specific IT department or team you're with?"
    
    if "if operations, is it wind, solar, or battery" in notes.lower() and "operations" in user_input.lower():
        return "I can help with that! Could you specify if this is for Wind, Solar, or Battery Operations?"
        
    if not qs:
        return "I'd be happy to help! To give you the right charging guidelines, could you tell me which team you're with (Operations, Engineering, Finance, or IT)?"
        
    if len(qs) == 1:
        q_text = qs[0] + "?"
    else:
        q_text = f"{qs[0]} and {qs[1]}?"
        
    return f"I can help with that. To point you to the right charging guideline, could you tell me {q_text}"

# --- Answer Generation Functions ---
def generate_charging_answer(user_input: str) -> Dict:
    """Generate charging-related answer using first version's logic"""
    retrieval = retrieve_from_kb(user_input)
    context = retrieval["context"]
    mv = memory.load_memory_variables({})
    messages = charging_answer_prompt.format_messages(context=context, chat_history=mv["chat_history"], input=user_input)
    llm_resp = chat_model.invoke(messages)
    
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(llm_resp.content)
    
    return {"answer_md": llm_resp.content, "sources": retrieval["sources"]}

def generate_general_answer(user_input: str) -> Dict:
    """Generate general knowledge answer using second version's logic"""
    mv = memory.load_memory_variables({})
    messages = general_answer_prompt.format_messages(
        chat_history=mv["chat_history"],
        input=user_input
    )
    llm_resp = chat_model.invoke(messages)
    
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(llm_resp.content)
    
    return {"answer_md": llm_resp.content, "sources": []}

# --- Main Query Processing ---
def process_query(user_input: str):
    """Main query processing logic"""
    
    # 1) Check for simple greetings first
    text_lower = user_input.lower().strip()
    greetings = {"hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"}
    if text_lower in greetings or any(text_lower.startswith(g) for g in greetings):
        greeting_response = "Hello! I'm Diva, your AI assistant for Deriva Energy's charging guidelines. I can help you find expense codes, department information, and charging policies. I can also answer general questions. How can I help you today?"
        
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(greeting_response)
        
        return {"type": "greeting", "response": greeting_response}
    
    # 2) Determine if it's charging-related
    if is_charging_related(user_input):
        # Use first version's routing logic for charging questions
        try:
            decision = route_turn(user_input)
        except:
            decision = {"intent": "answer", "questions": [], "known": {}, "notes": ""}
        
        if decision["intent"] == "clarify" and decision.get("questions"):
            clarifier = generate_clarification(user_input, decision["questions"], decision["notes"])
            
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(clarifier)
            
            return {"type": "clarification", "response": clarifier}
        else:
            # Generate charging answer
            result = generate_charging_answer(user_input)
            return {"type": "charging", "response": result["answer_md"], "sources": result["sources"]}
    
    else:
        # Use second version's logic for general questions
        result = generate_general_answer(user_input)
        return {"type": "general", "response": result["answer_md"]}

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

    with st.chat_message("assistant"):
        if not is_charging_related(user_input) and not any(user_input.lower().strip().startswith(g) for g in ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]):
            # For non-charging questions, no spinner
            try:
                result = process_query(user_input)
                st.markdown(result["response"])
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
        else:
            # For charging questions, show spinner
            with st.spinner("Thinking…"):
                try:
                    result = process_query(user_input)
                    st.markdown(result["response"])
                    
                    # Show sources for charging queries if needed (uncomment for dev)
                    # if result.get("sources"):
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
