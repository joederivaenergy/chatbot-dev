import os
import json
import uuid
import boto3
import streamlit as st
import pandas as pd
from typing import List, Dict
import time
import re

# --- Config ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")

# --- CSV File Paths ---
CSV_FILES = {
    'IT': 'csvs/Guidelines_cleaned_it.csv',
    'Finance': 'csvs/finance.csv',
    'HR': 'csvs/hr.csv',
    'Operations': 'csvs/operations.csv',
    'Engineering': 'csvs/engineering.csv',
}

# --- LangChain ---
from langchain_aws.chat_models import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# ============================================
# CUSTOM DYNAMODB CHAT HISTORY
# ============================================

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
                ScanIndexForward=True
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
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str):
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

# ============================================
# LOAD CSV DATA
# ============================================

@st.cache_data
def load_all_csvs():
    """Load all CSV files into memory"""
    data = {}
    for team, filepath in CSV_FILES.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Strip whitespace from all string columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                data[team] = df
                print(f"✅ Loaded {team}: {len(df)} rows")
            except Exception as e:
                print(f"⚠️ Error loading {team} CSV: {e}")
                data[team] = pd.DataFrame()
        else:
            print(f"⚠️ CSV not found for {team}: {filepath}")
            data[team] = pd.DataFrame()
    return data

# Load CSVs at startup
ALL_TEAM_DATA = load_all_csvs()

# ============================================
# AWS CLIENTS & TABLE SETUP
# ============================================

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

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
                    {'AttributeName': 'SessionId', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'SessionId', 'AttributeType': 'S'}
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

# ============================================
# STREAMLIT PAGE CONFIG
# ============================================

st.set_page_config(
    page_icon="deriva.jpg", 
    page_title="Diva the Chatbot", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# ============================================
# SIDEBAR
# ============================================

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

with st.sidebar.expander("�� Tools", expanded=True):
    if st.button("��️ Clear Chat"):
        reset_history()
        st.rerun()

with st.sidebar.expander("�� Support"):
    st.markdown("[Report an issue](mailto:joe.cheng@derivaenergy.com)")

st.sidebar.divider()
st.sidebar.caption("Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")

# ============================================
# HEADER
# ============================================

st.markdown("<h1 style='text-align: center;'>⚡Meet Diva!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deriva's AI Chatbot for Charging Guidelines.</p>", unsafe_allow_html=True)

# ============================================
# LLM & MEMORY SETUP
# ============================================

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

# ============================================
# CSV QUERY FUNCTIONS
# ============================================

def query_csv_data(team: str, description_query: str, location: str = None) -> pd.DataFrame:
    """Query CSV data for specific team and description"""
    
    if team not in ALL_TEAM_DATA or ALL_TEAM_DATA[team].empty:
        return pd.DataFrame()
    
    df = ALL_TEAM_DATA[team].copy()
    
    # Fuzzy match on description
    description_query_lower = description_query.lower()
    
    # Try exact match first
    matches = df[df['Description'].str.lower() == description_query_lower]
    
    # If no exact match, try partial match
    if matches.empty:
        matches = df[df['Description'].str.lower().str.contains(description_query_lower, na=False)]
    
    # Filter by location if specified
    if location and not matches.empty and 'Location' in matches.columns:
        location_matches = matches[matches['Location'].str.lower() == location.lower()]
        if not location_matches.empty:
            matches = location_matches
    
    return matches

def format_csv_results(team: str, matches: pd.DataFrame) -> str:
    """Format CSV query results into readable text"""
    
    if matches.empty:
        return ""
    
    # Check if multiple variants exist
    num_variants = len(matches)
    
    if num_variants == 1:
        # Single result - format as simple answer
        row = matches.iloc[0]
        result = f"""
Found charging guideline for **{team} Team - {row['Description']}**:

- **Description:** {row['Description']}
- **Company ID:** {row['Company ID']}
- **Location:** {row['Location']}
- **Department:** {row['Department']}
- **Project:** {row['Project']}
- **Account:** {row['Account'] if pd.notna(row['Account']) else 'Not specified'}
- **Account Name:** {row['Account Name'] if pd.notna(row['Account Name']) else 'Not specified'}
"""
    else:
        # Multiple variants - show all options
        result = f"""
Found **{num_variants} variants** for **{team} Team - {matches.iloc[0]['Description']}**:

⚠️ **Please specify which option applies to you:**

"""
        for idx, (_, row) in enumerate(matches.iterrows(), 1):
            result += f"""
**Option {idx}** - {row['Location']}:
- **Description:** {row['Description']}
- **Company ID:** {row['Company ID']}
- **Location:** {row['Location']}
- **Department:** {row['Department']}
- **Project:** {row['Project']}
- **Account:** {row['Account'] if pd.notna(row['Account']) else 'Not specified'}
- **Account Name:** {row['Account Name'] if pd.notna(row['Account Name']) else 'Not specified'}

"""
    
    return result.strip()

# ============================================
# EXTRACTION PROMPT
# ============================================

EXTRACTION_PROMPT = """
You are Diva, a charging guidelines assistant. Your job is to extract key information from the user's query.

Extract the following from the user's message and conversation history:
1. **team**: Which team (IT, Finance, HR, Operations, Engineering)
2. **description**: What they want to charge for (e.g., "Core ERP", "HR Labor", "Wind Maintenance")
3. **location**: Specific location/site if mentioned (e.g., "Houston", "DSOL", "Wind Site A")

Return ONLY valid JSON with this exact format:
{
  "team": "IT" | "Finance" | "HR" | "Operations" | "Engineering" | null,
  "description": "extracted description" | null,
  "location": "extracted location" | null
}

Examples:
- "how to charge core erp" → {"team": "IT", "description": "Core ERP", "location": null}
- "HR labor in Houston" → {"team": "HR", "description": "HR Labor", "location": "Houston"}
- "IT team DSOP support" → {"team": "IT", "description": "DSOP", "location": null}

If information is missing, use null. Be generous with partial matches on description.
"""

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

def extract_query_info(user_input: str) -> Dict:
    """Extract team, description, and location from user query using LLM"""
    
    try:
        mv = memory.load_memory_variables({})
        msgs = extraction_prompt.format_messages(
            chat_history=mv.get("chat_history", []), 
            input=user_input
        )
        resp = chat_model.invoke(msgs)
        content = resp.content.strip()
        
        # Strip code fences
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        
        return {
            "team": data.get("team"),
            "description": data.get("description"),
            "location": data.get("location")
        }
    except Exception as e:
        st.warning(f"Extraction error: {e}")
        return {"team": None, "description": None, "location": None}

# ============================================
# ROUTER SYSTEM
# ============================================

def route_turn(user_input: str) -> Dict:
    """Route user input to either clarify or answer"""
    
    text = user_input.lower().strip()
    
    # Quick greeting bypass
    GREETINGS = {"hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"}
    if text in GREETINGS or any(text.startswith(g) for g in GREETINGS):
        return {"intent": "answer", "missing": [], "extracted": {}}
    
    # Extract information from query
    extracted = extract_query_info(user_input)
    
    # Determine what's missing
    missing = []
    if not extracted.get("team"):
        missing.append("which team you're with (Operations, Engineering, Finance, IT, or HR)")
    if not extracted.get("description"):
        missing.append("what you're charging for (project name, activity, or work type)")
    
    # If we have team and description, we can answer
    if extracted.get("team") and extracted.get("description"):
        return {"intent": "answer", "missing": [], "extracted": extracted}
    
    # Otherwise, need clarification
    return {"intent": "clarify", "missing": missing[:2], "extracted": extracted}

# ============================================
# CLARIFICATION
# ============================================

def generate_clarification(missing: List[str]) -> str:
    """Generate a natural clarifying response"""
    if not missing:
        return "I can help with that! Could you provide more details?"
    
    if len(missing) == 1:
        q_text = missing[0] + "?"
    else:
        q_text = f"{missing[0]} and {missing[1]}?"
    
    return f"I can help with that! To find the right charging guideline, could you tell me {q_text}"

# ============================================
# ANSWER GENERATION
# ============================================

ANSWER_SYSTEM = """
You are Diva, Deriva's friendly charging guidelines assistant.

If the user is greeting you, respond warmly and offer to help with charging guidelines.

If providing charging information:
- Use the structured data provided
- Keep responses clear and concise
- If multiple options exist, present them clearly and ask which applies
- Always be helpful and friendly

Format your response naturally based on the data provided.
"""

answer_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM + "\n\nData from system:\n{data}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

def generate_answer(user_input: str, extracted: Dict) -> str:
    """Generate answer using CSV data"""
    
    # Handle greetings
    text = user_input.lower().strip()
    GREETINGS = {"hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"}
    if text in GREETINGS or any(text.startswith(g) for g in GREETINGS):
        greeting_data = "User is greeting. Respond warmly and offer to help with charging guidelines."
        mv = memory.load_memory_variables({})
        msgs = answer_prompt_template.format_messages(
            data=greeting_data,
            chat_history=mv["chat_history"],
            input=user_input
        )
        resp = chat_model.invoke(msgs)
        return resp.content
    
    # Query CSV data
    team = extracted.get("team")
    description = extracted.get("description")
    location = extracted.get("location")
    
    if not team or not description:
        return "I need more information to help you. Could you specify your team and what you're charging for?"
    
    matches = query_csv_data(team, description, location)
    
    if matches.empty:
        return f"I couldn't find charging guidelines for **{description}** in the **{team}** team. Could you verify the project/activity name or try rephrasing?"
    
    # Format results
    formatted_data = format_csv_results(team, matches)
    
    # Use LLM to create natural response
    mv = memory.load_memory_variables({})
    msgs = answer_prompt_template.format_messages(
        data=formatted_data,
        chat_history=mv["chat_history"],
        input=user_input
    )
    resp = chat_model.invoke(msgs)
    
    return resp.content

# ============================================
# RENDER EXISTING CHAT HISTORY
# ============================================

for m in chat_history_store.messages:
    role = "assistant" if m.type in ("ai", "assistant") else "user"
    with st.chat_message(role):
        st.markdown(m.content)

# ============================================
# CHAT LOOP
# ============================================

user_input = st.chat_input("Ask about charging codes, departments, projects, etc.")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Route decision
    decision = route_turn(user_input)
    
    # Handle clarification
    if decision["intent"] == "clarify" and decision.get("missing"):
        clarifier = generate_clarification(decision["missing"])
        
        # Save to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(clarifier)
        
        with st.chat_message("assistant"):
            st.markdown(clarifier)
    
    # Handle answer
    else:
        with st.chat_message("assistant"):
            with st.spinner("Looking up charging guidelines..."):
                answer = generate_answer(user_input, decision.get("extracted", {}))
                
                # Save to memory
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(answer)
                
                st.markdown(answer)

# ============================================
# FOOTER
# ============================================

st.divider()
footer = """
<style>
a:link, a:visited { color: blue; background-color: transparent; text-decoration: underline; }
a:hover, a:active { color: red; background-color: transparent; text-decoration: underline; }
.footer { position: fixed; left:0; bottom:0; width:100%; background-color:white; color:black; text-align:center; }
</style>
<div class="footer">
<p>Diva The Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
