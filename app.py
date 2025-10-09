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
    'Finance': 'csvs/Guidelines_cleaned_finance.csv',
    'HR': 'csvs/Guidelines_cleaned_hr.csv',
}

import os
import json
import uuid
import boto3
import streamlit as st
import pandas as pd
from typing import Dict, List
import time
import re

# --- Config ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")

# --- CSV File Paths ---
CSV_FILES = {
    'IT': 'csvs/Guidelines_cleaned_it.csv',
    'Finance': 'csvs/Guidelines_cleaned_finance.csv',
    'HR': 'csvs/Guidelines_cleaned_hr.csv',
}

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
            except Exception as e:
                st.error(f"Error loading {team} CSV: {e}")
                data[team] = pd.DataFrame()
        else:
            st.warning(f"CSV not found for {team}: {filepath}")
            data[team] = pd.DataFrame()
    return data

# Load CSVs at startup
ALL_TEAM_DATA = load_all_csvs()

# ============================================
# AWS CLIENTS & DYNAMODB SETUP
# ============================================

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

def create_dynamodb_table_if_not_exists():
    try:
        table = dynamodb.Table(DDB_TABLE_NAME)
        table.load()
    except dynamodb.meta.client.exceptions.ResourceNotFoundException:
        try:
            table = dynamodb.create_table(
                TableName=DDB_TABLE_NAME,
                KeySchema=[
                    {'AttributeName': 'session_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'message_timestamp', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'session_id', 'AttributeType': 'S'},
                    {'AttributeName': 'message_timestamp', 'AttributeType': 'S'}
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
# DYNAMODB CHAT HISTORY
# ============================================

class DynamoDBChatHistory:
    def __init__(self, table_name: str, session_id: str):
        self.table_name = table_name
        self.session_id = session_id
        self.dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        self.table = self.dynamodb.Table(table_name)
    
    def get_messages(self) -> List[Dict]:
        """Retrieve messages from DynamoDB"""
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id),
                ScanIndexForward=True
            )
            return response.get('Items', [])
        except Exception as e:
            st.warning(f"Could not load chat history: {e}")
            return []
    
    def add_message(self, role: str, content: str):
        """Add a message to DynamoDB"""
        try:
            self.table.put_item(
                Item={
                    'session_id': self.session_id,
                    'message_timestamp': str(int(time.time() * 1000)),
                    'message_type': role,
                    'content': content
                }
            )
        except Exception as e:
            st.error(f"Failed to save message: {e}")
    
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
    
    def get_formatted_history(self) -> str:
        """Get formatted history for Claude prompt"""
        messages = self.get_messages()
        if not messages:
            return "No previous conversation."
        
        history_text = "Previous conversation:\n"
        for msg in messages[-6:]:  # Last 6 messages (3 turns)
            role = msg.get('message_type', 'user')
            content = msg.get('content', '')
            history_text += f"{role.upper()}: {content}\n"
        return history_text

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

# Initialize chat history
chat_history = DynamoDBChatHistory(
    table_name=DDB_TABLE_NAME,
    session_id=st.session_state["session_id"]
)

def reset_history():
    try:
        chat_history.clear()
        st.success("Chat cleared!")
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
# LLM HELPER FUNCTIONS
# ============================================

def call_claude(system_prompt: str, user_message: str, include_history: bool = True) -> str:
    """Call Claude via Bedrock with conversation history"""
    try:
        # Build the message with history if needed
        if include_history:
            history = chat_history.get_formatted_history()
            if history != "No previous conversation.":
                full_message = history + f"\nCurrent question: {user_message}"
            else:
                full_message = user_message
        else:
            full_message = user_message
        
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": full_message
                    }
                ]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

# ============================================
# EXTRACTION PROMPT
# ============================================

EXTRACTION_PROMPT = """
You are Diva, a charging guidelines assistant. Your job is to extract key information from the user's query and conversation history.

Extract the following:
1. **team**: ONLY if explicitly mentioned (IT, Finance, HR, Operations, Engineering). Do NOT infer team from project names or descriptions.
2. **description**: What they want to charge for (e.g., "Core ERP", "HR Labor", "Wind Maintenance")
3. **location**: Specific location/site if mentioned (e.g., "DSOP", "DWOP", "DCS4", "DWE1", "STRG", "DSOL")

CRITICAL RULES:
- Look at ALL previous USER messages to find missing information
- Team must be EXPLICITLY stated by the user (e.g., "IT team", "I'm in Finance", "HR")
- If user previously mentioned a description (like "ERP" or "Core ERP"), KEEP that description even if current message doesn't mention it
- Do NOT guess team from project names like "Core ERP" or "Training"
- If user says "IT team Core ERP", then team: "IT"
- A short answer like "IT", "Finance", "Houston" is usually answering the assistant's clarification question
- Accumulate information across the conversation - don't lose context

Return ONLY valid JSON:
{
  "team": "IT" | "Finance" | "HR" | "Operations" | "Engineering" | null,
  "description": "extracted description" | null,
  "location": "extracted location" | null
}

Examples:
- "how to charge core erp" → {"team": null, "description": "Core ERP", "location": null}
- "IT team core erp" → {"team": "IT", "description": "Core ERP", "location": null}
- "I'm in finance, need accounting codes" → {"team": "Finance", "description": "accounting", "location": null}
- "HR labor in Houston" → {"team": null, "description": "HR Labor", "location": "Houston"}
- First: "I'm in HR" → {"team": "HR", "description": null, "location": null}
- Follow-up: "labor codes" → {"team": "HR", "description": "labor", "location": null}

Only extract team if the user explicitly mentions it in current or previous messages.
"""

def extract_query_info(user_query: str) -> Dict:
    """Extract team, description, and location from user query"""
    
    response = call_claude(EXTRACTION_PROMPT, user_query, include_history=True)
    
    if not response:
        return {"team": None, "description": None, "location": None}
    
    try:
        # Strip code fences if present
        content = response.strip()
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

        # Merge with existing context (accumulate information)
        merged = {}
        for key in ["team", "description", "location"]:
            # Use new value if present, otherwise keep old value
            if extracted.get(key):
                merged[key] = extracted[key]
            elif st.session_state.extracted_context.get(key):
                merged[key] = st.session_state.extracted_context[key]
            else:
                merged[key] = None
        
        # Update session state
        st.session_state.extracted_context = merged
        
        return merged
        
    except Exception as e:
        return st.session_state.extracted_context.copy()

# ============================================
# UPDATE RESET FUNCTION
# ============================================

def reset_history():
    try:
        chat_history.clear()
        # Clear extracted context
        st.session_state.extracted_context = {
            "team": None,
            "description": None,
            "location": None
        }
        st.success("Chat cleared!")
    except Exception as e:
        st.warning(f"Could not clear history: {e}")

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
    """Format CSV query results into consistent format"""
    
    if matches.empty:
        return ""
    
    # Check if multiple variants exist
    num_variants = len(matches)
    
    if num_variants == 1:
        # Single result - standard format
        row = matches.iloc[0]
        result = f"""**{team} Team - {row['Description']}**

**Description:** {row['Description']}
**Account number:** {row['Account'] if pd.notna(row['Account']) else 'Not specified'}
**Location:** {row['Location']}
**Company ID:** {row['Company ID']}
**Project:** {row['Project']}
**Department:** {row['Department']}"""
    else:
        # Multiple variants - show all with same format
        description = matches.iloc[0]['Description']
        result = f"""**{team} Team - {description}**

⚠️ **This charging code has {num_variants} variants. Please select the correct option based on your location/project:**

"""
        for idx, (_, row) in enumerate(matches.iterrows(), 1):
            differentiator = row['Location']
            
            result += f"""---
**OPTION {idx}: {differentiator}**

**Description:** {row['Description']}
**Account number:** {row['Account'] if pd.notna(row['Account']) else 'Not specified'}
**Location:** {row['Location']}
**Company ID:** {row['Company ID']}
**Project:** {row['Project']}
**Department:** {row['Department']}

"""
    
    return result.strip()

# ============================================
# ROUTER & CLARIFICATION
# ============================================

def needs_clarification(extracted: Dict) -> tuple:
    """Check if we need to ask clarifying questions"""
    missing = []
    
    if not extracted.get("team"):
        missing.append("team")
    if not extracted.get("description"):
        missing.append("description")
    
    needs_clarify = len(missing) > 0
    return needs_clarify, missing

def generate_clarification(missing: List[str], extracted: Dict) -> str:
    """Generate clarification question"""
    
    # If both team and description are missing
    if "team" in missing and "description" in missing:
        return "I can help with that! Could you tell me which team you're with (IT, Finance, HR, Operations, or Engineering) and what you're charging for?"
    
    # If only team is missing
    elif "team" in missing:
        return "Which team are you with? (IT, Finance, HR, Operations, or Engineering)"
    
    # If only description is missing
    elif "description" in missing:
        return "What project or activity are you charging for?"
    
    else:
        return "Could you provide more details?"

# ============================================
# MAIN CHAT LOGIC
# ============================================

def process_message(user_input: str) -> str:
    """Process user message and return response"""
    
    # Handle greetings
    greetings = ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]
    if user_input.lower().strip() in greetings or any(user_input.lower().strip().startswith(g) for g in greetings):
        return "Hi! I'm Diva, your charging guidelines assistant. How can I help you today? You can ask me about charging codes for any team or project."
    
    # Extract information
    extracted = extract_query_info(user_input)
    
    # Check if we need clarification
    needs_clarify, missing = needs_clarification(extracted)
    
    if needs_clarify:
        return generate_clarification(missing, extracted)
    
    # Query CSV data
    team = extracted.get("team")
    description = extracted.get("description")
    location = extracted.get("location")
    
    matches = query_csv_data(team, description, location)
    
    if matches.empty:
        return f"I couldn't find charging guidelines for '{description}' in the {team} team. Could you verify the project/activity name or try rephrasing?"
    
    # Format and return results
    result = format_csv_results(team, matches)
    
    if len(matches) > 1:
        result += "\n\n*Which location applies to you?*"
    
    return result

# ============================================
# RENDER EXISTING CHAT HISTORY
# ============================================

messages = chat_history.get_messages()
for msg in messages:
    role = "assistant" if msg.get('message_type') in ("ai", "assistant") else "user"
    content = msg.get('content', '')
    with st.chat_message(role):
        st.markdown(content)

# ============================================
# CHAT INPUT & PROCESSING
# ============================================

user_input = st.chat_input("Ask about charging codes, departments, projects, etc.")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Save user message
    chat_history.add_message("user", user_input)
    
    # Process and get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_message(user_input)
            st.markdown(response)
    
    # Save assistant response
    chat_history.add_message("assistant", response)

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
