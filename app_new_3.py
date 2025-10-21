import os
import json
import uuid
import boto3
import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
import time
import re

# --- Config ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")

# --- Charging Guidelines Link ---
CHARGING_GUIDELINES_LINK = "https://derivaenergy.sharepoint.com/:x:/r/sites/DerivaFinance/_layouts/15/Doc.aspx?sourcedoc=%7B3CD9F65D-C693-4CE8-904C-91074451F098%7D&file=Deriva%20OM%20Charging%20Guidelines.xlsx&action=default&mobileredirect=true"

# --- CSV File Paths ---
CSV_FILES = {
    'IT': 'csvs/Guidelines_cleaned_it.csv',
    'Finance': 'csvs/Guidelines_cleaned_Finance.csv',
    'HR': 'csvs/Guidelines_cleaned_HR.csv',
    'Legal': 'csvs/Guidelines_cleaned_Legal.csv',
    'Corporate': 'csvs/Guidelines_cleaned_Corporate.csv',
    'Land Services': 'csvs/Guidelines_cleaned_Land_Services.csv',
    'Commercial': 'csvs/Guidelines_cleaned_Commercial.csv',
    'Development': 'csvs/Guidelines_cleaned_Development.csv',
    'Tech Services': 'csvs/Guidelines_cleaned_Tech_Services.csv',
    # Operations sub-teams
    'Operations_High_Voltage': 'csvs/Guidelines_cleaned_Operation_High_Voltage.csv',
    'Operations_MST_FS': 'csvs/Guidelines_cleaned_Operation_MST_FS.csv',
    'Operations_MST_GBX': 'csvs/Guidelines_cleaned_Operation_MST_GBX.csv',
    'Operations_MST_LC': 'csvs/Guidelines_cleaned_Operation_MST_LC.csv',
    'Operations_Non_Controllable_QM': 'csvs/Guidelines_cleaned_Operation_Non_Controllable_QM.csv',
    'Operations_Ops_Support': 'csvs/Guidelines_cleaned_Operation_Ops_Support.csv',
    'Operations_Solar_Sites': 'csvs/Guidelines_cleaned_Operation_Solar_Sites.csv',
    'Operations_Wind_Sites': 'csvs/Guidelines_cleaned_Operation_Wind_Sites.csv',
    'Operations_Blade_Maintenance_Repair': 'csvs/Guidelines_cleaned_Operation_Blade_Maintenance_Repair.csv',
}


# --- Reference CSV Files ---
REFERENCE_CSV_FILES = {
    'department_list': 'csvs/Guidelines_cleaned_dept_list.csv',
    'activity_code': 'csvs/Guidelines_cleaned_Project_Activity_Sequence.csv'
}

# ============================================
# LOAD CSV DATA
# ============================================

@st.cache_data
def load_all_csvs():
    """Load all CSV files into memory"""
    data = {}
    
    # Load team-specific CSVs
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
    
    # Load reference CSVs
    for ref_type, filepath in REFERENCE_CSV_FILES.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                data[ref_type] = df
            except Exception as e:
                st.error(f"Error loading {ref_type} CSV: {e}")
                data[ref_type] = pd.DataFrame()
        else:
            st.warning(f"Reference CSV not found: {filepath}")
            data[ref_type] = pd.DataFrame()
    
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
        for msg in messages[-10:]:  # Last 10 messages (5 turns)
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
# SESSION STATE INITIALIZATION
# ============================================

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "extracted_context" not in st.session_state:
    st.session_state.extracted_context = {
        "team": None,
        "keywords": None,
        "location": None,
        "exact_description": None
    }

if "in_charging_flow" not in st.session_state:
    st.session_state.in_charging_flow = False

if "operations_search_results" not in st.session_state:
    st.session_state.operations_search_results = {}

if "operations_subteam" not in st.session_state:
    st.session_state.operations_subteam = None

# ============================================
# SIDEBAR
# ============================================

# Add logo at the very top of sidebar
if os.path.exists("Deriva-Logo.png"):
    st.sidebar.image("Deriva-Logo.png", width=200)

st.sidebar.title(" ")

# Initialize chat history
chat_history = DynamoDBChatHistory(
    table_name=DDB_TABLE_NAME,
    session_id=st.session_state["session_id"]
)

def reset_history():
    try:
        chat_history.clear()
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        st.success("Chat cleared!")
    except Exception as e:
        st.warning(f"Could not clear history: {e}")

with st.sidebar.expander("⚙️ Tools", expanded=True):
    if st.button("🗑️ Clear Chat"):
        reset_history()
        st.rerun()

with st.sidebar.expander("ℹ️ Charging Guidelines", expanded=False):
    st.markdown("""    
    About charging questions, Diva provides the following information based on description:
    - Account Number
    - Location
    - Company ID
    - Project (Concur, Timesheets)
    - Department    
    ---
    
    For additional info, please refer to [O&M Charging Guidelines](https://derivaenergy.sharepoint.com/:x:/r/sites/DerivaFinance/_layouts/15/Doc.aspx?sourcedoc=%7B3CD9F65D-C693-4CE8-904C-91074451F098%7D&file=Deriva%20OM%20Charging%20Guidelines.xlsx&action=default&mobileredirect=true).
    """)

st.sidebar.divider()
st.sidebar.caption("Diva The AI Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")

with st.sidebar.expander("📧 Support"):
    st.markdown("[Report an issue](mailto:joe.cheng@derivaenergy.com)")

# ============================================
# HEADER
# ============================================

st.markdown("<h1 style='text-align: center;'>⚡Meet Diva!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deriva's AI Chatbot for Charging Guidelines and More</p>", unsafe_allow_html=True)

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
# CHARGING QUESTION DETECTION
# ============================================

CHARGING_DETECTION_PROMPT = """
You are Diva, a charging guidelines assistant for Deriva Energy.

Determine if the user's question is about CHARGING GUIDELINES or something else.

Charging questions are about:
- How to charge time/expenses
- Charging codes, account numbers, projects, departments
- Where to charge specific work (ERP, labor, projects, etc.)
- Team-specific charging information

NOT charging questions:
- General greetings (hi, hello)
- General questions about Deriva Energy
- Questions about policies not related to charging
- Casual conversation
- Questions about other topics
- Department code lookups (that's reference info)
- Activity code lookups (that's reference info)

Return ONLY valid JSON:
{
  "is_charging_question": true | false,
  "confidence": "high" | "medium" | "low"
}

Examples:
- "how to charge erp" → {"is_charging_question": true, "confidence": "high"}
- "where do I charge labor?" → {"is_charging_question": true, "confidence": "high"}
- "IT team charging codes" → {"is_charging_question": true, "confidence": "high"}
- "what's the weather today?" → {"is_charging_question": false, "confidence": "high"}
- "tell me about Deriva Energy" → {"is_charging_question": false, "confidence": "high"}
- "who is the CEO?" → {"is_charging_question": false, "confidence": "high"}
"""

def is_charging_question(user_query: str) -> bool:
    """Detect if the question is about charging guidelines"""
    
    # Quick heuristic check first
    charging_keywords = [
        "charge", "charging", "code", "codes", "account", "project", 
        "expense", "time", "labor", "erp"
    ]
    
    user_lower = user_query.lower()
    has_charging_keyword = any(keyword in user_lower for keyword in charging_keywords)
    
    # If no charging keywords and it's a question, likely not charging
    if not has_charging_keyword and len(user_query.split()) > 3:
        return False
    
    # Use LLM for uncertain cases
    response = call_claude(CHARGING_DETECTION_PROMPT, user_query, include_history=False)
    
    if not response:
        # Default to charging question if LLM fails and has keywords
        return has_charging_keyword
    
    try:
        content = response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        
        return data.get("is_charging_question", has_charging_keyword)
    except:
        return has_charging_keyword

# ============================================
# REFERENCE QUESTION DETECTION
# ============================================

REFERENCE_DETECTION_PROMPT = """
You are Diva, an assistant for Deriva Energy.

Determine if the user is asking about REFERENCE INFORMATION (department codes, activity codes, etc.).

Reference questions are about:
- Department codes/numbers and what they mean
- Activity codes and their descriptions
- Looking up what a specific code means
- Questions like "what department is code X?" or "what does activity code Y mean?"

NOT reference questions:
- Charging guidelines questions (how to charge, where to charge)
- General conversation
- Greetings

Return ONLY valid JSON:
{
  "is_reference_question": true | false,
  "reference_type": "department" | "activity" | null,
  "confidence": "high" | "medium" | "low"
}

Examples:
- "what department is 1001?" → {"is_reference_question": true, "reference_type": "department", "confidence": "high"}
- "what does activity code 200 mean?" → {"is_reference_question": true, "reference_type": "activity", "confidence": "high"}
- "explain department codes" → {"is_reference_question": true, "reference_type": "department", "confidence": "high"}
- "how to charge erp" → {"is_reference_question": false, "reference_type": null, "confidence": "high"}
"""

def is_reference_question(user_query: str) -> Tuple[bool, str]:
    """
    Detect if the question is about reference data (departments, activity codes).
    Returns: (is_reference, reference_type)
    """
    user_lower = user_query.lower()
    
    # Quick heuristic check
    department_keywords = ["department", "dept", "department code", "department number"]
    activity_keywords = ["activity code", "activity", "code meaning"]
    
    has_dept_keyword = any(keyword in user_lower for keyword in department_keywords)
    has_activity_keyword = any(keyword in user_lower for keyword in activity_keywords)
    
    # Use LLM for better detection
    response = call_claude(REFERENCE_DETECTION_PROMPT, user_query, include_history=False)
    
    if not response:
        # Fallback to heuristics
        if has_dept_keyword:
            return True, "department"
        if has_activity_keyword:
            return True, "activity"
        return False, None
    
    try:
        content = response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        
        is_ref = data.get("is_reference_question", False)
        ref_type = data.get("reference_type")
        
        return is_ref, ref_type
    except:
        if has_dept_keyword:
            return True, "department"
        if has_activity_keyword:
            return True, "activity"
        return False, None

# ============================================
# NATURAL LANGUAGE RESPONSE
# ============================================

GENERAL_ASSISTANT_PROMPT = """
You are Diva, a friendly and helpful assistant for Deriva Energy employees.

IMPORTANT CONTEXT:
- "Charging" in this context means TIME CHARGING and EXPENSE CHARGING for accounting/billing purposes
- This is about charging time to projects, departments, and cost centers
- This is NOT about electric vehicle charging or battery charging
- You help with: timesheet codes, expense report codes, project codes, department codes, account numbers

Your primary purpose is to help with charging guidelines, but you can also answer general questions conversationally.

Guidelines:
- Be friendly, concise, and professional
- For questions about Deriva Energy that you don't know, say you're primarily designed for charging guidelines
- Keep responses brief (2-4 sentences unless more detail is needed)
- If the question seems related to charging, gently guide them: "If you're asking about charging guidelines, I can help with that!"
- NEVER make up information you don't know
- If you're unsure, be honest about it

You are an internal tool for Deriva Energy employees.
"""

def generate_natural_response(user_query: str) -> str:
    """Generate natural language response for non-charging questions"""
    
    response = call_claude(GENERAL_ASSISTANT_PROMPT, user_query, include_history=True)
    
    if not response:
        return "I'm here to help! My specialty is charging guidelines. Could you rephrase your question or ask about charging codes?"
    
    return response

# ============================================
# EXTRACTION PROMPT
# ============================================

# EXTRACTION_PROMPT = """
# You are Diva, a charging guidelines assistant. Extract key information from the user's query.

# Extract:
# 1. **team**: ONLY if explicitly mentioned (IT, Finance, HR, Legal, Corporate, Land Services, Commercial, Development, Tech Services, Operations)
# 2. **keywords**: Key words the user wants to search for (e.g., "erp", "labor", "maintenance")
# 3. **location**: Specific location if mentioned (e.g., "DSOP", "DSOL")
# 4. **is_new_query**: Is this a NEW charging question or a follow-up/clarification? (true/false)

# RULES FOR is_new_query:
# - TRUE if: User asks "how to charge X", "where to charge Y", "codes for Z", or any new charging question
# - FALSE if: User gives short answers like "IT", "Houston", "1", "option 2" (these are clarifications)
# - TRUE if: User asks about a DIFFERENT project/activity than previous conversation
# - FALSE if: User is answering assistant's clarification questions

# Return ONLY valid JSON:
# {
#   "team": "IT" | "Finance" | "HR" | "Legal" | "Corporate" | "Land Services" | "Commercial" | "Development" | "Tech Services" | "Operations" | null,
#   "keywords": "search terms" | null,
#   "location": "location name" | null,
#   "is_new_query": true | false
# }

# Examples:
# - "how to charge erp" → {"team": null, "keywords": "erp", "location": null, "is_new_query": true}
# - Previous asked team, Current: "IT" → {"team": "IT", "keywords": null, "location": null, "is_new_query": false}
# - Previous: "Core ERP codes", Current: "what about HR labor?" → {"team": null, "keywords": "labor", "location": null, "is_new_query": true}
# - Previous: showed 3 locations, Current: "DSOL" → {"team": null, "keywords": null, "location": "DSOL", "is_new_query": false}
# """

EXTRACTION_PROMPT = """
You are Diva, a charging guidelines assistant. Extract key information from the user's query.

Extract:
1. **team**: ONLY if explicitly mentioned (IT, Finance, HR, Legal, Corporate, Land Services, Commercial, Development, Tech Services, Operations)
2. **keywords**: Key words the user wants to search for (e.g., "erp", "labor", "maintenance")
3. **location**: Specific location if mentioned (e.g., "DSOP", "DSOL")
4. **operation_type**: ONLY if Operations team AND user specifies type:
   - "high_voltage" (High Voltage, HV)
   - "mst_fs" (MST FS, Field Service)
   - "mst_gbx" (MST GBX, Gearbox)
   - "mst_lc" (MST LC, Load Center)
   - "non_controllable_qm" (Non-Controllable QM, Quality and Maintenance)
   - "ops_support" (Operations Support, Ops Support)
   - "solar_sites" (Solar Sites, Solar)
   - "wind_sites" (Wind Sites, Wind)
   - "blade_maintenance" (Blade Maintenance & Repair, Blade)
5. **is_new_query**: Is this a NEW charging question or a follow-up/clarification? (true/false)

RULES FOR is_new_query:
- TRUE if: User asks "how to charge X", "where to charge Y", "codes for Z", or any new charging question
- FALSE if: User gives short answers like "IT", "Houston", "1", "option 2", "wind", "solar" (these are clarifications)
- TRUE if: User asks about a DIFFERENT project/activity than previous conversation
- FALSE if: User is answering assistant's clarification questions

Return ONLY valid JSON:
{
  "team": "IT" | "Finance" | "HR" | "Legal" | "Corporate" | "Land Services" | "Commercial" | "Development" | "Tech Services" | "Operations" | null,
  "keywords": "search terms" | null,
  "location": "location name" | null,
  "operation_type": "high_voltage" | "mst_fs" | "mst_gbx" | "mst_lc" | "non_controllable_qm" | "ops_support" | "solar_sites" | "wind_sites" | "blade_maintenance" | null,
  "is_new_query": true | false
}

Examples:
- "how to charge erp" → {"team": null, "keywords": "erp", "location": null, "operation_type": null, "is_new_query": true}
- Previous asked team, Current: "Operations" → {"team": "Operations", "keywords": null, "location": null, "operation_type": null, "is_new_query": false}
- Previous asked operation type, Current: "wind sites" → {"team": null, "keywords": null, "location": null, "operation_type": "wind_sites", "is_new_query": false}
- "Operations solar maintenance" → {"team": "Operations", "keywords": "maintenance", "location": null, "operation_type": "solar_sites", "is_new_query": true}
- Previous: "Operations team", Current: "blade" → {"team": null, "keywords": null, "location": null, "operation_type": "blade_maintenance", "is_new_query": false}
"""


# ============================================
# EXTRACTION - RESET TEAM FOR NEW QUERIES
# ============================================

def extract_query_info(user_query: str) -> Dict:
    """Extract team, keywords, and location from user query"""
    
    response = call_claude(EXTRACTION_PROMPT, user_query, include_history=True)
    
    if not response:
        return st.session_state.extracted_context.copy()
    
    try:
        content = response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        
        is_new_query = data.get("is_new_query", False)
        
        # If it's a new query, COMPLETELY RESET context (including team)
        if is_new_query:
            extracted = {
                "team": data.get("team"),
                "keywords": data.get("keywords"),
                "location": data.get("location"),
                "exact_description": None
            }
            
            st.session_state.extracted_context = extracted
            return extracted
        
        # If it's a follow-up, merge with existing context
        extracted = {
            "team": data.get("team"),
            "keywords": data.get("keywords"),
            "location": data.get("location"),
            "exact_description": st.session_state.extracted_context.get("exact_description")
        }
        
        # Merge with existing context
        merged = {}
        for key in ["team", "keywords", "location", "exact_description"]:
            if extracted.get(key):
                merged[key] = extracted[key]
            elif st.session_state.extracted_context.get(key):
                merged[key] = st.session_state.extracted_context[key]
            else:
                merged[key] = None
        
        st.session_state.extracted_context = merged
        return merged
        
    except Exception as e:
        return st.session_state.extracted_context.copy()

# ============================================
# HELPER: DETECT NEW QUERY (FALLBACK)
# ============================================

def is_likely_new_query(user_input: str) -> bool:
    """Fallback detection if LLM extraction fails"""
    user_lower = user_input.lower().strip()
    
    # Phrases that indicate new queries
    new_query_phrases = [
        "how to charge",
        "where to charge",
        "charge for",
        "charging for",
        "codes for",
        "what about",
        "how about",
        "need codes",
        "looking for"
    ]
    
    for phrase in new_query_phrases:
        if phrase in user_lower:
            return True
    
    # If it's a short answer (likely clarification)
    if len(user_input.split()) <= 3:
        return False
    
    # If it contains question words, likely new
    question_words = ["how", "what", "where", "which", "can", "do"]
    first_word = user_lower.split()[0] if user_lower.split() else ""
    if first_word in question_words:
        return True
    
    return False

# ============================================
# HELPER FUNCTION: Clean and format values
# ============================================
def clean_value(value, field_name=""):
    """
    Clean a value for display:
    - Convert NaN/None to "N/A"
    - Remove .0 from integers (especially Account numbers)
    - Keep strings as-is
    """
    # Handle NaN, None, or empty strings
    if pd.isna(value) or value is None or str(value).strip() == "":
        return "N/A"
    
    # Handle numeric values
    if isinstance(value, (int, float, np.integer, np.floating)):
        # Check if it's actually an integer (no decimal part)
        if float(value) == int(value):
            return str(int(value))
        else:
            return str(value)
    
    # Handle string values
    value_str = str(value).strip()
    
    # Check if string looks like "123.0" and convert to integer
    if field_name.lower() == "account" or field_name.lower() == "company id":
        try:
            float_val = float(value_str)
            if float_val == int(float_val):
                return str(int(float_val))
        except (ValueError, OverflowError):
            pass
    
    return value_str

def check_for_xxxx_in_project(project_value):
    """
    Check if project contains 'XXXX' or 'xxxx' and return appropriate note.
    Returns tuple: (has_xxxx, note_text)
    """
    if pd.isna(project_value):
        return False, ""
    
    project_str = str(project_value).strip()
    
    # Check for XXXX in various forms
    if 'XXXX' in project_str.upper():
        return True, "\n\n**Note:** XXXX = Site location code"
    
    return False, ""

# ============================================
# CSV SEARCH FUNCTIONS (WHOLE WORD MATCHING)
# ============================================

def search_descriptions_by_keywords(team: str, keywords: str) -> List[str]:
    """Search for descriptions containing whole words from keywords"""
    if team not in ALL_TEAM_DATA or ALL_TEAM_DATA[team].empty:
        return []
    
    df = ALL_TEAM_DATA[team]
    
    # Split keywords into individual words
    search_words = [w.strip().lower() for w in keywords.split() if w.strip()]
    
    matching_descriptions = set()
    
    for idx, row in df.iterrows():
        description = str(row['Description']).lower()
        description_words = re.findall(r'\b\w+\b', description)
        
        # Check if any search word matches a whole word in description
        for search_word in search_words:
            if search_word in description_words:
                matching_descriptions.add(row['Description'])
                break
    
    return sorted(list(matching_descriptions))

def get_charging_data(team: str, exact_description: str, location: str = None) -> Tuple[pd.DataFrame, bool]:
    """Get charging data for exact description"""
    if team not in ALL_TEAM_DATA or ALL_TEAM_DATA[team].empty:
        return pd.DataFrame(), False
    
    df = ALL_TEAM_DATA[team]
    
    # Exact match on description
    matches = df[df['Description'] == exact_description]
    
    if matches.empty:
        return pd.DataFrame(), False
    
    # Check if multiple locations
    has_multiple = len(matches) > 1
    
    # Filter by location if specified and multiple exist
    if location and has_multiple:
        location_matches = matches[matches['Location'].str.lower() == location.lower()]
        if not location_matches.empty:
            matches = location_matches
    
    return matches, has_multiple

# ============================================
# OPERATIONS-SPECIFIC SEARCH
# ============================================

def search_operations_by_description(keywords: str, subteam: str = None) -> Dict[str, List[pd.Series]]:
    """
    Search Operations team by description and group results by description.
    If subteam is provided, only search that specific subteam.
    Returns a dictionary where keys are unique descriptions and values are lists of matching rows.
    """
    # Get all Operations sub-teams or just the specified one
    if subteam:
        operations_teams = [subteam] if subteam in ALL_TEAM_DATA else []
    else:
        operations_teams = [team for team in ALL_TEAM_DATA.keys() if team.startswith('Operations_')]
    
    if not operations_teams:
        return {}
    
    # Split keywords into individual words
    search_words = [w.strip().lower() for w in keywords.split() if w.strip()]
    
    # Find all matching rows across all Operations sub-teams
    matching_rows = []
    for ops_team in operations_teams:
        df = ALL_TEAM_DATA[ops_team]
        if df.empty:
            continue
            
        for idx, row in df.iterrows():
            description = str(row['Description']).lower()
            description_words = re.findall(r'\b\w+\b', description)
            
            # Check if any search word matches a whole word in description
            for search_word in search_words:
                if search_word in description_words:
                    matching_rows.append(row)
                    break
    
    # Group by description
    grouped_results = {}
    for row in matching_rows:
        desc = row['Description']
        if desc not in grouped_results:
            grouped_results[desc] = []
        grouped_results[desc].append(row)
    
    return grouped_results

# ============================================
# Function to get operations subteams for a description
# ============================================
def get_operations_subteams_for_description(description: str) -> Dict[str, List[pd.Series]]:
    """
    Find which Operations sub-teams have a specific description.
    Returns dict with subteam names as keys and matching rows as values.
    """
    subteam_results = {}
    
    operations_teams = [team for team in ALL_TEAM_DATA.keys() if team.startswith('Operations_')]
    
    for ops_team in operations_teams:
        df = ALL_TEAM_DATA[ops_team]
        if df.empty:
            continue
        
        # Find rows with matching description (case-insensitive)
        matches = df[df['Description'].str.lower() == description.lower()]
        
        if not matches.empty:
            # Convert team name to readable format
            readable_name = ops_team.replace('Operations_', '').replace('_', ' ')
            subteam_results[readable_name] = matches.to_dict('records')
    
    return subteam_results

# ============================================
# Function to format operations subteam list
# ============================================
def format_operations_subteam_question(description: str, subteam_results: Dict) -> str:
    """
    Format a clarifying question asking which Operations sub-team the user needs.
    """
    subteam_names = list(subteam_results.keys())
    
    result = f"I found charging codes for **'{description}'** in multiple Operations teams:\n\n"
    
    for idx, subteam in enumerate(subteam_names, 1):
        result += f"{idx}. {subteam}\n"
    
    result += f"\nWhich Operations team are you asking about?"
    
    return result

# ============================================
# Function to detect operations subteam from user input
# ============================================
def extract_operations_subteam(user_input: str) -> str:
    """
    Extract Operations sub-team from user input.
    Returns the full team key (e.g., 'Operations_Wind_Sites') or None.
    """
    user_lower = user_input.lower()
    
    # Map of keywords to team names
    subteam_mapping = {
        'high voltage': 'Operations_High_Voltage',
        'mst fs': 'Operations_MST_FS',
        'fs': 'Operations_MST_FS',
        'mst gbx': 'Operations_MST_GBX',
        'gbx': 'Operations_MST_GBX',
        'gearbox': 'Operations_MST_GBX',
        'mst lc': 'Operations_MST_LC',
        'lc': 'Operations_MST_LC',
        'non controllable': 'Operations_Non_Controllable_QM',
        'qm': 'Operations_Non_Controllable_QM',
        'ops support': 'Operations_Ops_Support',
        'operations support': 'Operations_Ops_Support',
        'solar': 'Operations_Solar_Sites',
        'solar sites': 'Operations_Solar_Sites',
        'wind': 'Operations_Wind_Sites',
        'wind sites': 'Operations_Wind_Sites',
        'blade': 'Operations_Blade_Maintenance_Repair',
        'blade maintenance': 'Operations_Blade_Maintenance_Repair',
        'blade repair': 'Operations_Blade_Maintenance_Repair',
    }
    
    # Check for matches
    for keyword, team_key in subteam_mapping.items():
        if keyword in user_lower:
            return team_key
    
    # Check if user input is a number (selecting from list)
    user_input_stripped = user_input.strip()
    if user_input_stripped.isdigit():
        # This will be handled by the selection logic
        return None
    
    return None

# ============================================
# Handle operations subteam selection
# ============================================
def handle_operations_subteam_selection(user_input: str, pending_subteams: Dict, description: str):
    """
    Handle when user selects an Operations sub-team from a list.
    """
    user_input_stripped = user_input.strip()
    
    # Check if user input is a number
    if user_input_stripped.isdigit():
        selection_num = int(user_input_stripped)
        subteam_names = list(pending_subteams.keys())
        
        if 1 <= selection_num <= len(subteam_names):
            selected_subteam = subteam_names[selection_num - 1]
            rows = pending_subteams[selected_subteam]
            
            # If only one variant, return it directly
            if len(rows) == 1:
                st.session_state.extracted_context = {
                    "team": None,
                    "keywords": None,
                    "location": None,
                    "exact_description": None
                }
                st.session_state.operations_subteam = None
                st.session_state.pending_operations_subteams = {}
                st.session_state.in_charging_flow = False
                return format_charging_info(rows[0])
            else:
                # Multiple variants - show all
                result = f"**Operations - {selected_subteam} - {description}**\n\n"
                result += f"This charging code has **{len(rows)} variants**:\n\n"
                
                for idx, row in enumerate(rows, 1):
                    result += f"---\n**VARIANT {idx}:**\n"
                    result += f"- **Description:** {row['Description']}\n"
                    result += f"- **Account:** {row['Account']}\n"
                    result += f"- **Location:** {row['Location']}\n"
                    result += f"- **Company ID:** {row['Company ID']}\n"
                    result += f"- **Project:** {row['Project']}\n"
                    result += f"- **Department:** {row['Department']}\n\n"
                
                st.session_state.extracted_context = {
                    "team": None,
                    "keywords": None,
                    "location": None,
                    "exact_description": None
                }
                st.session_state.operations_subteam = None
                st.session_state.pending_operations_subteams = {}
                st.session_state.in_charging_flow = False
                return result.strip()
    
    # Check if user mentioned a subteam name
    for subteam_name in pending_subteams.keys():
        if subteam_name.lower() in user_input.lower():
            rows = pending_subteams[subteam_name]
            
            if len(rows) == 1:
                st.session_state.extracted_context = {
                    "team": None,
                    "keywords": None,
                    "location": None,
                    "exact_description": None
                }
                st.session_state.operations_subteam = None
                st.session_state.pending_operations_subteams = {}
                st.session_state.in_charging_flow = False
                return format_charging_info(rows[0])
            else:
                # Multiple variants
                result = f"**Operations - {subteam_name} - {description}**\n\n"
                result += f"This charging code has **{len(rows)} variants**:\n\n"
                
                for idx, row in enumerate(rows, 1):
                    result += f"---\n**VARIANT {idx}:**\n"
                    result += f"- **Description:** {row['Description']}\n"
                    result += f"- **Account:** {row['Account']}\n"
                    result += f"- **Location:** {row['Location']}\n"
                    result += f"- **Company ID:** {row['Company ID']}\n"
                    result += f"- **Project:** {row['Project']}\n"
                    result += f"- **Department:** {row['Department']}\n\n"
                
                st.session_state.extracted_context = {
                    "team": None,
                    "keywords": None,
                    "location": None,
                    "exact_description": None
                }
                st.session_state.operations_subteam = None
                st.session_state.pending_operations_subteams = {}
                st.session_state.in_charging_flow = False
                return result.strip()
    
    return None


def format_operations_results(grouped_results: Dict[str, List[pd.Series]]) -> str:
    """
    Format Operations results showing all related info for each description.
    """
    if not grouped_results:
        return f"I couldn't find any matching Operations charging codes.\n\nPlease refer to the [O&M Charging Guidelines]({CHARGING_GUIDELINES_LINK}) for more information."
    
    # If only one unique description found
    if len(grouped_results) == 1:
        description = list(grouped_results.keys())[0]
        variants = grouped_results[description]
        
        if len(variants) == 1:
            # Single unique result
            row = variants[0]
            result = f"**Operations - {description}**\n\n"
            result += format_charging_info(row)
            return result
        else:
            # Multiple variants of the same description
            result = f"**Operations - {description}**\n\n"
            result += f"This charging code has **{len(variants)} variants** based on department/location:\n\n"
            
            # Track if any variant has XXXX
            has_any_xxxx = False
            
            for idx, row in enumerate(variants, 1):
                # Clean all values
                desc = clean_value(row.get('Description'), 'description')
                account = clean_value(row.get('Account'), 'account')
                location = clean_value(row.get('Location'), 'location')
                company_id = clean_value(row.get('Company ID'), 'company id')
                project = clean_value(row.get('Project'), 'project')
                department = clean_value(row.get('Department'), 'department')
                
                # Check for XXXX
                has_xxxx, _ = check_for_xxxx_in_project(project)
                if has_xxxx:
                    has_any_xxxx = True
                
                result += f"---\n**VARIANT {idx}:**\n"
                result += f"- **Description:** {desc}\n"
                result += f"- **Account:** {account}\n"
                result += f"- **Location:** {location}\n"
                result += f"- **Company ID:** {company_id}\n"
                result += f"- **Project:** {project}\n"
                result += f"- **Department:** {department}\n\n"
            
            # Add XXXX note at the end if any variant has it
            if has_any_xxxx:
                result += "\n**Note:** XXXX = Site location code\n"
            
            return result.strip()
    
    # Multiple different descriptions found
    result = f"I found **{len(grouped_results)} different charging codes** in Operations:\n\n"
    
    for idx, description in enumerate(grouped_results.keys(), 1):
        variant_count = len(grouped_results[description])
        if variant_count == 1:
            result += f"{idx}. {description}\n"
        else:
            result += f"{idx}. {description} ({variant_count} variants)\n"
    
    result += "\nWhich one are you looking for? (You can reply with the number or name)"
    
    return result


def handle_operations_selection(user_input: str, grouped_results: Dict[str, List[pd.Series]]) -> str:
    """
    Handle user selecting from multiple Operations descriptions.
    """
    descriptions = list(grouped_results.keys())
    
    # Check if user input is a number
    user_input_clean = user_input.strip()
    if user_input_clean.isdigit():
        idx = int(user_input_clean) - 1
        if 0 <= idx < len(descriptions):
            selected_desc = descriptions[idx]
            variants = grouped_results[selected_desc]
            
            # Store the selected description
            st.session_state.extracted_context["exact_description"] = selected_desc
            
            if len(variants) == 1:
                row = variants[0]
                return format_charging_info(row)
            else:
                result = f"**Operations - {selected_desc}**\n\n"
                result += f"This charging code has **{len(variants)} variants**:\n\n"
                
                for idx, row in enumerate(variants, 1):
                    result += f"---\n**VARIANT {idx}:**\n"
                    result += f"- **Description:** {row['Description']}\n"
                    result += f"- **Account:** {row['Account']}\n"
                    result += f"- **Location:** {row['Location']}\n"
                    result += f"- **Company ID:** {row['Company ID']}\n"
                    result += f"- **Project:** {row['Project']}\n"
                    result += f"- **Department:** {row['Department']}\n\n"
                
                return result.strip()
    
    # Check if user input matches one of the descriptions
    user_lower = user_input_clean.lower()
    for desc in descriptions:
        if desc.lower() == user_lower or desc.lower() in user_lower or user_lower in desc.lower():
            variants = grouped_results[desc]
            
            st.session_state.extracted_context["exact_description"] = desc
            
            if len(variants) == 1:
                row = variants[0]
                return format_charging_info(row)
            else:
                result = f"**Operations - {desc}**\n\n"
                result += f"This charging code has **{len(variants)} variants**:\n\n"
                
                for idx, row in enumerate(variants, 1):
                    result += f"---\n**VARIANT {idx}:**\n"
                    result += f"- **Description:** {row['Description']}\n"
                    result += f"- **Account:** {row['Account']}\n"
                    result += f"- **Location:** {row['Location']}\n"
                    result += f"- **Company ID:** {row['Company ID']}\n"
                    result += f"- **Project:** {row['Project']}\n"
                    result += f"- **Department:** {row['Department']}\n\n"
                
                return result.strip()
    
    return None

# ============================================
# REFERENCE DATA LOOKUP FUNCTIONS
# ============================================

def search_department_list(search_term: str) -> pd.DataFrame:
    """Search department list by code or description"""
    if 'department_list' not in ALL_TEAM_DATA or ALL_TEAM_DATA['department_list'].empty:
        return pd.DataFrame()
    
    df = ALL_TEAM_DATA['department_list']
    search_lower = search_term.lower().strip()
    
    # Try exact match on code first (adjust column names based on your CSV)
    code_col = 'Department Code' if 'Department Code' in df.columns else df.columns[0]
    desc_col = 'Department Name' if 'Department Name' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    # Search by code (exact match)
    if search_term.replace('-', '').replace('_', '').isdigit():
        matches = df[df[code_col].astype(str).str.replace('-', '').str.replace('_', '') == search_term.replace('-', '').replace('_', '')]
        if not matches.empty:
            return matches
    
    # Search by description (partial match)
    matches = df[df[desc_col].astype(str).str.lower().str.contains(search_lower, na=False)]
    
    return matches

def search_activity_codes(search_term: str) -> pd.DataFrame:
    """Search activity codes by code or description"""
    if 'activity_code' not in ALL_TEAM_DATA or ALL_TEAM_DATA['activity_code'].empty:
        return pd.DataFrame()
    
    df = ALL_TEAM_DATA['activity_code']
    search_lower = search_term.lower().strip()
    
    # Adjust column names based on your CSV
    code_col = 'Activity Code' if 'Activity Code' in df.columns else df.columns[0]
    desc_col = 'Activity Description' if 'Activity Description' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    # Search by code (exact match)
    if search_term.replace('-', '').replace('_', '').isdigit():
        matches = df[df[code_col].astype(str).str.replace('-', '').str.replace('_', '') == search_term.replace('-', '').replace('_', '')]
        if not matches.empty:
            return matches
    
    # Search by description (partial match)
    matches = df[df[desc_col].astype(str).str.lower().str.contains(search_lower, na=False)]
    
    return matches

def format_reference_results(matches: pd.DataFrame, reference_type: str) -> str:
    """Format reference lookup results"""
    if matches.empty:
        return f"I couldn't find any {reference_type} information matching your search.\n\nPlease refer to the [O&M Charging Guidelines]({CHARGING_GUIDELINES_LINK}) for more information."
    
    if len(matches) == 1:
        row = matches.iloc[0]
        result = f"**{reference_type.title()} Information:**\n\n"
        for col in matches.columns:
            result += f"- **{col}:** {row[col]}\n"
        return result
    
    # Multiple matches
    result = f"I found **{len(matches)} {reference_type} entries**:\n\n"
    for idx, (_, row) in enumerate(matches.iterrows(), 1):
        result += f"**{idx}. "
        # Use first column as title
        result += f"{row.iloc[0]}**\n"
        for col in matches.columns[1:]:
            result += f"   - {col}: {row[col]}\n"
        result += "\n"
    
    return result.strip()

# ============================================
# REFERENCE QUESTION PROCESSING
# ============================================

def process_reference_question(user_input: str, reference_type: str) -> str:
    """Process reference lookup questions (department, activity codes)"""
    
    # Extract search term from user input
    search_term = user_input
    
    # Remove common question words
    for word in ["what", "is", "does", "mean", "means", "code", "department", "activity", "the", "a", "an", "tell", "me", "about"]:
        search_term = re.sub(rf'\b{word}\b', '', search_term, flags=re.IGNORECASE)
    
    search_term = search_term.strip()
    
    if not search_term:
        if reference_type == "department":
            return "What department code or name would you like to look up?"
        elif reference_type == "activity":
            return "What activity code would you like to look up?"
        else:
            return "What would you like to look up?"
    
    # Perform lookup
    if reference_type == "department":
        matches = search_department_list(search_term)
        return format_reference_results(matches, "department")
    elif reference_type == "activity":
        matches = search_activity_codes(search_term)
        return format_reference_results(matches, "activity code")
    else:
        return f"I'm not sure what type of information you're looking for. Could you clarify?\n\nYou can also refer to the [O&M Charging Guidelines]({CHARGING_GUIDELINES_LINK})."

# ============================================
# FORMATTING FUNCTIONS
# ============================================

def format_charging_info(row: pd.Series) -> str:
    """Format a single charging code with markdown bullets"""
    
    # Clean all values
    description = clean_value(row.get('Description'), 'description')
    account = clean_value(row.get('Account'), 'account')
    location = clean_value(row.get('Location'), 'location')
    company_id = clean_value(row.get('Company ID'), 'company id')
    project = clean_value(row.get('Project'), 'project')
    department = clean_value(row.get('Department'), 'department')
    
    # Check for XXXX in project
    has_xxxx, xxxx_note = check_for_xxxx_in_project(project)
    
    result = f"""- **Description:** {description}
- **Account:** {account}
- **Location:** {location}
- **Company ID:** {company_id}
- **Project:** {project}
- **Department:** {department}"""
    
    # Add XXXX note if needed
    if has_xxxx:
        result += xxxx_note
    
    return result

def format_multiple_variants(team: str, matches: pd.DataFrame) -> str:
    """Format multiple variants of the same description"""
    description = clean_value(matches.iloc[0].get('Description'), 'description')
    
    result = f"**{team} Team - {description}**\n\n"
    result += f"This charging code has **{len(matches)} options**. Please refer to the department list for more details:\n\n"
    
    # Track if any variant has XXXX
    has_any_xxxx = False
    
    for idx, (_, row) in enumerate(matches.iterrows(), 1):
        # Clean all values
        desc = clean_value(row.get('Description'), 'description')
        account = clean_value(row.get('Account'), 'account')
        location = clean_value(row.get('Location'), 'location')
        company_id = clean_value(row.get('Company ID'), 'company id')
        project = clean_value(row.get('Project'), 'project')
        department = clean_value(row.get('Department'), 'department')
        
        # Check for XXXX
        has_xxxx, _ = check_for_xxxx_in_project(project)
        if has_xxxx:
            has_any_xxxx = True
        
        result += f"---\n**OPTION {idx}:**\n"
        result += f"- **Description:** {desc}\n"
        result += f"- **Account:** {account}\n"
        result += f"- **Location:** {location}\n"
        result += f"- **Company ID:** {company_id}\n"
        result += f"- **Project:** {project}\n"
        result += f"- **Department:** {department}\n\n"
    
    # Add XXXX note at the end if any variant has it
    if has_any_xxxx:
        result += "\n**Note:** XXXX = Site location code\n"
    
    return result.strip()


# ============================================
# HANDLE USER SELECTING FROM LIST
# ============================================

def check_if_selecting_from_list(user_input: str, extracted: Dict) -> str:
    """Check if user is selecting a description from a list"""
    team = extracted.get("team")
    keywords = extracted.get("keywords")
    
    if not team or not keywords:
        return None
    
    # Get matching descriptions
    matching_descriptions = search_descriptions_by_keywords(team, keywords)
    
    if len(matching_descriptions) <= 1:
        return None
    
    # Check if user input is a number
    user_input_clean = user_input.strip()
    if user_input_clean.isdigit():
        idx = int(user_input_clean) - 1
        if 0 <= idx < len(matching_descriptions):
            return matching_descriptions[idx]
    
    # Check if user input matches one of the descriptions
    user_lower = user_input_clean.lower()
    for desc in matching_descriptions:
        if desc.lower() == user_lower or desc.lower() in user_lower:
            return desc
    
    return None

# ============================================
# CHARGING FLOW - WITH OPERATIONS SPECIAL HANDLING
# ============================================

def process_charging_question(user_input: str) -> str:
    """Process charging-related questions"""
    
    st.session_state.in_charging_flow = True
    
    # Merge any new info with existing context
    if st.session_state.extracted_context["team"] or st.session_state.extracted_context["keywords"]:
        if is_likely_new_query(user_input) and not is_follow_up(user_input):
            st.session_state.extracted_context = {
                "team": None,
                "keywords": None,
                "location": None,
                "exact_description": None
            }
            st.session_state.operations_search_results = {}
            st.session_state.operations_subteam = None
            if 'pending_operations_subteams' in st.session_state:
                st.session_state.pending_operations_subteams = {}
    
    # Extract information
    extracted = extract_query_info(user_input)
    
    team = extracted.get("team")
    keywords = extracted.get("keywords")
    location = extracted.get("location")
    exact_description = extracted.get("exact_description")
    
    # Step 1: Need team
    if not team:
        return "Which team are you with? (IT, Finance, HR, Legal, Corporate, Land Services, Commercial, Development, Tech Services, or Operations)"
    
    # Step 2: Need keywords (if no exact description yet)
    if not keywords and not exact_description:
        return "What would you like to charge for?"
    
    # ============================================
    # SPECIAL HANDLING FOR OPERATIONS TEAM
    # ============================================
    
    if team == "Operations":
        # Check if we're waiting for operations subteam selection
        if 'pending_operations_subteams' in st.session_state and st.session_state.pending_operations_subteams:
            pending_desc = st.session_state.get('pending_operations_description', '')
            selection_result = handle_operations_subteam_selection(
                user_input,
                st.session_state.pending_operations_subteams,
                pending_desc
            )
            
            if selection_result:
                return selection_result
            else:
                # Couldn't understand selection, ask again
                return format_operations_subteam_question(
                    pending_desc,
                    st.session_state.pending_operations_subteams
                )
        
        # Check if user specified an operations subteam
        detected_subteam = extract_operations_subteam(user_input)
        if detected_subteam:
            st.session_state.operations_subteam = detected_subteam
        
        # Check if user is selecting from previous Operations results
        if st.session_state.operations_search_results:
            selection_result = handle_operations_selection(
                user_input, 
                st.session_state.operations_search_results
            )
            
            if selection_result:
                # Clear context after providing answer
                st.session_state.extracted_context = {
                    "team": None,
                    "keywords": None,
                    "location": None,
                    "exact_description": None
                }
                st.session_state.operations_search_results = {}
                st.session_state.operations_subteam = None
                st.session_state.in_charging_flow = False
                return selection_result
        
        # If we have an exact description, search for it
        if exact_description:
            # First check if this description exists in multiple subteams
            subteam_results = get_operations_subteams_for_description(exact_description)
            
            if not subteam_results:
                st.session_state.extracted_context["exact_description"] = None
                st.session_state.operations_search_results = {}
                return f"I couldn't find charging codes for '{exact_description}' in Operations team.\n\nPlease refer to the [O&M Charging Guidelines]({CHARGING_GUIDELINES_LINK}) for more information."
            
            # If description exists in only one subteam, return results directly
            if len(subteam_results) == 1:
                subteam_name = list(subteam_results.keys())[0]
                rows = subteam_results[subteam_name]
                
                if len(rows) == 1:
                    # Single unique result
                    st.session_state.extracted_context = {
                        "team": None,
                        "keywords": None,
                        "location": None,
                        "exact_description": None
                    }
                    st.session_state.operations_subteam = None
                    st.session_state.in_charging_flow = False
                    return format_charging_info(rows[0])
                else:
                    # Multiple variants - show all
                    result = f"**Operations - {subteam_name} - {exact_description}**\n\n"
                    result += f"This charging code has **{len(rows)} variants**:\n\n"
                    
                    for idx, row in enumerate(rows, 1):
                        result += f"---\n**VARIANT {idx}:**\n"
                        result += f"- **Description:** {row['Description']}\n"
                        result += f"- **Account:** {row['Account']}\n"
                        result += f"- **Location:** {row['Location']}\n"
                        result += f"- **Company ID:** {row['Company ID']}\n"
                        result += f"- **Project:** {row['Project']}\n"
                        result += f"- **Department:** {row['Department']}\n\n"
                    
                    st.session_state.extracted_context = {
                        "team": None,
                        "keywords": None,
                        "location": None,
                        "exact_description": None
                    }
                    st.session_state.operations_subteam = None
                    st.session_state.in_charging_flow = False
                    return result.strip()
            
            # Description exists in multiple subteams - ask clarifying question
            st.session_state.pending_operations_subteams = subteam_results
            st.session_state.pending_operations_description = exact_description
            return format_operations_subteam_question(exact_description, subteam_results)
        
        # Search by keywords for Operations
        if keywords:
            # Use specific subteam if detected
            subteam_to_search = st.session_state.operations_subteam if st.session_state.operations_subteam else None
            
            grouped_results = search_operations_by_description(keywords, subteam_to_search)
            
            if not grouped_results:
                st.session_state.operations_search_results = {}
                return f"I couldn't find any Operations charging codes matching '{keywords}'.\n\nPlease refer to the [O&M Charging Guidelines]({CHARGING_GUIDELINES_LINK}) for more information."
            
            # If only one unique description found
            if len(grouped_results) == 1:
                description = list(grouped_results.keys())[0]
                
                # Check which subteams have this description
                subteam_results = get_operations_subteams_for_description(description)
                
                # If only in one subteam, return results
                if len(subteam_results) == 1:
                    subteam_name = list(subteam_results.keys())[0]
                    rows = subteam_results[subteam_name]
                    
                    if len(rows) == 1:
                        st.session_state.extracted_context = {
                            "team": None,
                            "keywords": None,
                            "location": None,
                            "exact_description": None
                        }
                        st.session_state.operations_subteam = None
                        st.session_state.in_charging_flow = False
                        return format_charging_info(rows[0])
                    else:
                        # Multiple variants
                        result = f"**Operations - {subteam_name} - {description}**\n\n"
                        result += f"This charging code has **{len(rows)} variants**:\n\n"
                        
                        for idx, row in enumerate(rows, 1):
                            result += f"---\n**VARIANT {idx}:**\n"
                            result += f"- **Description:** {row['Description']}\n"
                            result += f"- **Account:** {row['Account']}\n"
                            result += f"- **Location:** {row['Location']}\n"
                            result += f"- **Company ID:** {row['Company ID']}\n"
                            result += f"- **Project:** {row['Project']}\n"
                            result += f"- **Department:** {row['Department']}\n\n"
                        
                        st.session_state.extracted_context = {
                            "team": None,
                            "keywords": None,
                            "location": None,
                            "exact_description": None
                        }
                        st.session_state.operations_subteam = None
                        st.session_state.in_charging_flow = False
                        return result.strip()
                
                # Description in multiple subteams - ask clarifying question
                st.session_state.pending_operations_subteams = subteam_results
                st.session_state.pending_operations_description = description
                return format_operations_subteam_question(description, subteam_results)
            
            # Multiple different descriptions found - let user choose
            st.session_state.operations_search_results = grouped_results
            return format_operations_results(grouped_results)
    
    # ============================================
    # STANDARD HANDLING FOR OTHER TEAMS
    # ============================================
    
    # Check if user is selecting from a standard list
    if keywords and not exact_description:
        selected_description = check_if_selecting_from_list(user_input, extracted)
        if selected_description:
            st.session_state.extracted_context["exact_description"] = selected_description
            exact_description = selected_description
    
    # If we have exact description, get the data
    if exact_description:
        matches, has_multiple = get_charging_data(team, exact_description, location)
        
        if matches.empty:
            st.session_state.extracted_context["exact_description"] = None
            return f"I couldn't find charging codes for '{exact_description}' in {team} team.\n\nPlease refer to the [O&M Charging Guidelines]({CHARGING_GUIDELINES_LINK}) for more information."
        
        # If multiple variants exist, show all of them
        if has_multiple:
            st.session_state.extracted_context = {
                "team": None,
                "keywords": None,
                "location": None,
                "exact_description": None
            }
            st.session_state.in_charging_flow = False
            return format_multiple_variants(team, matches)
        
        # Single result - return the charging info
        row = matches.iloc[0]
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        return format_charging_info(row)
    
    # Search for descriptions matching keywords (standard teams)
    matching_descriptions = search_descriptions_by_keywords(team, keywords)
    
    if not matching_descriptions:
        return f"I couldn't find any charging codes matching '{keywords}' in {team} team.\n\nPlease refer to the [O&M Charging Guidelines]({CHARGING_GUIDELINES_LINK}) for more information."
    
    # If only one matching description, get its data
    if len(matching_descriptions) == 1:
        st.session_state.extracted_context["exact_description"] = matching_descriptions[0]
        matches, has_multiple = get_charging_data(team, matching_descriptions[0], location)
        
        # If multiple variants, show all
        if has_multiple:
            st.session_state.extracted_context = {
                "team": None,
                "keywords": None,
                "location": None,
                "exact_description": None
            }
            st.session_state.in_charging_flow = False
            return format_multiple_variants(team, matches)
        
        # Single result
        row = matches.iloc[0]
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        return format_charging_info(row)
    
    # Multiple different descriptions found
    result = f"I found {len(matching_descriptions)} charging codes matching '{keywords}' in {team} team:\n\n"
    for idx, desc in enumerate(matching_descriptions, 1):
        result += f"{idx}. {desc}\n"
    result += "\nWhich one are you looking for?"
    
    return result

# ============================================
# MAIN PROCESSING LOGIC
# ============================================

def process_message(user_input: str) -> str:
    """Main message processing - routes to charging, reference, or general conversation"""
    
    # Handle greetings
    greetings = ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]
    if user_input.lower().strip() in greetings or any(user_input.lower().strip().startswith(g) for g in greetings):
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        st.session_state.operations_search_results = {}
        return "Hi! I'm Diva, your charging guidelines assistant. How can I help you today?"
    
    # If we're already in a charging flow, continue
    if st.session_state.in_charging_flow:
        if is_likely_new_query(user_input) and not is_charging_question(user_input):
            st.session_state.in_charging_flow = False
            st.session_state.operations_search_results = {}
            # Check if it's a reference question
            is_ref, ref_type = is_reference_question(user_input)
            if is_ref:
                return process_reference_question(user_input, ref_type)
            return generate_natural_response(user_input)
        else:
            return process_charging_question(user_input)
    
    # Check for reference questions (check this before charging)
    is_ref, ref_type = is_reference_question(user_input)
    if is_ref:
        return process_reference_question(user_input, ref_type)
    
    # Detect if this is a charging question
    if is_charging_question(user_input):
        return process_charging_question(user_input)
    else:
        # Handle as general conversation
        st.session_state.in_charging_flow = False
        st.session_state.operations_search_results = {}
        return generate_natural_response(user_input)

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

user_input = st.chat_input("Ask about charging codes or chat with me...")

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
