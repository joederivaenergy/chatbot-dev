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

# --- CSV File Paths ---
CSV_FILES = {
    'IT': 'csvs/Guidelines_cleaned_IT.csv',
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
        "exact_description": None,
        "operation_type": None
    }

if "in_charging_flow" not in st.session_state:
    st.session_state.in_charging_flow = False

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
        "department", "expense", "time", "labor", "erp"
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
    """Extract team, keywords, location, and operation_type from user query"""
    
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
                "exact_description": None,
                "operation_type": data.get("operation_type")
            }
            
            st.session_state.extracted_context = extracted
            return extracted
        
        # If it's a follow-up, merge with existing context
        extracted = {
            "team": data.get("team"),
            "keywords": data.get("keywords"),
            "location": data.get("location"),
            "exact_description": st.session_state.extracted_context.get("exact_description"),
            "operation_type": data.get("operation_type")
        }
        
        # Merge with existing context
        merged = {}
        for key in ["team", "keywords", "location", "exact_description", "operation_type"]:
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
# HELPER: GET CORRECT TEAM CSV KEY
# ============================================

def get_team_csv_key(team: str, operation_type: str = None) -> str:
    """Get the correct CSV key for the team, handling Operations sub-types"""
    if team == "Operations" and operation_type:
        # Map operation_type to CSV key
        operation_map = {
            "high_voltage": "Operations_High_Voltage",
            "mst_fs": "Operations_MST_FS",
            "mst_gbx": "Operations_MST_GBX",
            "mst_lc": "Operations_MST_LC",
            "non_controllable_qm": "Operations_Non_Controllable_QM",
            "ops_support": "Operations_Ops_Support",
            "solar_sites": "Operations_Solar_Sites",
            "wind_sites": "Operations_Wind_Sites",
            "blade_maintenance": "Operations_Blade_Maintenance_Repair"
        }
        return operation_map.get(operation_type.lower(), team)
    return team

# ============================================
# CSV SEARCH FUNCTIONS (WHOLE WORD MATCHING)
# ============================================

def search_descriptions_by_keywords(team: str, keywords: str, operation_type: str = None) -> List[str]:
    """Search for descriptions containing whole words from keywords"""
    team_key = get_team_csv_key(team, operation_type)
    
    if team_key not in ALL_TEAM_DATA or ALL_TEAM_DATA[team_key].empty:
        return []
    
    df = ALL_TEAM_DATA[team_key]
    
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

def get_charging_data(team: str, exact_description: str, location: str = None, operation_type: str = None) -> Tuple[pd.DataFrame, bool]:
    """Get charging data for exact description"""
    team_key = get_team_csv_key(team, operation_type)
    
    if team_key not in ALL_TEAM_DATA or ALL_TEAM_DATA[team_key].empty:
        return pd.DataFrame(), False
    
    df = ALL_TEAM_DATA[team_key]
    
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
# FORMATTING FUNCTIONS
# ============================================

def format_charging_info(row: pd.Series) -> str:
    """Format a single charging code with markdown bullets"""
    result = f"""- **Description:** {row['Description']}
- **Account:** {row['Account']}
- **Location:** {row['Location']}
- **Company ID:** {row['Company ID']}
- **Project:** {row['Project']}
- **Department:** {row['Department']}"""
    return result

def format_multiple_variants(team: str, matches: pd.DataFrame) -> str:
    """Format multiple variants of the same description"""
    description = matches.iloc[0]['Description']
    
    result = f"**{team} Team - {description}**\n\n"
    result += f"This charging code has **{len(matches)} options**. Please refer to the department list for more details:\n\n"
    
    for idx, (_, row) in enumerate(matches.iterrows(), 1):
        result += f"---\n**OPTION {idx}:**\n"
        result += f"- **Description:** {row['Description']}\n"
        result += f"- **Account:** {row['Account']}\n"
        result += f"- **Location:** {row['Location']}\n"
        result += f"- **Company ID:** {row['Company ID']}\n"
        result += f"- **Project:** {row['Project']}\n"
        result += f"- **Department:** {row['Department']}\n\n"
    
    return result.strip()

# ============================================
# HANDLE USER SELECTING FROM LIST
# ============================================

def check_if_selecting_from_list(user_input: str, extracted: Dict) -> str:
    """Check if user is selecting a description from a list"""
    team = extracted.get("team")
    keywords = extracted.get("keywords")
    operation_type = extracted.get("operation_type")
    
    if not team or not keywords:
        return None
    
    # Get matching descriptions
    matching_descriptions = search_descriptions_by_keywords(team, keywords, operation_type)
    
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
# CHARGING FLOW - WITH OPERATIONS SUPPORT
# ============================================

def process_charging_question(user_input: str) -> str:
    """Process charging guidelines question"""
    
    # Mark that we're in charging flow
    st.session_state.in_charging_flow = True
    
    # Check if this looks like a new query (fallback detection)
    if is_likely_new_query(user_input):
        # For NEW charging questions, completely reset context
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None,
            "operation_type": None
        }
    
    # Extract information
    extracted = extract_query_info(user_input)
    
    # Check if user is selecting from a list
    if extracted.get("keywords") and not extracted.get("exact_description"):
        selected_description = check_if_selecting_from_list(user_input, extracted)
        if selected_description:
            st.session_state.extracted_context["exact_description"] = selected_description
            extracted["exact_description"] = selected_description
    
    # Continue with normal processing
    team = extracted.get("team")
    keywords = extracted.get("keywords")
    location = extracted.get("location")
    exact_description = extracted.get("exact_description")
    operation_type = extracted.get("operation_type")
    
    # Step 1: Need team
    if not team:
        return "Which team are you with? (IT, Finance, HR, Legal, Corporate, Land Services, Commercial, Development, Tech Services, or Operations)"
    
    # Step 1.5: If Operations team, need operation type
    if team == "Operations" and not operation_type:
        return """Which Operations area are you working in?

1. **High Voltage** (HV)
2. **MST FS** (Field Service)
3. **MST GBX** (Gearbox)
4. **MST LC** (Load Center)
5. **Non-Controllable Q&M** (Quality & Maintenance)
6. **Ops Support** (Operations Support)
7. **Solar Sites**
8. **Wind Sites**
9. **Blade Maintenance & Repair**

Please select one or type the name."""
    
    # Step 2: Need keywords (if no exact description yet)
    if not keywords and not exact_description:
        return "What would you like to charge for?"
    
    # Step 3: If we have exact description, get the data
    if exact_description:
        matches, has_multiple = get_charging_data(team, exact_description, location, operation_type)
        
        if matches.empty:
            st.session_state.extracted_context["exact_description"] = None
            team_display = f"{team} - {operation_type.replace('_', ' ').title()}" if operation_type else team
            return f"I couldn't find charging codes for '{exact_description}' in {team_display} team."
        
        # If multiple variants exist, show all of them
        if has_multiple:
            st.session_state.extracted_context = {
                "team": None,
                "keywords": None,
                "location": None,
                "exact_description": None,
                "operation_type": None
            }
            st.session_state.in_charging_flow = False
            
            team_display = f"{team} - {operation_type.replace('_', ' ').title()}" if operation_type else team
            return format_multiple_variants(team_display, matches)
        
        # Single result - return the charging info
        row = matches.iloc[0]
        
        # After providing answer, completely clear context
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None,
            "operation_type": None
        }
        st.session_state.in_charging_flow = False
        
        return format_charging_info(row)
    
    # Step 4: Search for descriptions matching keywords
    matching_descriptions = search_descriptions_by_keywords(team, keywords, operation_type)
    
    if not matching_descriptions:
        team_display = f"{team} - {operation_type.replace('_', ' ').title()}" if operation_type else team
        return f"I couldn't find any charging codes matching '{keywords}' in {team_display} team."
    
    # If only one matching description, get its data
    if len(matching_descriptions) == 1:
        st.session_state.extracted_context["exact_description"] = matching_descriptions[0]
        matches, has_multiple = get_charging_data(team, matching_descriptions[0], location, operation_type)
        
        # If multiple variants, show all
        if has_multiple:
            st.session_state.extracted_context = {
                "team": None,
                "keywords": None,
                "location": None,
                "exact_description": None,
                "operation_type": None
            }
            st.session_state.in_charging_flow = False
            
            team_display = f"{team} - {operation_type.replace('_', ' ').title()}" if operation_type else team
            return format_multiple_variants(team_display, matches)
        
        # Single result
        row = matches.iloc[0]
        
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None,
            "operation_type": None
        }
        st.session_state.in_charging_flow = False
        
        return format_charging_info(row)
    
    # Multiple different descriptions found
    team_display = f"{team} - {operation_type.replace('_', ' ').title()}" if operation_type else team
    result = f"I found {len(matching_descriptions)} charging codes matching '{keywords}' in {team_display} team:\n\n"
    for idx, desc in enumerate(matching_descriptions, 1):
        result += f"{idx}. {desc}\n"
    result += "\nWhich one are you looking for?"
    
    return result

# ============================================
# MAIN PROCESSING LOGIC
# ============================================

def process_message(user_input: str) -> str:
    """Main message processing - routes to charging or general conversation"""
    
    # Handle greetings
    greetings = ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]
    if user_input.lower().strip() in greetings or any(user_input.lower().strip().startswith(g) for g in greetings):
        # Clear context on greeting
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None,
            "operation_type": None
        }
        st.session_state.in_charging_flow = False
        return "Hi! I'm Diva, your charging guidelines assistant. How can I help you today?"
    
    # If we're already in a charging flow (answering clarifications), continue
    if st.session_state.in_charging_flow:
        # Check if it's a new question or continuing the flow
        if is_likely_new_query(user_input) and not is_charging_question(user_input):
            # Exit charging flow and handle as general question
            st.session_state.in_charging_flow = False
            return generate_natural_response(user_input)
        else:
            # Continue charging flow
            return process_charging_question(user_input)
    
    # Detect if this is a charging question
    if is_charging_question(user_input):
        return process_charging_question(user_input)
    else:
        # Handle as general conversation
        st.session_state.in_charging_flow = False
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
