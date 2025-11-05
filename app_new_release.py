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
    'IT': 'csvs/Guidelines_cleaned_it.csv',
    'Finance': 'csvs/Guidelines_cleaned_Finance.csv',
    'HR': 'csvs/Guidelines_cleaned_HR.csv',
    'Legal': 'csvs/Guidelines_cleaned_Legal.csv',
    'Corporate': 'csvs/Guidelines_cleaned_Corporate.csv',
    'Land Services': 'csvs/Guidelines_cleaned_Land_Services.csv',
    'Commercial': 'csvs/Guidelines_cleaned_Commercial.csv',
    'Development': 'csvs/Guidelines_cleaned_Development.csv',
    'Tech Services': 'csvs/Guidelines_cleaned_Tech_Services.csv',
}

# --- Reference CSV Files ---
REFERENCE_CSV_FILES = {
    'department_list': 'csvs/Guidelines_cleaned_dept_list.csv'
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
                df = pd.read_csv(filepath, encoding='utf-8-sig')  # Handle BOM
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

# ============================================
# SIDEBAR
# ============================================

# Add logo at the very top of sidebar
if os.path.exists("Deriva-Logo.png"):
    st.sidebar.image("Deriva-Logo.png", width=200)
else:
    st.sidebar.warning("⚠️ logo.png not found")

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

with st.sidebar.expander("⚙️ **Tools**", expanded=True):
    if st.button("��️ Clear Chat"):
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

with st.sidebar.expander("�� Support"):
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
You are Diva, a charging guidelines and department information assistant for Deriva Energy.

Determine if the user's question is about CHARGING GUIDELINES or something else.

Charging questions are about:
- How to charge time/expenses
- Charging codes, account numbers, projects, departments
- Where to charge specific work (ERP, labor, projects, etc.)
- Team-specific charging information

Department questions are about:
- Department names, numbers, or codes
- Department information and details
- How many departments exist
- Listing or searching for departments

NOT charging/department questions:
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
        "department", "expense", "time", "labor", "timesheet", "bill"
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

Your primary purpose is to help with charging guidelines, but you can also answer general questions conversationally and help coding.

Guidelines:
- Be friendly, concise, and professional
- For questions about Deriva Energy that you don't know, say you're primarily designed for charging guidelines
- Keep responses brief (2-4 sentences unless more detail is needed)
- If the question seems related to charging, time or timesheet, gently guide them: "If you're asking about charging guidelines, I can help with that!"

You are an internal tool for Deriva Energy employees.
"""

def generate_natural_response(user_query: str) -> str:
    """Generate natural language response for non-charging questions"""
    
    response = call_claude(GENERAL_ASSISTANT_PROMPT, user_query, include_history=True)
    
    if not response:
        return "I'm here to help! My specialty is providing info regarding charging guidelines. Could you rephrase your question or ask about charging codes?"
    
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
4. **is_new_query**: Is this a NEW charging question or a follow-up/clarification? (true/false)

RULES FOR is_new_query:
- TRUE if: User asks "how to charge X", "where to charge Y", "codes for Z", or any new charging question
- FALSE if: User gives short answers like "IT", "Houston", "1", "option 2" (these are clarifications)
- TRUE if: User asks about a DIFFERENT project/activity than previous conversation
- FALSE if: User is answering assistant's clarification questions

Return ONLY valid JSON:
{
  "team": "IT" | "Finance" | "HR" | "Legal" | "Corporate" | "Land Services" | "Commercial" | "Development" | "Tech Services" | "Operations" | null,
  "keywords": "search terms" | null,
  "location": "location name" | null,
  "is_new_query": true | false
}

Examples:
- "how to charge erp" → {"team": null, "keywords": "erp", "location": null, "is_new_query": true}
- Previous asked team, Current: "IT" → {"team": "IT", "keywords": null, "location": null, "is_new_query": false}
- Previous: "Core ERP codes", Current: "what about HR labor?" → {"team": null, "keywords": "labor", "location": null, "is_new_query": true}
- Previous: showed 3 locations, Current: "DSOL" → {"team": null, "keywords": null, "location": "DSOL", "is_new_query": false}
"""

# ============================================
# EXTRACTION - RESET TEAM FOR NEW QUERIES (UPDATED)
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
        # User must specify team again for each new charging question
        if is_new_query:
            extracted = {
                "team": data.get("team"),  # Only use if explicitly mentioned in new query
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
    result += f"This charging code has **{len(matches)} options**. Please refer to the deparment list for more details:\n\n"
    
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
# CHARGING FLOW - RESET TEAM FOR NEW QUESTIONS (UPDATED)
# ============================================

def process_charging_question(user_input: str) -> str:
    """Process charging guidelines question"""
    
    # Mark that we're in charging flow
    st.session_state.in_charging_flow = True
    
    # Check if this looks like a new query (fallback detection)
    if is_likely_new_query(user_input):
        # For NEW charging questions, completely reset context
        # User must specify team again
        st.session_state.extracted_context = {
            "team": None,  # Reset team for new questions
            "keywords": None,
            "location": None,
            "exact_description": None
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
    
    # Step 1: Need team (ALWAYS ask for new charging questions)
    if not team:
        return "Which team are you with? (IT, Finance, HR, Legal, Corporate, Land Services, Commerical, Development, Tech Services, or Operations)"
    
    # Check if team is Operations - redirect to guidelines
    if team and team.lower() == "operations":
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        return "For Operations team charging codes, please refer to the **Operations tabs** in the [O&M Charging Guidelines](https://derivaenergy.sharepoint.com/:x:/r/sites/DerivaFinance/_layouts/15/Doc.aspx?sourcedoc=%7B3CD9F65D-C693-4CE8-904C-91074451F098%7D&file=Deriva%20OM%20Charging%20Guidelines.xlsx&action=default&mobileredirect=true)."
    
    # Step 2: Need keywords (if no exact description yet)
    if not keywords and not exact_description:
        return "What would you like to charge for?"
    
    # Step 3: If we have exact description, get the data
    if exact_description:
        matches, has_multiple = get_charging_data(team, exact_description, location)
        
        if matches.empty:
            st.session_state.extracted_context["exact_description"] = None
            return f"I couldn't find charging codes for '{exact_description}' in {team} team."
        
        # If multiple variants exist, show all of them
        if has_multiple:
                    # After providing answer, completely clear context
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
        
        # After providing answer, completely clear context
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        
        return format_charging_info(row)
    
    # Step 4: Search for descriptions matching keywords
    matching_descriptions = search_descriptions_by_keywords(team, keywords)
    
    if not matching_descriptions:
        return f"I couldn't find any charging information regarding '{keywords}' in {team} team. Could you check the description and try again?"
    
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
# DEPARTMENT QUERY FUNCTIONS
# ============================================

def is_department_question(user_query: str) -> bool:
    """Detect if the user is asking about departments"""
    query_lower = user_query.lower()
    
    department_keywords = [
        'department', 'dept', 'departments', 'depts',
        'department number', 'dept number', 'department code',
        'how many department', 'list department', 'show department',
        'what department', 'which department', 'all department'
    ]
    
    return any(keyword in query_lower for keyword in department_keywords)

def search_departments(query: str) -> pd.DataFrame:
    """Search for departments based on user query"""
    if 'department_list' not in ALL_TEAM_DATA:
        return pd.DataFrame()
    
    df = ALL_TEAM_DATA['department_list']
    query_lower = query.lower()
    
    # If asking for all departments or count
    if any(word in query_lower for word in ['all', 'list', 'show all', 'how many']):
        return df
    
    # Search by department number
    dept_numbers = re.findall(r'\b\d{4}\b', query)
    if dept_numbers:
        return df[df['Department Number'].astype(str).isin(dept_numbers)]
    
    # Search by name, HR function group, or HR group
    search_cols = ['Department Name', 'HR Function Group', 'HR Group']
    mask = pd.Series([False] * len(df))
    
    for col in search_cols:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(query_lower, na=False, regex=False)
    
    return df[mask]

def format_department_info(departments_df: pd.DataFrame, query: str) -> str:
    """Format department information for display"""
    if departments_df.empty:
        return "I couldn't find any departments matching your query. Please try again with different keywords."
    
    query_lower = query.lower()
    
    # If asking for count or all departments
    if any(word in query_lower for word in ['how many', 'count', 'total']):
        total = len(departments_df)
        response = f"**Total Departments: {total}**\n\n"
        
        # Group by HR Function Group
        if 'HR Function Group' in departments_df.columns:
            grouped = departments_df.groupby('HR Function Group').size().reset_index(name='Count')
            response += "**Breakdown by HR Function Group:**\n"
            for _, row in grouped.iterrows():
                response += f"- {row['HR Function Group']}: {row['Count']} department(s)\n"
        
        return response
    
    # If asking for list of all departments
    if any(word in query_lower for word in ['list', 'all', 'show all']) and len(departments_df) > 5:
        response = f"**Found {len(departments_df)} departments:**\n\n"
        
        # Group by HR Function Group for better organization
        if 'HR Function Group' in departments_df.columns:
            for group in departments_df['HR Function Group'].unique():
                group_depts = departments_df[departments_df['HR Function Group'] == group]
                response += f"\n**{group}:**\n"
                for _, row in group_depts.iterrows():
                    response += f"- {row['Department Number']}: {row['Department Name']}\n"
        else:
            for _, row in departments_df.iterrows():
                response += f"- {row['Department Number']}: {row['Department Name']}\n"
        
        return response
    
    # Detailed view for specific departments (≤5 results)
    if len(departments_df) <= 5:
        response = f"**Found {len(departments_df)} department(s):**\n\n"
        
        for idx, row in departments_df.iterrows():
            response += f"**{row['Department Number']} - {row['Department Name']}**\n"
            if pd.notna(row.get('Resp Center')):
                response += f"- Responsibility Center: {row['Resp Center']}\n"
            if pd.notna(row.get('HR Function Group')):
                response += f"- HR Function Group: {row['HR Function Group']}\n"
            if pd.notna(row.get('HR Group')):
                response += f"- HR Group: {row['HR Group']}\n"
            if pd.notna(row.get('SG&A/OPS/DEVEX')):
                response += f"- Category: {row['SG&A/OPS/DEVEX']}\n"
            response += "\n"
        
        return response
    
    # More than 5 but not asking for all - show summary
    response = f"**Found {len(departments_df)} departments. Here are the first 10:**\n\n"
    for _, row in departments_df.head(10).iterrows():
        response += f"- {row['Department Number']}: {row['Department Name']}\n"
    
    if len(departments_df) > 10:
        response += f"\n*...and {len(departments_df) - 10} more. Please refine your search for specific departments.*"
    
    return response

def process_department_question(user_input: str) -> str:
    """Process department-related questions"""
    departments = search_departments(user_input)
    return format_department_info(departments, user_input)

# ============================================
# MAIN PROCESSING LOGIC
# ============================================

def process_message(user_input: str) -> str:
    """Main message processing - routes to charging, department, or general conversation"""
    
    # Handle greetings
    greetings = ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]
    if user_input.lower().strip() in greetings or any(user_input.lower().strip().startswith(g) for g in greetings):
        # Clear context on greeting
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        return "Hi there! I'm Diva, your AI assistant. How can I help you today?"
    
    # Check for department questions (before charging flow check)
    if is_department_question(user_input):
        st.session_state.in_charging_flow = False
        return process_department_question(user_input)
    
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
