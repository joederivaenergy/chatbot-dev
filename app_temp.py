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
    'Finance': 'csvs/Guidelines_cleaned_finance.csv',
    'HR': 'csvs/Guidelines_cleaned_hr.csv',
}

# ============================================
# LOAD CSV DATA (FIXED - PROPER NAN HANDLING)
# ============================================

@st.cache_data
def load_all_csvs():
    """Load all CSV files into memory"""
    data = {}
    for team, filepath in CSV_FILES.items():
        if os.path.exists(filepath):
            try:
                # Read CSV with Account as string to preserve formatting
                df = pd.read_csv(filepath, dtype={'Account': str}, keep_default_na=False, na_values=[''])
                
                # Strip whitespace from all string columns
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                
                # Replace empty strings and 'nan' strings with actual NaN
                df = df.replace(['', 'nan', 'NaN', 'NAN', 'None'], pd.NA)
                
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
        "exact_description": None
    }

if "in_charging_flow" not in st.session_state:
    st.session_state.in_charging_flow = False

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("⚙️ Settings")

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
1. **team**: ONLY if explicitly mentioned (IT, Finance, HR, Operations, Engineering)
2. **keywords**: Key words the user wants to search for (e.g., "erp", "labor", "maintenance")
3. **location**: Specific location if mentioned (e.g., "Houston", "DSOL")
4. **is_new_query**: Is this a NEW charging question or a follow-up/clarification? (true/false)

RULES FOR is_new_query:
- TRUE if: User asks "how to charge X", "where to charge Y", "codes for Z", or any new charging question
- FALSE if: User gives short answers like "IT", "Houston", "1", "option 2" (these are clarifications)
- TRUE if: User asks about a DIFFERENT project/activity than previous conversation
- FALSE if: User is answering assistant's clarification questions

Return ONLY valid JSON:
{
  "team": "IT" | "Finance" | "HR" | "Operations" | "Engineering" | null,
  "keywords": "search terms" | null,
  "location": "location name" | null,
  "is_new_query": true | false
}

Examples:
- "how to charge erp" → {"team": null, "keywords": "erp", "location": null, "is_new_query": true}
- Previous asked team, Current: "IT" → {"team": "IT", "keywords": null, "location": null, "is_new_query": false}
- Previous: "Core ERP codes", Current: "what about HR labor?" → {"team": null, "keywords": "labor", "location": null, "is_new_query": true}
- Previous: showed 3 locations, Current: "Houston" → {"team": null, "keywords": null, "location": "Houston", "is_new_query": false}
"""

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
# FORMATTING HELPER (ADDED)
# ============================================

def format_field(value) -> str:
    """Format any field value, replacing NaN/None/empty with N/A"""
    if pd.isna(value) or value is None:
        return 'N/A'
    
    value_str = str(value).strip()
    
    # Check for string representations of NaN
    if value_str.lower() in ['nan', 'none', '']:
        return 'N/A'
    
    return value_str

# ============================================
# FORMATTING FUNCTIONS (FIXED)
# ============================================

def format_charging_info(row: pd.Series) -> str:
    """Format a single charging code with markdown bullets"""
    result = f"""- **Description:** {format_field(row.get('Description'))}
- **Account number:** {format_field(row.get('Account'))}
- **Location:** {format_field(row.get('Location'))}
- **Company ID:** {format_field(row.get('Company ID'))}
- **Project:** {format_field(row.get('Project'))}
- **Department:** {format_field(row.get('Department'))}"""
    return result

def format_multiple_variants(team: str, matches: pd.DataFrame) -> str:
    """Format multiple variants of the same description"""
    description = format_field(matches.iloc[0]['Description'])
    
    result = f"**{team} Team - {description}**\n\n"
    result += f"This charging code has **{len(matches)} options**. Please use the one that applies to your situation:\n\n"
    
    for idx, (_, row) in enumerate(matches.iterrows(), 1):
        result += f"---\n**OPTION {idx}:**\n"
        result += f"- **Description:** {format_field(row.get('Description'))}\n"
        result += f"- **Account number:** {format_field(row.get('Account'))}\n"
        result += f"- **Location:** {format_field(row.get('Location'))}\n"
        result += f"- **Company ID:** {format_field(row.get('Company ID'))}\n"
        result += f"- **Project:** {format_field(row.get('Project'))}\n"
        result += f"- **Department:** {format_field(row.get('Department'))}\n\n"
    
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
# CHARGING FLOW PROCESSING
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
    
    # Step 1: Need team
    if not team:
        return "Which team are you with? (IT, Finance, HR, Operations, or Engineering)"
    
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
        return f"I couldn't find any charging codes matching '{keywords}' in {team} team."
    
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
    """Main message processing - routes to charging or general conversation"""
    
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
