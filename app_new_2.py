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
# HELPER FUNCTION: NORMALIZE TEXT FOR MATCHING
# ============================================

def normalize_for_matching(text: str) -> str:
    """
    Normalize text to handle plural/singular variations and common differences.
    This helps match 'contractor' with 'contractors', 'employee' with 'employees', etc.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    
    # Remove common punctuation
    text = re.sub(r'[,.\-_/]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def create_search_variants(text: str) -> List[str]:
    """
    Create variations of the search text to match plural/singular forms.
    Returns a list of possible variations to search for.
    """
    if not isinstance(text, str):
        return [""]
    
    variants = [text]
    text_lower = text.lower().strip()
    
    # Add the normalized version
    normalized = normalize_for_matching(text)
    if normalized not in variants:
        variants.append(normalized)
    
    # Handle plural/singular variations
    words = text_lower.split()
    variant_words = []
    
    for word in words:
        word_variants = [word]
        
        # If ends with 's', add singular form
        if word.endswith('s') and len(word) > 1:
            singular = word[:-1]
            word_variants.append(singular)
            
            # Handle words ending in 'ies' -> 'y' (e.g., companies -> company)
            if word.endswith('ies') and len(word) > 3:
                word_variants.append(word[:-3] + 'y')
            
            # Handle words ending in 'es' -> remove 'es' (e.g., boxes -> box)
            if word.endswith('es') and len(word) > 2:
                word_variants.append(word[:-2])
        
        # If doesn't end with 's', add plural forms
        else:
            word_variants.append(word + 's')
            
            # Handle words ending in 'y' -> 'ies' (e.g., company -> companies)
            if word.endswith('y') and len(word) > 1:
                word_variants.append(word[:-1] + 'ies')
            
            # Handle words that need 'es' (e.g., box -> boxes)
            if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
                word_variants.append(word + 'es')
        
        variant_words.append(word_variants)
    
    # If single word, just return its variants
    if len(variant_words) == 1:
        return list(set(variant_words[0]))
    
    # For multi-word phrases, we'll use the original and normalized versions
    return list(set(variants))

def flexible_search(df: pd.DataFrame, search_column: str, search_term: str) -> pd.DataFrame:
    """
    Perform flexible search that handles plural/singular variations.
    Returns matching rows from the dataframe.
    """
    if df.empty or search_column not in df.columns:
        return pd.DataFrame()
    
    # Create search variants
    search_variants = create_search_variants(search_term)
    
    # Create a mask for matching rows
    mask = pd.Series([False] * len(df))
    
    # Normalize the column for searching
    normalized_column = df[search_column].apply(normalize_for_matching)
    
    # Search for each variant
    for variant in search_variants:
        variant_normalized = normalize_for_matching(variant)
        if variant_normalized:
            # Check if the normalized variant appears in the normalized column
            mask |= normalized_column.str.contains(variant_normalized, case=False, na=False, regex=False)
    
    return df[mask]

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
    page_icon="��", 
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
# CHAT HISTORY INSTANCE
# ============================================

chat_history = DynamoDBChatHistory(
    table_name=DDB_TABLE_NAME,
    session_id=st.session_state["session_id"]
)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("�� Diva The Chatbot")
    st.markdown("Your AI assistant for charging codes and company information.")
    
    st.divider()
    
    if st.button("��️ Clear Chat History"):
        chat_history.clear()
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        st.rerun()
    
    st.divider()
    st.markdown("### Quick Tips")
    st.markdown("- Ask about charging codes")
    st.markdown("- Search for departments")
    st.markdown("- Get company information")

# ============================================
# HELPER FUNCTIONS
# ============================================

def is_charging_question(user_query: str) -> bool:
    """Detect if the question is about charging codes"""
    query_lower = user_query.lower()
    
    charging_keywords = [
        'charge', 'charging', 'account', 'code', 'cost',
        'expense', 'budget', 'billing', 'invoice',
        'gl', 'general ledger', 'account number'
    ]
    
    return any(keyword in query_lower for keyword in charging_keywords)

def is_likely_new_query(user_query: str) -> bool:
    """Check if this seems like a new question vs continuing conversation"""
    query_lower = user_query.lower().strip()
    
    # Question words indicate new query
    question_starters = ['what', 'where', 'when', 'who', 'why', 'how', 'which', 'can', 'could', 'would', 'should']
    if any(query_lower.startswith(q) for q in question_starters):
        return True
    
    # Continuation words indicate not new
    continuation_words = ['yes', 'no', 'yeah', 'nope', 'correct', 'wrong', 'right', 'that', 'this', 'it']
    if any(query_lower.startswith(c) for c in continuation_words):
        return False
    
    # If it's longer and doesn't reference previous context, likely new
    if len(query_lower.split()) > 5:
        return True
    
    return False

def call_claude_api(prompt: str, max_tokens: int = 2000) -> str:
    """Call Claude API via AWS Bedrock"""
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    except Exception as e:
        st.error(f"Error calling Claude API: {e}")
        return "I'm having trouble connecting right now. Please try again."

# ============================================
# CONTEXT EXTRACTION
# ============================================

def extract_initial_context(user_query: str) -> Dict:
    """
    Extract team, keywords, location, and description from user query.
    CRITICAL: This function should ONLY extract context, NOT generate account numbers.
    """
    
    prompt = f"""You are helping to extract context from a user's question about charging codes.

User question: "{user_query}"

Available teams: {', '.join(CSV_FILES.keys())}

Extract the following information ONLY if explicitly mentioned or clearly implied:
1. Team/Department (must be one of the available teams listed above)
2. Keywords for searching (specific terms like 'travel', 'office supplies', etc.)
3. Location (if mentioned, like 'Texas', 'Oklahoma', etc.)
4. Exact description phrase (the main thing they're asking about - could be plural or singular)

IMPORTANT: 
- Only extract information that is clearly present in the query
- Do NOT make up or infer information that isn't there
- Do NOT generate account numbers or codes
- For description, extract the exact phrase they used

Return ONLY a JSON object with these keys: team, keywords, location, exact_description
If something is not mentioned, use null for that field.

Example response format:
{{"team": "IT", "keywords": "laptop", "location": null, "exact_description": "laptop purchases"}}"""

    response = call_claude_api(prompt, max_tokens=500)
    
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            context = json.loads(json_match.group())
            return context
        else:
            return {"team": None, "keywords": None, "location": None, "exact_description": None}
    except json.JSONDecodeError:
        return {"team": None, "keywords": None, "location": None, "exact_description": None}

# ============================================
# CHARGING QUERY PROCESSING - WITH STRICT VALIDATION
# ============================================

def search_charging_codes_strict(team: str, description: str, location: str = None) -> pd.DataFrame:
    """
    Search for charging codes with STRICT validation against CSV data.
    Uses flexible matching to handle plural/singular variations.
    NEVER returns data that doesn't exist in the CSV.
    """
    if team not in ALL_TEAM_DATA or ALL_TEAM_DATA[team].empty:
        return pd.DataFrame()
    
    df = ALL_TEAM_DATA[team]
    
    # Use flexible search for description matching
    if description:
        matched_df = flexible_search(df, 'description', description)
    else:
        matched_df = df
    
    # Filter by location if specified
    if location and 'location' in matched_df.columns:
        # Use flexible search for location too
        matched_df = flexible_search(matched_df, 'location', location)
    
    return matched_df

def validate_account_numbers(account_numbers: List[str], team: str) -> Tuple[List[str], List[str]]:
    """
    Validate that account numbers actually exist in the CSV.
    Returns (valid_numbers, invalid_numbers)
    """
    if team not in ALL_TEAM_DATA or ALL_TEAM_DATA[team].empty:
        return [], account_numbers
    
    df = ALL_TEAM_DATA[team]
    
    # Check if 'account' column exists
    if 'account' not in df.columns:
        return [], account_numbers
    
    valid_accounts = df['account'].astype(str).tolist()
    
    valid = []
    invalid = []
    
    for acc in account_numbers:
        if str(acc) in valid_accounts:
            valid.append(acc)
        else:
            invalid.append(acc)
    
    return valid, invalid

def format_charging_results(results_df: pd.DataFrame, query: str) -> str:
    """
    Format the charging code results for display.
    ONLY shows data that actually exists in the CSV.
    """
    if results_df.empty:
        return None  # Will trigger a "not found" message
    
    # Limit to top 10 results
    display_df = results_df.head(10)
    
    response = f"**Found {len(results_df)} charging code(s)**"
    if len(results_df) > 10:
        response += f" (showing first 10)"
    response += ":\n\n"
    
    for idx, row in display_df.iterrows():
        response += f"**Account: {row['account']}**\n"
        response += f"- Description: {row['description']}\n"
        
        if 'location' in row and pd.notna(row['location']):
            response += f"- Location: {row['location']}\n"
        
        if 'notes' in row and pd.notna(row['notes']):
            response += f"- Notes: {row['notes']}\n"
        
        response += "\n"
    
    if len(results_df) > 10:
        response += f"*...and {len(results_df) - 10} more results. Please refine your search if needed.*"
    
    return response

def process_charging_question(user_input: str) -> str:
    """Process charging-related questions with STRICT validation"""
    
    st.session_state.in_charging_flow = True
    
    # Check if we have existing context
    context = st.session_state.extracted_context
    
    # If no context yet, extract it
    if not context.get("team") and not context.get("exact_description"):
        context = extract_initial_context(user_input)
        st.session_state.extracted_context = context
    
    # If we still don't have enough info, ask for it
    if not context.get("team"):
        available_teams = ', '.join(CSV_FILES.keys())
        return f"To help you find the right charging code, which team or department is this for?\n\nAvailable teams: {available_teams}"
    
    if not context.get("exact_description"):
        return "What type of expense or activity are you looking to charge? (e.g., 'travel expenses', 'office supplies', 'contractor payments')"
    
    # Now search with the context we have
    team = context["team"]
    description = context["exact_description"]
    location = context.get("location")
    
    # STRICT SEARCH - only return what exists in CSV
    results = search_charging_codes_strict(team, description, location)
    
    if results.empty:
        # Try broader search without location
        if location:
            results = search_charging_codes_strict(team, description, None)
            
            if not results.empty:
                formatted = format_charging_results(results, user_input)
                return formatted + "\n\n*Note: I found results for other locations. Let me know if you need codes specific to a location.*"
        
        # Still no results - provide helpful message
        response = f"I couldn't find any charging codes matching '{description}' in the {team} team."
        response += "\n\nTry:"
        response += "\n- Using different keywords (e.g., 'contractor' instead of 'consultants')"
        response += "\n- Checking if you have the right team selected"
        response += "\n- Asking me to list all codes for this team"
        
        # Reset context for next search
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False
        
        return response
    
    # Format and return results
    formatted_response = format_charging_results(results, user_input)
    
    # Reset context after successful search
    st.session_state.extracted_context = {
        "team": None,
        "keywords": None,
        "location": None,
        "exact_description": None
    }
    st.session_state.in_charging_flow = False
    
    return formatted_response

# ============================================
# GENERAL CONVERSATION HANDLING
# ============================================

def generate_natural_response(user_input: str) -> str:
    """Generate conversational response using Claude"""
    
    # Get chat history for context
    history = chat_history.get_formatted_history()
    
    prompt = f"""You are Diva, a helpful AI assistant for Deriva Energy employees.

{history}

Current user message: "{user_input}"

Guidelines:
- Be conversational and helpful
- If asked about charging codes, guide them to ask specific questions
- If asked about departments, help them search
- Keep responses concise and friendly
- Do NOT make up account numbers or codes
- Do NOT provide specific charging information unless you have verified it from the database

Respond naturally to the user's message:"""

    response = call_claude_api(prompt, max_tokens=1000)
    
    return response               

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
    """Search for departments based on user query using flexible matching"""
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
    
    # Search by name, HR function group, or HR group using flexible search
    search_cols = ['Department Name', 'HR Function Group', 'HR Group']
    mask = pd.Series([False] * len(df))
    
    for col in search_cols:
        if col in df.columns:
            # Use flexible search for each column
            matched = flexible_search(df, col, query)
            mask |= df.index.isin(matched.index)
    
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
