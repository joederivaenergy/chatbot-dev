import os
import json
import uuid
import boto3
import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
import time
import re
import base64
import io

# --- Config ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
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

# --- Chilton Manual CSV Files (placeholder - add paths when ready) ---
CHILTON_CSV_FILES = {
    # 'wind_turbine_maintenance': 'csvs/chilton_wind_turbine.csv',
    # Add your Chilton Manual CSVs here
}

# ============================================
# MODES
# ============================================

MODES = {
    "general": {
        "label": "General Chat",
        "icon": " ",
        "color": "#4A90D9",
        "description": "Ask questions, explore ideas, and get instant answers!",
        "button_style": "primary"
    },
    "charging": {
        "label": "Charging Guidelines",
        "icon": " ",
        "color": "#F5A623",
        "description": "Get charging codes, account numbers, projects & departments.",
        "button_style": "secondary"
    },
    "chilton": {
        "label": "Chilton Manual",
        "icon": " ",
        "color": "#7ED321",
        "description": "Wind farm maintenance & troubleshooting guidance.",
        "button_style": "secondary"
    }
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
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                data[team] = df
            except Exception as e:
                st.error(f"Error loading {team} CSV: {e}")
                data[team] = pd.DataFrame()
        else:
            data[team] = pd.DataFrame()
    
    for ref_type, filepath in REFERENCE_CSV_FILES.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                data[ref_type] = df
            except Exception as e:
                st.error(f"Error loading {ref_type} CSV: {e}")
                data[ref_type] = pd.DataFrame()
        else:
            data[ref_type] = pd.DataFrame()

    for name, filepath in CHILTON_CSV_FILES.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
                data[name] = df
            except Exception as e:
                st.error(f"Error loading Chilton CSV {name}: {e}")
                data[name] = pd.DataFrame()
    
    return data

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
        messages = self.get_messages()
        if not messages:
            return "No previous conversation."
        history_text = "Previous conversation:\n"
        for msg in messages[-10:]:
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
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
/* ===== MODE BUTTON STYLING ===== */
div[data-testid="stButton"] > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    text-align: left;
    justify-content: flex-start;
    padding: 12px 16px;
    margin-bottom: 6px;
    font-size: 0.95rem;
    border: 3px solid #ddd;
    background-color: #f8f9fa;
    color: #333;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background-color: #f0f2f6;
    border-color: #4A90D9;
    color: #1a6fa8;
}

/* ===== ACTIVE MODE BANNER ===== */
.mode-banner {
    padding: 10px 16px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 12px;
    text-align: center;
}
.mode-banner-general  { background: #e8f4fd; color: #1a6fa8; border: 1.5px solid #4A90D9; }
.mode-banner-charging { background: #fff8ec; color: #a0660a; border: 1.5px solid #F5A623; }
.mode-banner-chilton  { background: #f0faec; color: #3a7a10; border: 1.5px solid #7ED321; }

/* ===== SIDEBAR SECTION HEADERS ===== */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin: 16px 0 6px 0;
}

/* ===== ATTACHMENT PREVIEW STRIP ===== */
.attachment-preview {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 6px 0;
    margin-bottom: 4px;
}

.attachment-chip {
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.8rem;
    color: #3730a3;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ===== PLUS BUTTON (ATTACH) ===== */
section[data-testid="stMain"] div[data-testid="stButton"].plus-btn > button {
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    font-size: 1.3rem !important;
    background: #f3f4f6 !important;
    border: 1.5px solid #d1d5db !important;
    color: #374151 !important;
    min-width: unset !important;
    display: flex;
    align-items: center;
    justify-content: center;
}

section[data-testid="stMain"] div[data-testid="stButton"].plus-btn > button:hover {
    background: #e5e7eb !important;
    border-color: #4A90D9 !important;
    color: #1a6fa8 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "general"  # default: free chat

if "extracted_context" not in st.session_state:
    st.session_state.extracted_context = {
        "team": None,
        "keywords": None,
        "location": None,
        "exact_description": None
    }

if "in_charging_flow" not in st.session_state:
    st.session_state.in_charging_flow = False

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "pending_files" not in st.session_state:
    st.session_state.pending_files = []

# ============================================
# SIDEBAR
# ============================================

if os.path.exists("Deriva-Logo.png"):
    st.sidebar.image("Deriva-Logo.png", width=200)
else:
    st.sidebar.markdown("## ⚡ Diva")

st.sidebar.markdown("---")

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
        
with st.sidebar.expander(" ** **", expanded=True):
    if st.button("New Session"):
        reset_history()
        st.rerun()
    
# --- MODE SELECTOR ---
st.sidebar.markdown('<div class="sidebar-section">⚙️ Select Mode</div>', unsafe_allow_html=True)

current_mode = st.session_state.chat_mode
general_active = "✅ " if current_mode == "general" else ""
charging_active = "✅ " if current_mode == "charging" else ""
chilton_active = "✅ " if current_mode == "chilton" else ""

def set_mode(mode: str):
    if st.session_state.chat_mode != mode:
        st.session_state.chat_mode = mode
        # Reset charging context when switching modes
        st.session_state.extracted_context = {
            "team": None,
            "keywords": None,
            "location": None,
            "exact_description": None
        }
        st.session_state.in_charging_flow = False

if st.sidebar.button(f"{general_active} General Chat", key="btn_general"):
    set_mode("general")
    st.rerun()

if st.sidebar.button(f"{charging_active} Charging Guidelines", key="btn_charging"):
    set_mode("charging")
    st.rerun()

if st.sidebar.button(f"{chilton_active} Chilton Manual", key="btn_chilton"):
    set_mode("chilton")
    st.rerun()

st.sidebar.markdown("---")

# Mode-specific info panels
if current_mode == "charging":
    with st.sidebar.expander("ℹ️ Charging Guidelines", expanded=False):
        st.markdown("""
        About charging questions, Diva provides:
        - Account Number
        - Location
        - Company ID
        - Project
        - Department
        ---
        **Note**: the project ID is for Concur and Timesheets.
        For more info: [O&M Charging Guidelines](https://derivaenergy.sharepoint.com/:x:/r/sites/DerivaFinance/_layouts/15/Doc.aspx?sourcedoc=%7B3CD9F65D-C693-4CE8-904C-91074451F098%7D&file=Deriva%20OM%20Charging%20Guidelines.xlsx&action=default&mobileredirect=true)
        """)

elif current_mode == "chilton":
    with st.sidebar.expander("ℹ️ Chilton Manual", expanded=False):
        st.markdown("""
        The Chilton Manual mode helps with:
        - Wind turbine maintenance procedures
        - Troubleshooting guides
        - Scheduled maintenance tasks
        - Component-specific guidance
        ---
        *CSV data for this module will be loaded when available.*
        """)

st.sidebar.divider()

with st.sidebar.expander("📧 Support"):
    st.markdown("[Report an issue](mailto:joe.cheng@derivaenergy.com)")

st.sidebar.caption("Diva The AI Chatbot is made by Deriva Energy and is for internal use only. It may contain errors.")
# ============================================
# HEADER
# ============================================

st.markdown("<h1 style='text-align: center;'>⚡ Meet Diva!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Deriva's AI Chatbot</p>", unsafe_allow_html=True)

# Active mode banner
mode_info = MODES[current_mode]
banner_class = f"mode-banner mode-banner-{current_mode}"
st.markdown(
    f'<div class="{banner_class}">{mode_info["icon"]} Mode: <strong>{mode_info["label"]}</strong> — {mode_info["description"]}</div>',
    unsafe_allow_html=True
)

# ============================================
# LLM HELPER FUNCTIONS
# ============================================

def call_claude(system_prompt: str, user_message: str, include_history: bool = True) -> str:
    """Call Claude via Bedrock with conversation history"""
    try:
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
                "messages": [{"role": "user", "content": full_message}]
            })
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

# ============================================
# GENERAL ASSISTANT PROMPT
# ============================================

GENERAL_ASSISTANT_PROMPT = """
You are Diva, a friendly and helpful AI assistant for Deriva Energy employees.

You can help with anything — coding, writing, analysis, general questions, brainstorming, and more.
You are currently in GENERAL CHAT mode, so behave like a capable, warm, and helpful AI assistant.

If users ask about charging codes or wind farm maintenance, let them know they can switch to the
appropriate mode using the sidebar buttons for more focused help.

Keep responses concise and friendly. You are an internal tool for Deriva Energy employees.
"""

# def generate_natural_response(user_query: str) -> str:
#     """Generate natural language response for general chat mode"""
#     response = call_claude(GENERAL_ASSISTANT_PROMPT, user_query, include_history=True)
#     if not response:
#         return "I'm here to help! Could you rephrase your question?"
#     return response

def generate_natural_response(user_query: str, uploaded_files: list = None) -> str:
    """Generate response, optionally with attached files/images"""
    if uploaded_files:
        response = call_claude_with_media(GENERAL_ASSISTANT_PROMPT, user_query, uploaded_files)
    else:
        response = call_claude(GENERAL_ASSISTANT_PROMPT, user_query, include_history=True)
    if not response:
        return "I'm here to help! Could you rephrase your question?"
    return response

# ============================================
# FILE / IMAGE PROCESSING (GENERAL CHAT)
# ============================================

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "webp"]
SUPPORTED_FILE_TYPES = ["pdf", "txt", "csv", "py", "json", "md", "docx"]

def encode_image_to_base64(file_bytes: bytes) -> str:
    """Encode image bytes to base64 string"""
    return base64.standard_b64encode(file_bytes).decode("utf-8")

def extract_text_from_file(uploaded_file) -> str:
    """Extract text content from uploaded file"""
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".txt") or filename.endswith(".md") or filename.endswith(".py"):
        return file_bytes.decode("utf-8", errors="ignore")

    elif filename.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            return f"CSV File: {uploaded_file.name}\n\n{df.to_string(index=False)}"
        except Exception as e:
            return f"Could not parse CSV: {e}"

    elif filename.endswith(".json"):
        try:
            data = json.loads(file_bytes.decode("utf-8"))
            return f"JSON File: {uploaded_file.name}\n\n{json.dumps(data, indent=2)}"
        except Exception as e:
            return f"Could not parse JSON: {e}"

    elif filename.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return f"PDF File: {uploaded_file.name}\n\n{text}"
        except ImportError:
            return "PDF support requires `pypdf`. Run: pip install pypdf"
        except Exception as e:
            return f"Could not parse PDF: {e}"

    else:
        return f"[Unsupported file type: {uploaded_file.name}]"


def call_claude_with_media(system_prompt: str, user_message: str, uploaded_files: list) -> str:
    """Call Claude with text + optional images/files"""
    try:
        content_blocks = []

        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            uploaded_file.seek(0)  # reset pointer

            if file_ext in SUPPORTED_IMAGE_TYPES:
                # Add as image block
                image_data = encode_image_to_base64(uploaded_file.read())
                media_type_map = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "gif": "image/gif", "webp": "image/webp"
                }
                media_type = media_type_map.get(file_ext, "image/png")
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                })
            else:
                # Extract text and add as text block
                file_text = extract_text_from_file(uploaded_file)
                content_blocks.append({
                    "type": "text",
                    "text": f"[Attached file: {uploaded_file.name}]\n\n{file_text}"
                })

        # Add the user's text message
        history = chat_history.get_formatted_history()
        if history != "No previous conversation.":
            full_text = history + f"\nCurrent question: {user_message}"
        else:
            full_text = user_message

        content_blocks.append({"type": "text", "text": full_text})

        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content_blocks}]
            })
        )

        result = json.loads(response['body'].read())
        return result['content'][0]['text']

    except Exception as e:
        st.error(f"Error calling Claude with media: {e}")
        return None

# ============================================
# CHARGING QUESTION DETECTION
# ============================================

CHARGING_DETECTION_PROMPT = """
You are Diva, a charging guidelines assistant for Deriva Energy.

Determine if the user's question is about CHARGING GUIDELINES or DEPARTMENTS.

Return ONLY valid JSON:
{
  "is_charging_question": true | false,
  "confidence": "high" | "medium" | "low"
}
"""

def is_charging_question(user_query: str) -> bool:
    charging_keywords = [
        "charge", "charging", "code", "codes", "account", "project",
        "department", "expense", "time", "labor", "timesheet", "bill"
    ]
    user_lower = user_query.lower()
    has_charging_keyword = any(keyword in user_lower for keyword in charging_keywords)

    if not has_charging_keyword and len(user_query.split()) > 3:
        return False

    response = call_claude(CHARGING_DETECTION_PROMPT, user_query, include_history=False)
    if not response:
        return has_charging_keyword

    try:
        content = response.strip()
        for marker in ["```json", "```"]:
            content = content.replace(marker, "")
        content = content.strip()
        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        return data.get("is_charging_question", has_charging_keyword)
    except:
        return has_charging_keyword

# ============================================
# EXTRACTION PROMPT
# ============================================

EXTRACTION_PROMPT = """
You are Diva, a charging guidelines assistant. Extract key information from the user's query.

Extract:
1. **team**: ONLY if explicitly mentioned (IT, Finance, HR, Legal, Corporate, Land Services, Commercial, Development, Tech Services, Operations)
2. **keywords**: Key words the user wants to search for
3. **location**: Specific location if mentioned
4. **is_new_query**: Is this a NEW charging question or a follow-up? (true/false)

Return ONLY valid JSON:
{
  "team": "IT" | "Finance" | "HR" | "Legal" | "Corporate" | "Land Services" | "Commercial" | "Development" | "Tech Services" | "Operations" | null,
  "keywords": "search terms" | null,
  "location": "location name" | null,
  "is_new_query": true | false
}
"""

def extract_query_info(user_query: str) -> Dict:
    response = call_claude(EXTRACTION_PROMPT, user_query, include_history=True)
    if not response:
        return st.session_state.extracted_context.copy()
    
    try:
        content = response.strip()
        for marker in ["```json", "```"]:
            content = content.replace(marker, "")
        content = content.strip()
        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        is_new_query = data.get("is_new_query", False)

        if is_new_query:
            extracted = {
                "team": data.get("team"),
                "keywords": data.get("keywords"),
                "location": data.get("location"),
                "exact_description": None
            }
            st.session_state.extracted_context = extracted
            return extracted

        extracted = {
            "team": data.get("team"),
            "keywords": data.get("keywords"),
            "location": data.get("location"),
            "exact_description": st.session_state.extracted_context.get("exact_description")
        }
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
    except:
        return st.session_state.extracted_context.copy()

def is_likely_new_query(user_input: str) -> bool:
    user_lower = user_input.lower().strip()
    new_query_phrases = ["how to charge", "where to charge", "charge for", "charging for", "codes for", "what about", "how about", "need codes", "looking for"]
    for phrase in new_query_phrases:
        if phrase in user_lower:
            return True
    if len(user_input.split()) <= 3:
        return False
    question_words = ["how", "what", "where", "which", "can", "do"]
    first_word = user_lower.split()[0] if user_lower.split() else ""
    if first_word in question_words:
        return True
    return False

# ============================================
# CSV SEARCH FUNCTIONS
# ============================================

def search_descriptions_by_keywords(team: str, keywords: str) -> List[str]:
    if team not in ALL_TEAM_DATA or ALL_TEAM_DATA[team].empty:
        return []
    df = ALL_TEAM_DATA[team]
    search_words = [w.strip().lower() for w in keywords.split() if w.strip()]
    matching_descriptions = set()
    for idx, row in df.iterrows():
        description = str(row['Description']).lower()
        description_words = re.findall(r'\b\w+\b', description)
        for search_word in search_words:
            if search_word in description_words:
                matching_descriptions.add(row['Description'])
                break
    return sorted(list(matching_descriptions))

def get_charging_data(team: str, exact_description: str, location: str = None) -> Tuple[pd.DataFrame, bool]:
    if team not in ALL_TEAM_DATA or ALL_TEAM_DATA[team].empty:
        return pd.DataFrame(), False
    df = ALL_TEAM_DATA[team]
    matches = df[df['Description'] == exact_description]
    if matches.empty:
        return pd.DataFrame(), False
    has_multiple = len(matches) > 1
    if location and has_multiple:
        location_matches = matches[matches['Location'].str.lower() == location.lower()]
        if not location_matches.empty:
            matches = location_matches
    return matches, has_multiple

# ============================================
# FORMATTING FUNCTIONS
# ============================================

def format_charging_info(row: pd.Series) -> str:
    return f"""- **Description:** {row['Description']}
- **Account:** {row['Account']}
- **Location:** {row['Location']}
- **Company ID:** {row['Company ID']}
- **Project:** {row['Project']}
- **Department:** {row['Department']}"""

def format_multiple_variants(team: str, matches: pd.DataFrame) -> str:
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

def check_if_selecting_from_list(user_input: str, extracted: Dict) -> str:
    team = extracted.get("team")
    keywords = extracted.get("keywords")
    if not team or not keywords:
        return None
    matching_descriptions = search_descriptions_by_keywords(team, keywords)
    if len(matching_descriptions) <= 1:
        return None
    user_input_clean = user_input.strip()
    if user_input_clean.isdigit():
        idx = int(user_input_clean) - 1
        if 0 <= idx < len(matching_descriptions):
            return matching_descriptions[idx]
    user_lower = user_input_clean.lower()
    for desc in matching_descriptions:
        if desc.lower() == user_lower or desc.lower() in user_lower:
            return desc
    return None

# ============================================
# CHARGING FLOW
# ============================================

def process_charging_question(user_input: str) -> str:
    st.session_state.in_charging_flow = True
    if is_likely_new_query(user_input):
        st.session_state.extracted_context = {"team": None, "keywords": None, "location": None, "exact_description": None}
    extracted = extract_query_info(user_input)
    if extracted.get("keywords") and not extracted.get("exact_description"):
        selected_description = check_if_selecting_from_list(user_input, extracted)
        if selected_description:
            st.session_state.extracted_context["exact_description"] = selected_description
            extracted["exact_description"] = selected_description

    team = extracted.get("team")
    keywords = extracted.get("keywords")
    location = extracted.get("location")
    exact_description = extracted.get("exact_description")

    if not team:
        return "Which team are you with? (IT, Finance, HR, Legal, Corporate, Land Services, Commercial, Development or Tech Services)"

    if team and team.lower() == "operations":
        st.session_state.extracted_context = {"team": None, "keywords": None, "location": None, "exact_description": None}
        st.session_state.in_charging_flow = False
        return "For Operations team charging codes, please refer to the **Operations Sections** in the [O&M Charging Guidelines](https://derivaenergy.sharepoint.com/:x:/r/sites/DerivaFinance/_layouts/15/Doc.aspx?sourcedoc=%7B3CD9F65D-C693-4CE8-904C-91074451F098%7D&file=Deriva%20OM%20Charging%20Guidelines.xlsx&action=default&mobileredirect=true)."

    if not keywords and not exact_description:
        return "What would you like to charge for?"

    if exact_description:
        matches, has_multiple = get_charging_data(team, exact_description, location)
        if matches.empty:
            st.session_state.extracted_context["exact_description"] = None
            return f"I couldn't find charging codes for '{exact_description}' in {team} team."
        st.session_state.extracted_context = {"team": None, "keywords": None, "location": None, "exact_description": None}
        st.session_state.in_charging_flow = False
        if has_multiple:
            return format_multiple_variants(team, matches)
        return format_charging_info(matches.iloc[0])

    matching_descriptions = search_descriptions_by_keywords(team, keywords)
    if not matching_descriptions:
        return f"I couldn't find any charging information for '{keywords}' in {team} team. Could you check the description and try again?"

    if len(matching_descriptions) == 1:
        st.session_state.extracted_context["exact_description"] = matching_descriptions[0]
        matches, has_multiple = get_charging_data(team, matching_descriptions[0], location)
        st.session_state.extracted_context = {"team": None, "keywords": None, "location": None, "exact_description": None}
        st.session_state.in_charging_flow = False
        if has_multiple:
            return format_multiple_variants(team, matches)
        return format_charging_info(matches.iloc[0])

    result = f"I found {len(matching_descriptions)} charging codes matching '{keywords}' in {team} team:\n\n"
    for idx, desc in enumerate(matching_descriptions, 1):
        result += f"{idx}. {desc}\n"
    result += "\nWhich one are you looking for?"
    return result

# ============================================
# DEPARTMENT FUNCTIONS
# ============================================

def is_department_question(user_query: str) -> bool:
    query_lower = user_query.lower()
    department_keywords = ['department', 'dept', 'departments', 'depts', 'department number', 'dept number', 'department code', 'how many department', 'list department', 'show department', 'what department', 'which department', 'all department']
    return any(keyword in query_lower for keyword in department_keywords)

def search_departments(query: str) -> pd.DataFrame:
    if 'department_list' not in ALL_TEAM_DATA:
        return pd.DataFrame()
    df = ALL_TEAM_DATA['department_list']
    query_lower = query.lower()
    if any(word in query_lower for word in ['all', 'list', 'show all', 'how many']):
        return df
    dept_numbers = re.findall(r'\b\d{4}\b', query)
    if dept_numbers:
        return df[df['Department Number'].astype(str).isin(dept_numbers)]
    search_cols = ['Department Name', 'HR Function Group', 'HR Group']
    mask = pd.Series([False] * len(df))
    for col in search_cols:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(query_lower, na=False, regex=False)
    return df[mask]

def format_department_info(departments_df: pd.DataFrame, query: str) -> str:
    if departments_df.empty:
        return "I couldn't find any departments matching your query. Please try again with different keywords."
    query_lower = query.lower()
    if any(word in query_lower for word in ['how many', 'count', 'total']):
        total = len(departments_df)
        response = f"**Total Departments: {total}**\n\n"
        if 'HR Function Group' in departments_df.columns:
            grouped = departments_df.groupby('HR Function Group').size().reset_index(name='Count')
            response += "**Breakdown by HR Function Group:**\n"
            for _, row in grouped.iterrows():
                response += f"- {row['HR Function Group']}: {row['Count']} department(s)\n"
        return response
    if any(word in query_lower for word in ['list', 'all', 'show all']) and len(departments_df) > 5:
        response = f"**Found {len(departments_df)} departments:**\n\n"
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
    response = f"**Found {len(departments_df)} departments. Here are the first 10:**\n\n"
    for _, row in departments_df.head(10).iterrows():
        response += f"- {row['Department Number']}: {row['Department Name']}\n"
    if len(departments_df) > 10:
        response += f"\n*...and {len(departments_df) - 10} more. Please refine your search.*"
    return response

def process_department_question(user_input: str) -> str:
    departments = search_departments(user_input)
    return format_department_info(departments, user_input)

# ============================================
# CHILTON MANUAL MODE
# ============================================

CHILTON_SYSTEM_PROMPT = """
You are Diva in CHILTON MANUAL mode — a wind farm maintenance expert for Deriva Energy.

Your focus is exclusively on:
- Wind turbine maintenance procedures and schedules
- Troubleshooting mechanical and electrical issues
- Component-specific guidance (blades, gearboxes, generators, pitch systems, yaw systems, etc.)
- Safety procedures for wind farm operations
- Preventive and corrective maintenance

When CSV data is provided in the conversation, use it to answer questions precisely.
If no data is available yet for a specific topic, answer from general wind turbine maintenance knowledge.

Be precise, safety-conscious, and practical. Use clear step-by-step formatting when describing procedures.
Always mention relevant safety precautions.
"""

def process_chilton_question(user_input: str) -> str:
    """Process wind farm / Chilton Manual questions"""
    # In the future, this will query CHILTON_CSV_FILES similar to charging
    # For now, use LLM with the specialized system prompt
    response = call_claude(CHILTON_SYSTEM_PROMPT, user_input, include_history=True)
    if not response:
        return "I couldn't retrieve maintenance guidance at this time. Please try again or refer to your Chilton Manual documentation."
    return response

# ============================================
# MAIN PROCESSING LOGIC
# ============================================

# def process_message(user_input: str) -> str:
#     """Route message to the correct handler based on active mode"""
    
#     mode = st.session_state.chat_mode

#     # Handle greetings regardless of mode
#     greetings = ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]
#     if user_input.lower().strip() in greetings or any(user_input.lower().strip().startswith(g + " ") for g in greetings):
#         if mode == "general":
#             return "Hi there! I'm Diva, your AI assistant. How can I help you today?"
#         elif mode == "charging":
#             return "Hi! I'm in **Charging Guidelines** mode. Ask me about charging codes, accounts, projects, or departments!"
#         elif mode == "chilton":
#             return "Hi! I'm in **Chilton Manual** mode. Ask me about wind turbine maintenance, troubleshooting, or procedures!"

#     # ---- GENERAL MODE ----
#     if mode == "general":
#         st.session_state.in_charging_flow = False
#         return generate_natural_response(user_input)

#     # ---- CHARGING MODE ----
#     elif mode == "charging":
#         if is_department_question(user_input):
#             st.session_state.in_charging_flow = False
#             return process_department_question(user_input)
#         if st.session_state.in_charging_flow:
#             return process_charging_question(user_input)
#         if is_charging_question(user_input):
#             return process_charging_question(user_input)
#         else:
#             # Still in charging mode but question seems off-topic — redirect politely
#             return (
#                 "I'm currently focused on **Charging Guidelines**. "
#                 "Ask me about charging codes, account numbers, projects, or departments. "
#                 "If you'd like to chat freely, switch to ** General Chat** mode in the sidebar!"
#             )

#     # ---- CHILTON MODE ----
#     elif mode == "chilton":
#         return process_chilton_question(user_input)

#     # Fallback
#     return generate_natural_response(user_input)

def process_message(user_input: str, uploaded_files: list = None) -> str:
    mode = st.session_state.chat_mode

    greetings = ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]
    if user_input.lower().strip() in greetings or any(user_input.lower().strip().startswith(g + " ") for g in greetings):
        if mode == "general":
            return "Hi there! I'm Diva, your AI assistant. How can I help you today?"
        elif mode == "charging":
            return "Hi! I'm in **Charging Guidelines** mode. Ask me about charging codes, accounts, projects, or departments!"
        elif mode == "chilton":
            return "Hi! I'm in **Chilton Manual** mode. Ask me about wind turbine maintenance, troubleshooting, or procedures!"

    # ---- GENERAL MODE ----
    if mode == "general":
        st.session_state.in_charging_flow = False
        return generate_natural_response(user_input, uploaded_files=uploaded_files)

    # ---- CHARGING MODE ----
    elif mode == "charging":
        if is_department_question(user_input):
            st.session_state.in_charging_flow = False
            return process_department_question(user_input)
        if st.session_state.in_charging_flow:
            return process_charging_question(user_input)
        if is_charging_question(user_input):
            return process_charging_question(user_input)
        else:
            return (
                "I'm currently focused on **Charging Guidelines**. "
                "Ask me about charging codes, account numbers, projects, or departments. "
                "If you'd like to chat freely, switch to **General Chat** mode in the sidebar!"
            )

    # ---- CHILTON MODE ----
    elif mode == "chilton":
        return process_chilton_question(user_input)

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

# Inject additional CSS
st.markdown(ADDITIONAL_CSS, unsafe_allow_html=True)

placeholders = {
    "general": "Ask me anything...",
    "charging": "Ask about charging codes, accounts, or departments...",
    "chilton": "Ask about wind turbine maintenance or procedures..."
}

# ---- General mode: Claude-style input with + button ----
if st.session_state.chat_mode == "general":

    # Show attachment previews above input if files are staged
    if st.session_state.pending_files:
        preview_html = '<div class="attachment-preview">'
        for f in st.session_state.pending_files:
            ext = f.name.split(".")[-1].upper()
            icon = "🖼️" if ext.lower() in SUPPORTED_IMAGE_TYPES else "📄"
            preview_html += f'<span class="attachment-chip">{icon} {f.name}</span>'
        preview_html += "</div>"
        st.markdown(preview_html, unsafe_allow_html=True)

    # Layout: [+] [chat input] in columns
    col_plus, col_input = st.columns([0.06, 0.94])

    with col_plus:
        # Use a unique key so toggle works
        st.markdown('<div class="plus-btn">', unsafe_allow_html=True)
        if st.button("＋", key="toggle_uploader", help="Attach image or file"):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_input:
        user_input = st.chat_input(
            placeholders["general"],
            key="general_chat_input"
        )

    # Show file uploader when + is clicked
    if st.session_state.show_uploader:
        with st.container():
            new_files = st.file_uploader(
                "Attach files or images",
                type=SUPPORTED_IMAGE_TYPES + SUPPORTED_FILE_TYPES,
                accept_multiple_files=True,
                key=f"file_uploader_{len(st.session_state.get('messages_count', []))}",
                help="Supports: PNG, JPG, GIF, WEBP, PDF, TXT, CSV, PY, JSON, MD",
                label_visibility="visible"
            )
            if new_files:
                st.session_state.pending_files = new_files
                st.session_state.show_uploader = False
                st.rerun()

            col_cancel, _ = st.columns([0.2, 0.8])
            with col_cancel:
                if st.button("✕ Cancel", key="cancel_upload"):
                    st.session_state.show_uploader = False
                    st.rerun()

    # --- Handle paste via clipboard (JavaScript bridge) ---
    # Streamlit doesn't natively support paste, but we inject a JS listener
    # that encodes a pasted image and sends it via a query param workaround.
    # NOTE: This is a best-effort approach — for full paste support, a custom
    # Streamlit component would be needed. We show a helpful tip instead:
    if st.session_state.pending_files:
        st.caption(
            f"📎 {len(st.session_state.pending_files)} file(s) attached. "
            "Type your message and press Enter to send."
        )
        if st.button("🗑️ Clear attachments", key="clear_attachments"):
            st.session_state.pending_files = []
            st.rerun()

    # Process submission
    if user_input:
        uploaded_files = st.session_state.pending_files or None

        display_message = user_input
        if uploaded_files:
            file_names = ", ".join([f.name for f in uploaded_files])
            display_message += f"\n\n📎 *Attached: {file_names}*"

        with st.chat_message("user"):
            st.markdown(display_message)
            # Show image previews inline in chat bubble
            if uploaded_files:
                for f in uploaded_files:
                    if f.name.split(".")[-1].lower() in SUPPORTED_IMAGE_TYPES:
                        f.seek(0)
                        st.image(f, width=300)

        chat_history.add_message("user", display_message)

        with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
                response = process_message(user_input, uploaded_files=None)

            if response:
                st.markdown(response)
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
