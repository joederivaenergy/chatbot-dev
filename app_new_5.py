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
from datetime import datetime

# --- Config ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
DDB_TABLE_NAME = os.getenv("DDB_TABLE_NAME", "diva_chat_history")
DDB_SESSIONS_TABLE = os.getenv("DDB_SESSIONS_TABLE", "diva_sessions")

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

REFERENCE_CSV_FILES = {
    'department_list': 'csvs/Guidelines_cleaned_dept_list.csv'
}

CHILTON_CSV_FILES = {}

# ============================================
# MODES
# ============================================

MODES = {
    "general": {
        "label": "General Chat",
        "icon": "��",
        "color": "#4A90D9",
        "description": "Ask questions, explore ideas, and get instant answers!",
    },
    "charging": {
        "label": "Charging Guidelines",
        "icon": "��",
        "color": "#F5A623",
        "description": "Get charging codes, account numbers, projects & departments.",
    },
    "chilton": {
        "label": "Chilton Manual",
        "icon": "��",
        "color": "#7ED321",
        "description": "Wind farm maintenance & troubleshooting guidance.",
    }
}

# ============================================
# LOAD CSV DATA
# ============================================

@st.cache_data
def load_all_csvs():
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
                data[name] = pd.DataFrame()

    return data

ALL_TEAM_DATA = load_all_csvs()

# ============================================
# AWS CLIENTS
# ============================================

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

def create_tables_if_not_exist():
    # Chat history table
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
        except Exception as e:
            st.error(f"❌ Failed to create chat table: {e}")
            return False

    # Sessions metadata table
    try:
        table = dynamodb.Table(DDB_SESSIONS_TABLE)
        table.load()
    except dynamodb.meta.client.exceptions.ResourceNotFoundException:
        try:
            table = dynamodb.create_table(
                TableName=DDB_SESSIONS_TABLE,
                KeySchema=[
                    {'AttributeName': 'session_id', 'KeyType': 'HASH'},
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'session_id', 'AttributeType': 'S'},
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            table.wait_until_exists()
        except Exception as e:
            st.error(f"❌ Failed to create sessions table: {e}")
            return False

    return True

if not create_tables_if_not_exist():
    st.stop()

# ============================================
# SESSION MANAGER
# ============================================

class SessionManager:
    def __init__(self):
        self.table = dynamodb.Table(DDB_SESSIONS_TABLE)
        self.chat_table = dynamodb.Table(DDB_TABLE_NAME)

    def upsert_session(self, session_id: str, title: str = None, mode: str = "general"):
        now = datetime.utcnow().isoformat()
        item = {
            'session_id': session_id,
            'updated_at': now,
            'mode': mode,
        }
        # Only set created_at and title on first creation
        try:
            existing = self.table.get_item(Key={'session_id': session_id}).get('Item')
            if existing:
                # Update timestamp and optionally title
                update_expr = "SET updated_at = :ua"
                expr_vals = {':ua': now}
                if title and not existing.get('title'):
                    update_expr += ", title = :t"
                    expr_vals[':t'] = title
                self.table.update_item(
                    Key={'session_id': session_id},
                    UpdateExpression=update_expr,
                    ExpressionAttributeValues=expr_vals
                )
            else:
                item['created_at'] = now
                item['title'] = title or "New conversation"
                self.table.put_item(Item=item)
        except Exception as e:
            st.error(f"Session upsert error: {e}")

    def get_recent_sessions(self, limit: int = 20) -> List[Dict]:
        try:
            response = self.table.scan()
            sessions = response.get('Items', [])
            sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            return sessions[:limit]
        except Exception as e:
            return []

    def delete_session(self, session_id: str):
        try:
            # Delete session metadata
            self.table.delete_item(Key={'session_id': session_id})
            # Delete all messages
            response = self.chat_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(session_id)
            )
            for item in response.get('Items', []):
                self.chat_table.delete_item(
                    Key={
                        'session_id': item['session_id'],
                        'message_timestamp': str(item['message_timestamp'])
                    }
                )
        except Exception as e:
            st.error(f"Delete session error: {e}")

    def get_session(self, session_id: str) -> Dict:
        try:
            return self.table.get_item(Key={'session_id': session_id}).get('Item', {})
        except:
            return {}

session_manager = SessionManager()

# ============================================
# DYNAMODB CHAT HISTORY
# ============================================

class DynamoDBChatHistory:
    def __init__(self, table_name: str, session_id: str):
        self.session_id = session_id
        self.table = dynamodb.Table(table_name)

    def get_messages(self) -> List[Dict]:
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('session_id').eq(self.session_id),
                ScanIndexForward=True
            )
            return response.get('Items', [])
        except Exception as e:
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

    def get_messages_for_bedrock(self) -> List[Dict]:
        """Return properly formatted messages array for Bedrock API"""
        messages = self.get_messages()
        formatted = []
        for msg in messages[-20:]:  # last 20 messages for context window
            role = msg.get('message_type', 'user')
            # Normalize role
            if role in ('ai', 'assistant'):
                role = 'assistant'
            else:
                role = 'user'
            content = msg.get('content', '')
            if content:
                formatted.append({"role": role, "content": content})
        return formatted

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_icon="⚡",
    page_title="Diva — Deriva Energy",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================
# POLISHED CSS
# ============================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Sidebar section labels ── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a5568 !important;
    margin: 18px 0 8px 0;
    padding-left: 2px;
}

/* ── Mode buttons ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    padding: 10px 14px;
    margin-bottom: 4px;
    font-size: 0.88rem;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    border-radius: 8px;
    border: 1px solid transparent;
    background: transparent;
    color: #94a3b8 !important;
    transition: all 0.15s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #1a1f2e !important;
    color: #e2e8f0 !important;
    border-color: #2d3748;
}

/* ── Active mode button ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    background: #1a2744 !important;
    color: #60a5fa !important;
    border-color: #2563eb;
}

/* ── New chat button ── */
.new-chat-btn > div[data-testid="stButton"] > button {
    background: #1d4ed8 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 14px !important;
    width: 100% !important;
    transition: background 0.15s ease !important;
}
.new-chat-btn > div[data-testid="stButton"] > button:hover {
    background: #1e40af !important;
}

/* ── Session history items ── */
.session-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 8px;
    margin-bottom: 2px;
    cursor: pointer;
    transition: background 0.12s ease;
}
.session-item:hover {
    background: #1a1f2e;
}
.session-item.active {
    background: #1a2744;
    border-left: 3px solid #2563eb;
}
.session-title {
    font-size: 0.83rem;
    font-weight: 500;
    color: #cbd5e1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 160px;
}
.session-date {
    font-size: 0.72rem;
    color: #4a5568;
    margin-top: 1px;
}

/* ── Mode banner ── */
.mode-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 16px;
}
.mode-banner-general  { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.mode-banner-charging { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.mode-banner-chilton  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

/* ── Header ── */
.diva-header {
    text-align: center;
    padding: 8px 0 4px 0;
}
.diva-header h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #0f172a;
    margin: 0;
}
.diva-header p {
    font-size: 0.9rem;
    color: #64748b;
    margin: 4px 0 0 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 0;
}

/* ── Input box ── */
[data-testid="stChatInputContainer"] {
    border-radius: 12px;
    border: 1.5px solid #e2e8f0;
    background: #fff;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08);
}

/* ── File uploader in sidebar ── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #1a1f2e;
    border: 1px dashed #2d3748;
    border-radius: 8px;
    padding: 8px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    font-size: 0.8rem !important;
}

/* ── Divider ── */
section[data-testid="stSidebar"] hr {
    border-color: #1e2130 !important;
    margin: 12px 0 !important;
}

/* ── Footer ── */
.footer-note {
    text-align: center;
    font-size: 0.72rem;
    color: #94a3b8;
    padding: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "general"

if "extracted_context" not in st.session_state:
    st.session_state.extracted_context = {
        "team": None, "keywords": None, "location": None, "exact_description": None
    }

if "in_charging_flow" not in st.session_state:
    st.session_state.in_charging_flow = False

if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False

# Initialize chat history object
chat_history = DynamoDBChatHistory(
    table_name=DDB_TABLE_NAME,
    session_id=st.session_state["session_id"]
)

# Register session in sessions table on first load
if not st.session_state.session_initialized:
    session_manager.upsert_session(
        session_id=st.session_state["session_id"],
        mode=st.session_state.chat_mode
    )
    st.session_state.session_initialized = True

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    # Logo
    if os.path.exists("Deriva-Logo.png"):
        st.image("Deriva-Logo.png", width=160)
    else:
        st.markdown("### ⚡ Diva")

    st.divider()

    # ── NEW CHAT BUTTON ──
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", key="new_chat_btn"):
        # Save current and start fresh
        new_id = str(uuid.uuid4())
        st.session_state["session_id"] = new_id
        st.session_state.extracted_context = {
            "team": None, "keywords": None, "location": None, "exact_description": None
        }
        st.session_state.in_charging_flow = False
        st.session_state.session_initialized = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── MODE SELECTOR ──
    st.markdown('<div class="sidebar-label">Mode</div>', unsafe_allow_html=True)

    current_mode = st.session_state.chat_mode

    for mode_key, mode_info in MODES.items():
        is_active = current_mode == mode_key
        label = f"{mode_info['icon']}  {mode_info['label']}"
        if is_active:
            label = "✓  " + label
        if st.button(label, key=f"mode_{mode_key}", type="primary" if is_active else "secondary"):
            if current_mode != mode_key:
                st.session_state.chat_mode = mode_key
                st.session_state.extracted_context = {
                    "team": None, "keywords": None, "location": None, "exact_description": None
                }
                st.session_state.in_charging_flow = False
                st.rerun()

    st.divider()

    # ── FILE UPLOADER (General mode only) ──
    uploaded_files = None
    if st.session_state.chat_mode == "general":
        st.markdown('<div class="sidebar-label">�� Attach Files</div>', unsafe_allow_html=True)
        SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "webp"]
        SUPPORTED_FILE_TYPES = ["pdf", "txt", "csv", "py", "json", "md"]
        uploaded_files = st.file_uploader(
            "Attach files or images",
            type=SUPPORTED_IMAGE_TYPES + SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed"
        )
        if uploaded_files:
            for f in uploaded_files:
                st.caption(f"�� {f.name}")
        st.divider()

    # ── CHAT HISTORY ──
    st.markdown('<div class="sidebar-label">Recent Chats</div>', unsafe_allow_html=True)

    recent_sessions = session_manager.get_recent_sessions(limit=20)

    if not recent_sessions:
        st.caption("No previous chats.")
    else:
        for sess in recent_sessions:
            sid = sess.get('session_id', '')
            title = sess.get('title', 'New conversation')
            updated = sess.get('updated_at', '')
            is_current = sid == st.session_state["session_id"]

            # Format date
            try:
                dt = datetime.fromisoformat(updated)
                now = datetime.utcnow()
                delta = now - dt
                if delta.days == 0:
                    date_label = "Today"
                elif delta.days == 1:
                    date_label = "Yesterday"
                elif delta.days < 7:
                    date_label = f"{delta.days} days ago"
                else:
                    date_label = dt.strftime("%b %d")
            except:
                date_label = ""

            # Truncate title
            display_title = title[:28] + "…" if len(title) > 28 else title

            col1, col2 = st.columns([5, 1])
            with col1:
                active_class = "active" if is_current else ""
                st.markdown(
                    f"""<div class="session-item {active_class}">
                        <div>
                            <div class="session-title">{display_title}</div>
                            <div class="session-date">{date_label}</div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )
                if st.button("Load", key=f"load_{sid}", help=f"Load: {title}"):
                    st.session_state["session_id"] = sid
                    st.session_state.extracted_context = {
                        "team": None, "keywords": None, "location": None, "exact_description": None
                    }
                    st.session_state.in_charging_flow = False
                    st.session_state.session_initialized = True
                    st.rerun()
            with col2:
                if st.button("��", key=f"del_{sid}", help="Delete this chat"):
                    session_manager.delete_session(sid)
                    if is_current:
                        st.session_state["session_id"] = str(uuid.uuid4())
                        st.session_state.session_initialized = False
                    st.rerun()

    st.divider()

    # ── SUPPORT ──
    with st.expander("�� Support"):
        st.markdown("[Report an issue](mailto:joe.cheng@derivaenergy.com)")

    st.caption("Diva is for internal Deriva Energy use only. It may contain errors.")

# ============================================
# HEADER
# ============================================

st.markdown("""
<div class="diva-header">
    <h1>⚡ Diva</h1>
    <p>Deriva Energy's AI Assistant</p>
</div>
""", unsafe_allow_html=True)

# Mode banner
mode_info = MODES[st.session_state.chat_mode]
st.markdown(
    f'<div class="mode-banner mode-banner-{st.session_state.chat_mode}">'
    f'{mode_info["icon"]} <strong>{mode_info["label"]}</strong> — {mode_info["description"]}'
    f'</div>',
    unsafe_allow_html=True
)

# ============================================
# LLM — CORE CALL WITH PROPER HISTORY
# ============================================

def call_claude(system_prompt: str, user_message: str, include_history: bool = True) -> str:
    """Call Claude via Bedrock with properly structured conversation history"""
    try:
        messages = []

        if include_history:
            # Get properly formatted message history
            history_messages = chat_history.get_messages_for_bedrock()
            messages.extend(history_messages)

        # Append current user message
        messages.append({"role": "user", "content": user_message})

        # Ensure messages alternate correctly (deduplicate consecutive same-role)
        cleaned = []
        for msg in messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages
                cleaned[-1]["content"] += "\n" + msg["content"]
            else:
                cleaned.append(msg)

        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": cleaned
            })
        )

        result = json.loads(response['body'].read())
        return result['content'][0]['text']

    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

# ============================================
# FILE PROCESSING
# ============================================

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "webp"]
SUPPORTED_FILE_TYPES = ["pdf", "txt", "csv", "py", "json", "md"]

def encode_image_to_base64(file_bytes: bytes) -> str:
    return base64.standard_b64encode(file_bytes).decode("utf-8")

def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith((".txt", ".md", ".py")):
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
            text = "".join(page.extract_text() + "\n" for page in reader.pages)
            return f"PDF File: {uploaded_file.name}\n\n{text}"
        except ImportError:
            return "PDF support requires `pypdf`. Run: pip install pypdf"
        except Exception as e:
            return f"Could not parse PDF: {e}"
    return f"[Unsupported file type: {uploaded_file.name}]"

def call_claude_with_media(system_prompt: str, user_message: str, uploaded_files: list) -> str:
    try:
        content_blocks = []

        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            uploaded_file.seek(0)

            if file_ext in SUPPORTED_IMAGE_TYPES:
                image_data = encode_image_to_base64(uploaded_file.read())
                media_type_map = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "gif": "image/gif", "webp": "image/webp"
                }
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type_map.get(file_ext, "image/png"),
                        "data": image_data
                    }
                })
            else:
                file_text = extract_text_from_file(uploaded_file)
                content_blocks.append({
                    "type": "text",
                    "text": f"[Attached file: {uploaded_file.name}]\n\n{file_text}"
                })

        content_blocks.append({"type": "text", "text": user_message})

        # Build messages with history
        history_messages = chat_history.get_messages_for_bedrock()
        messages = history_messages + [{"role": "user", "content": content_blocks}]

        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": messages
            })
        )

        result = json.loads(response['body'].read())
        return result['content'][0]['text']

    except Exception as e:
        st.error(f"Error calling Claude with media: {e}")
        return None

# ============================================
# SYSTEM PROMPTS
# ============================================

GENERAL_ASSISTANT_PROMPT = """
You are Diva, a friendly and capable AI assistant for Deriva Energy employees.
You can help with anything — coding, writing, analysis, general questions, brainstorming, and more.
Keep responses helpful, concise, and warm. You are an internal tool for Deriva Energy employees.
If users ask about charging codes or wind farm maintenance, let them know they can switch modes in the sidebar.
"""

CHILTON_SYSTEM_PROMPT = """
You are Diva in CHILTON MANUAL mode — a wind farm maintenance expert for Deriva Energy.
Focus exclusively on:
- Wind turbine maintenance procedures and schedules
- Troubleshooting mechanical and electrical issues
- Component-specific guidance (blades, gearboxes, generators, pitch systems, yaw systems, etc.)
- Safety procedures for wind farm operations
- Preventive and corrective maintenance

Be precise, safety-conscious, and practical. Use clear step-by-step formatting.
Always mention relevant safety precautions.
"""

# ============================================
# CHARGING DETECTION & EXTRACTION
# ============================================

CHARGING_DETECTION_PROMPT = """
You are a charging guidelines assistant for Deriva Energy.
Determine if the user's question is about CHARGING GUIDELINES or DEPARTMENTS.
Return ONLY valid JSON: {"is_charging_question": true | false, "confidence": "high" | "medium" | "low"}
"""

EXTRACTION_PROMPT = """
You are a charging guidelines assistant. Extract key information from the user's query.
Extract:
1. team: ONLY if explicitly mentioned (IT, Finance, HR, Legal, Corporate, Land Services, Commercial, Development, Tech Services, Operations)
2. keywords: Key words for searching
3. location: Specific location if mentioned
4. is_new_query: Is this a NEW charging question or a follow-up? (true/false)

Return ONLY valid JSON:
{
  "team": "IT" | "Finance" | "HR" | "Legal" | "Corporate" | "Land Services" | "Commercial" | "Development" | "Tech Services" | "Operations" | null,
  "keywords": "search terms" | null,
  "location": "location name" | null,
  "is_new_query": true | false
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
        content = response.strip().replace("```json", "").replace("```", "").strip()
        m = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(m.group() if m else content)
        return data.get("is_charging_question", has_charging_keyword)
    except:
        return has_charging_keyword

def extract_query_info(user_query: str) -> Dict:
    response = call_claude(EXTRACTION_PROMPT, user_query, include_history=True)
    if not response:
        return st.session_state.extracted_context.copy()
    try:
        content = response.strip().replace("```json", "").replace("```", "").strip()
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
            merged[key] = extracted.get(key) or st.session_state.extracted_context.get(key)
        st.session_state.extracted_context = merged
        return merged
    except:
        return st.session_state.extracted_context.copy()

def is_likely_new_query(user_input: str) -> bool:
    user_lower = user_input.lower().strip()
    new_query_phrases = ["how to charge", "where to charge", "charge for", "charging for",
                         "codes for", "what about", "how about", "need codes", "looking for"]
    for phrase in new_query_phrases:
        if phrase in user_lower:
            return True
    if len(user_input.split()) <= 3:
        return False
    question_words = ["how", "what", "where", "which", "can", "do"]
    first_word = user_lower.split()[0] if user_lower.split() else ""
    return first_word in question_words

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

def format_charging_info(row: pd.Series) -> str:
    return (
        f"- **Description:** {row['Description']}\n"
        f"- **Account:** {row['Account']}\n"
        f"- **Location:** {row['Location']}\n"
        f"- **Company ID:** {row['Company ID']}\n"
        f"- **Project:** {row['Project']}\n"
        f"- **Department:** {row['Department']}"
    )

def format_multiple_variants(team: str, matches: pd.DataFrame) -> str:
    description = matches.iloc[0]['Description']
    result = f"**{team} Team — {description}**\n\nThis charging code has **{len(matches)} options**:\n\n"
    for idx, (_, row) in enumerate(matches.iterrows(), 1):
        result += f"---\n**Option {idx}:**\n"
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
        st.session_state.extracted_context = {
            "team": None, "keywords": None, "location": None, "exact_description": None
        }
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
        st.session_state.extracted_context = {
            "team": None, "keywords": None, "location": None, "exact_description": None
        }
        st.session_state.in_charging_flow = False
        return ("For Operations team charging codes, please refer to the **Operations Sections** in the "
                "[O&M Charging Guidelines](https://derivaenergy.sharepoint.com/:x:/r/sites/DerivaFinance/_layouts/15/Doc.aspx?sourcedoc=%7B3CD9F65D-C693-4CE8-904C-91074451F098%7D&file=Deriva%20OM%20Charging%20Guidelines.xlsx&action=default&mobileredirect=true).")

    if not keywords and not exact_description:
        return "What would you like to charge for?"

    if exact_description:
        matches, has_multiple = get_charging_data(team, exact_description, location)
        if matches.empty:
            st.session_state.extracted_context["exact_description"] = None
            return f"I couldn't find charging codes for '{exact_description}' in {team} team."
        st.session_state.extracted_context = {
            "team": None, "keywords": None, "location": None, "exact_description": None
        }
        st.session_state.in_charging_flow = False
        return format_multiple_variants(team, matches) if has_multiple else format_charging_info(matches.iloc[0])

    matching_descriptions = search_descriptions_by_keywords(team, keywords)
    if not matching_descriptions:
        return f"I couldn't find any charging information for '{keywords}' in {team} team. Could you check the description and try again?"

    if len(matching_descriptions) == 1:
        st.session_state.extracted_context["exact_description"] = matching_descriptions[0]
        matches, has_multiple = get_charging_data(team, matching_descriptions[0], location)
        st.session_state.extracted_context = {
            "team": None, "keywords": None, "location": None, "exact_description": None
        }
        st.session_state.in_charging_flow = False
        return format_multiple_variants(team, matches) if has_multiple else format_charging_info(matches.iloc[0])

    result = f"I found **{len(matching_descriptions)}** charging codes matching '{keywords}' in **{team}** team:\n\n"
    for idx, desc in enumerate(matching_descriptions, 1):
        result += f"{idx}. {desc}\n"
    result += "\nWhich one are you looking for?"
    return result

# ============================================
# DEPARTMENT FUNCTIONS
# ============================================

def is_department_question(user_query: str) -> bool:
    query_lower = user_query.lower()
    department_keywords = ['department', 'dept', 'departments', 'depts', 'department number',
                           'dept number', 'department code', 'how many department', 'list department',
                           'show department', 'what department', 'which department', 'all department']
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
        for _, row in departments_df.iterrows():
            response += f"**{row['Department Number']} — {row['Department Name']}**\n"
            for col, label in [('Resp Center', 'Responsibility Center'), ('HR Function Group', 'HR Function Group'),
                                ('HR Group', 'HR Group'), ('SG&A/OPS/DEVEX', 'Category')]:
                if col in row and pd.notna(row.get(col)):
                    response += f"- **{label}:** {row[col]}\n"
            response += "\n"
        return response
    response = f"**Found {len(departments_df)} departments. Showing first 10:**\n\n"
    for _, row in departments_df.head(10).iterrows():
        response += f"- {row['Department Number']}: {row['Department Name']}\n"
    if len(departments_df) > 10:
        response += f"\n*…and {len(departments_df) - 10} more. Please refine your search.*"
    return response

def process_department_question(user_input: str) -> str:
    departments = search_departments(user_input)
    return format_department_info(departments, user_input)

# ============================================
# CHILTON MODE
# ============================================

def process_chilton_question(user_input: str) -> str:
    response = call_claude(CHILTON_SYSTEM_PROMPT, user_input, include_history=True)
    if not response:
        return "I couldn't retrieve maintenance guidance at this time. Please try again or refer to your documentation."
    return response

# ============================================
# GENERAL RESPONSE
# ============================================

def generate_natural_response(user_query: str, uploaded_files: list = None) -> str:
    if uploaded_files:
        response = call_claude_with_media(GENERAL_ASSISTANT_PROMPT, user_query, uploaded_files)
    else:
        response = call_claude(GENERAL_ASSISTANT_PROMPT, user_query, include_history=True)
    return response or "I'm here to help! Could you rephrase your question?"

# ============================================
# MAIN MESSAGE ROUTER
# ============================================

def process_message(user_input: str, uploaded_files: list = None) -> str:
    mode = st.session_state.chat_mode

    greetings = ["hi", "hello", "hey", "yo", "hiya", "good morning", "good afternoon", "good evening"]
    if user_input.lower().strip() in greetings or any(
        user_input.lower().strip().startswith(g + " ") for g in greetings
    ):
        if mode == "general":
            return "Hi there! I'm Diva, Deriva Energy's AI assistant. How can I help you today?"
        elif mode == "charging":
            return "Hi! I'm in **Charging Guidelines** mode. Ask me about charging codes, accounts, projects, or departments!"
        elif mode == "chilton":
            return "Hi! I'm in **Chilton Manual** mode. Ask me about wind turbine maintenance, troubleshooting, or procedures!"

    if mode == "general":
        st.session_state.in_charging_flow = False
        return generate_natural_response(user_input, uploaded_files=uploaded_files)

    elif mode == "charging":
        if is_department_question(user_input):
            st.session_state.in_charging_flow = False
            return process_department_question(user_input)
        if st.session_state.in_charging_flow:
            return process_charging_question(user_input)
        if is_charging_question(user_input):
            return process_charging_question(user_input)
        return (
            "I'm currently in **Charging Guidelines** mode. "
            "Ask me about charging codes, account numbers, projects, or departments. "
            "Switch to **General Chat** in the sidebar for other questions!"
        )

    elif mode == "chilton":
        return process_chilton_question(user_input)

    return generate_natural_response(user_input)

# ============================================
# RENDER CHAT HISTORY
# ============================================

# Re-initialize chat_history with current session_id (may have changed)
chat_history = DynamoDBChatHistory(
    table_name=DDB_TABLE_NAME,
    session_id=st.session_state["session_id"]
)

messages = chat_history.get_messages()
for msg in messages:
    role = "assistant" if msg.get('message_type') in ("ai", "assistant") else "user"
    content = msg.get('content', '')
    with st.chat_message(role):
        st.markdown(content)

# ============================================
# CHAT INPUT
# ============================================

placeholders = {
    "general": "Ask me anything…",
    "charging": "Ask about charging codes, accounts, or departments…",
    "chilton": "Ask about wind turbine maintenance or procedures…"
}

user_input = st.chat_input(placeholders.get(st.session_state.chat_mode, "Type your message…"))

if user_input:
    # Build display message
    display_message = user_input
    if uploaded_files:
        file_names = ", ".join([f.name for f in uploaded_files])
        display_message += f"\n\n�� *Attached: {file_names}*"

    with st.chat_message("user"):
        st.markdown(display_message)
    chat_history.add_message("user", display_message)

    # Update session title from first user message
    current_session = session_manager.get_session(st.session_state["session_id"])
    if not current_session.get('title') or current_session.get('title') == 'New conversation':
        title = user_input[:50]
        session_manager.upsert_session(
            session_id=st.session_state["session_id"],
            title=title,
            mode=st.session_state.chat_mode
        )
    else:
        session_manager.upsert_session(
            session_id=st.session_state["session_id"],
            mode=st.session_state.chat_mode
        )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = process_message(user_input, uploaded_files=uploaded_files)
            st.markdown(response)

    chat_history.add_message("assistant", response)

# ============================================
# FOOTER
# ============================================

st.markdown(
    '<div class="footer-note">Diva is made by Deriva Energy · Internal use only · May contain errors</div>',
    unsafe_allow_html=True
)
