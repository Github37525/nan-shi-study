import streamlit as st
import os
import time
import json
import random
import asyncio
import edge_tts
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.schema import Document 
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from google.api_core.exceptions import InvalidArgument
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="南师书房", page_icon="🍵", layout="centered") # 改为 mobile 布局尝试更紧凑，但 streamlit web版依然是宽屏

# 语录库
NAN_QUOTES = [
    "功成、名遂、<br>身退，<br>天之道。", "世上本无事，<br>庸人自扰之。", 
    "应无所住，<br>而生其心。", "能控制早晨的人，<br>就能控制人生。",
    "静坐修道<br>与长生不老，<br>都在这个“静”字。",
    "英雄到老皆归佛，<br>宿将还山不论兵。"
]

# --- 2. 核心：复刻图1的 CSS 样式 ---
st.markdown("""
<style>
    /* 引入字体 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;700&display=swap');

    /* 全局重置 */
    .stApp {
        /* 核心背景：青瓷色 -> 暖米色 垂直渐变 */
        background: linear-gradient(180deg, #D4E2D4 0%, #F7F5EE 60%, #F7F5EE 100%);
        background-attachment: fixed;
    }
    
    /* 强制字体 */
    html, body, p, div, span {
        font-family: 'Noto Serif SC', serif !important;
        color: #4A3C31;
    }

    /* 隐藏顶部红线和菜单 */
    header, #MainMenu, footer {visibility: hidden;}
    
    /* 标题样式 */
    h1 {
        font-family: 'Noto Serif SC', serif !important;
        color: #3E2723 !important;
        font-weight: 800 !important;
        text-shadow: 0 1px 0 rgba(255,255,255,0.5);
        margin-bottom: 0px !important;
    }

    /* --- 核心组件：语录卡片 (HTML实现) --- */
    .quote-container {
        background-color: #FFFFFF;
        border: 2px solid #5D4037; /* 深褐边框 */
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0 40px 0;
        box-shadow: 0 8px 20px rgba(62, 39, 35, 0.05); /* 极淡的阴影 */
        position: relative;
    }
    .quote-text {
        font-size: 26px;
        font-weight: 700;
        line-height: 1.6;
        color: #3E2723;
        margin-bottom: 20px;
    }
    .quote-author {
        text-align: right;
        font-size: 14px;
        color: #8D6E63;
        margin-top: 10px;
    }
    /* 卡片上的装饰圆点 */
    .dot {
        height: 12px; width: 12px; background-color: #5D4037; border-radius: 50%;
        position: absolute; top: 20px;
    }
    .dot-left { left: 20px; }
    .dot-right { right: 20px; }

    /* --- 聊天气泡美化 --- */
    
    /* 南师 (AI) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        padding: 15px;
    }
    
    /* 用户 (我) - 对应图1虽然没显示用户，但我们要配个色 */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #6D7D70; /* 莫兰迪深青色 */
        border-radius: 16px;
        color: white !important;
        text-align: right;
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessage"]:nth-child(even) p { color: white !important; }

    /* --- 底部输入框悬浮美化 --- */
    /* 这是一个比较暴力的 CSS hack，试图让输入框变圆 */
    .stChatInput {
        padding-bottom: 20px;
    }
    div[data-testid="stChatInput"] {
        border-radius: 40px !important;
        border: 1px solid #D7CCC8 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* 追问按钮样式 */
    .stButton button {
        background-color: rgba(255,255,255,0.4);
        border: 1px solid #8D6E63;
        color: #5D4037;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 页面布局重构 (把内容移到主界面) ---

# 标题区
st.markdown("<h1 style='text-align: center;'>🍵 南师书房</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8D6E63; font-size: 0.8em; margin-bottom: 20px; letter-spacing: 2px;'>—— 此时此处，调息静心 ——</p>", unsafe_allow_html=True)

# ★★★ 关键修改：语录卡片移到主界面顶部 ★★★
if "daily_quote" not in st.session_state:
    st.session_state.daily_quote = random.choice(NAN_QUOTES)

# 使用 HTML 直接渲染那个漂亮的卡片
st.markdown(f"""
    <div class="quote-container">
        <div class="dot dot-left"></div>
        <div class="dot dot-right"></div>
        <div class="quote-text">{st.session_state.daily_quote}</div>
        <div class="quote-author">—— 南怀瑾</div>
    </div>
    
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <span style="font-size: 1.2em; margin-right: 5px;">📜</span>
        <span style="font-weight: bold; color: #5D4037;">今日参悟</span>
    </div>
""", unsafe_allow_html=True)


# --- (以下逻辑功能代码保持不变，只需粘贴你的旧功能代码) ---
# 为了保证代码能跑，我把核心功能函数简写在这里，请务必保留你原来的 RAG 逻辑
# ...

# 1. 功能函数定义区
async def generate_speech(text, output_file="speech_output.mp3"):
    """使用 Edge TTS 生成语音"""
    VOICE = "zh-CN-YunzeNeural"
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_file)
        return True 
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def save_to_logs(user_question, ai_answer, sources):
    """日志记录"""
    try:
        if "gcp_service_account" not in st.secrets: return
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("南师书房日志").sheet1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        source_str = "; ".join([f"{doc.metadata.get('source')}·{doc.metadata.get('chapter')}" for doc in sources]) if sources else "无引用"
        sheet.append_row([timestamp, user_question, ai_answer, source_str])
    except Exception: pass

def get_suggestions(answer_text, llm_engine):
    if not llm_engine: return []
    try:
        prompt = f"基于回答：'{answer_text[:500]}...'，生成3个简短追问。只返回问题，每行一个。"
        res = llm_engine.invoke(prompt)
        return [q.strip() for q in res.content.split('\n') if q.strip()][:3]
    except: return []

# --- RAG 初始化 (请保留你完整的 RAG 代码) ---
@st.cache_resource
def initialize_rag():
    if "GOOGLE_API_KEY" not in st.secrets: st.error("请配置 API Key"); return None
    api_key = st.secrets["GOOGLE_API_KEY"]
    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    index_path = "faiss_index"
    vectorstore = None
    if os.path.exists(index_path):
        try: vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except: pass
    if vectorstore is None:
        # 兜底逻辑
        return None
    retriever = vectorstore.as_retriever()
    
    # 定义 Prompt (V2.0)
    qa_system_prompt = (
        "你现在是南怀瑾先生（南师）。语气慈悲、通俗、幽默。"
        "必须基于参考资料 (Context) 回答。\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),
    ])
    
    # 历史感知
    context_system_prompt = "改写问题..."
    context_prompt = ChatPromptTemplate.from_messages([("system", context_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    history_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)
    return rag_chain, llm

rag_setup = initialize_rag()
if rag_setup: rag_chain, llm_engine = rag_setup
else: rag_chain, llm_engine = None, None

# --- 聊天交互逻辑 ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "哎呀，随便坐。今天心里有什么放不下的吗？"}]

# 显示历史消息
for msg in st.session_state.messages:
    # 这里的 avatar 使用默认，因为 CSS 已经控制了样式，或者你可以换成图片路径
    avatar = "assets/nanshi_icon.png" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "audio_path" in msg and os.path.exists(msg["audio_path"]):
             st.audio(msg["audio_path"], format="audio/mp3")

# 输入框与生成逻辑
if prompt := st.chat_input("请在此输入您与南师的对话..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="assets/nanshi_icon.png"): st.markdown(prompt)

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar="🍵"):
        message_placeholder = st.empty()
        if rag_chain:
            with st.spinner("南师正在沉思..."):
                try:
                    # RAG 逻辑
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user": chat_history.append(HumanMessage(content=msg["content"]))
                        else: chat_history.append(AIMessage(content=msg["content"]))
                    
                    response = rag_chain.invoke({"input": st.session_state.messages[-1]["content"], "chat_history": chat_history})
                    answer = response["answer"]
                    source_documents = response["context"]
                    
                    message_placeholder.markdown(answer)
                    
                    # 引用折叠
                    with st.expander("🔍 点击查看出处"):
                        if source_documents:
                            for i, doc in enumerate(source_documents):
                                st.markdown(f"**📖 {doc.metadata.get('source')}**"); st.caption(doc.page_content); st.markdown("---")
                    
                    # 语音与日志
                    audio_file = f"speech_{int(time.time())}.mp3"
                    is_audio_success = asyncio.run(generate_speech(answer[:300], audio_file))
                    save_to_logs(st.session_state.messages[-1]["content"], answer, source_documents)
                    
                    # 存入历史
                    msg_data = {"role": "assistant", "content": answer}
                    if is_audio_success:
                        st.audio(audio_file, format="audio/mp3")
                        msg_data["audio_path"] = audio_file
                    st.session_state.messages.append(msg_data)
                    
                    # 追问建议
                    suggestions = get_suggestions(answer, llm_engine)
                    st.session_state.current_suggestions = suggestions
                    st.rerun()
                except Exception as e:
                    message_placeholder.markdown(f"Error: {e}")
        else:
            message_placeholder.markdown("API Error")

# 追问按钮
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if "current_suggestions" in st.session_state and st.session_state.current_suggestions:
        st.markdown("<h3 style='font-size: 1.1em; color: #5D4037; margin-top: 20px;'>🤔 您可能想问：</h3>", unsafe_allow_html=True)
        cols = st.columns(1) # 改成单列，像图1那样竖着排
        for i, question in enumerate(st.session_state.current_suggestions):
            if cols[0].button(question, key=f"sugg_{i}", use_container_width=True): # use_container_width 让按钮撑满
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.current_suggestions = []
                st.rerun()
