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

# --- 强制代理 (如果你的语音生成失败，请取消下面两行的注释并修改端口) ---
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# --- 1. 页面配置 ---
st.set_page_config(page_title="南师书房", page_icon="🍵", layout="wide")

NAN_QUOTES = [
    "世上本无事，庸人自扰之。", "应无所住，而生其心。", "能控制早晨的人，就能控制人生。",
    "静坐修道与长生不老，都在这个“静”字。", "人生最高境界是：佛为心，道为骨，儒为表，大度看世界。",
    "功成、名遂、身退，天之道。", "知止而后有定，定而后能静，静而后能安。",
    "真正的修行，不离日常生活。", "心平气和，就是道。", "大丈夫处其厚，不居其薄；处其实，不居其华。",
    "英雄到老皆归佛，宿将还山不论兵。", "多言数穷，不如守中。"
]

st.markdown("""
<style>
    .stApp {
        background-color: #F9F7F1;
        background-image: url("https://www.transparenttextures.com/patterns/rice-paper-2.png");
        font-family: "楷体", "KaiTi", "Songti SC", serif;
    }
    [data-testid="stChatMessage"]:nth-child(odd) { background-color: rgba(239, 239, 239, 0.7); border-radius: 15px; padding: 15px; border: 1px solid #D3D3D3; }
    [data-testid="stChatMessage"]:nth-child(even) { background-color: rgba(240, 230, 210, 0.8); border-radius: 15px; padding: 15px; border-left: 4px solid #8B4513; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    h1 { color: #4A3B2A; text-align: center; font-weight: bold; letter-spacing: 2px; padding-bottom: 10px; border-bottom: 2px solid #8B4513; display: inline-block; }
    .title-container { text-align: center; margin-bottom: 30px; }
    [data-testid="stSidebar"] { background-color: #F4EFE5; border-right: 1px solid #D8CFC4; }
    .quote-card { background-color: #FDFBF7; border: 2px solid #8B4513; border-radius: 8px; padding: 20px; text-align: center; font-size: 1.3em; font-weight: bold; color: #5C4033; box-shadow: 3px 3px 8px rgba(139, 69, 19, 0.2); position: relative; margin-bottom: 20px; }
    .quote-card::before, .quote-card::after { content: '•'; color: #8B4513; font-size: 2em; position: absolute; top: -15px; }
    .quote-card::before { left: 10px; } .quote-card::after { right: 10px; }
    .stButton button { background-color: #F0E6D2; border: 1px solid #8B4513; color: #5C4033; }
    .stButton button:hover { background-color: #E6D8B8; border-color: #5C4033; color: #3E2B22; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-container'><h1>🍵 南师书房</h1></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #777; font-size: 1em; font-style: italic;'>—— 此时此处，调息静心，与南师对话 ——</p>", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("## 🎍 书房一隅")
    st.markdown("### 📜 今日参悟")
    if "daily_quote" not in st.session_state:
        st.session_state.daily_quote = random.choice(NAN_QUOTES)
    st.markdown(f"<div class='quote-card'>“{st.session_state.daily_quote}”</div><p style='text-align: right; color: #999; font-size: 0.9em;'>—— 南怀瑾</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎵 伴读琴韵")
    bgm_path = "assets/bgm.mp3"
    audio_source = bgm_path if os.path.exists(bgm_path) else "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
    st.audio(audio_source, format="audio/mp3", start_time=0)
    st.caption("💡 建议点击播放后，将音量调至轻柔。")

# --- 功能函数定义区 ---

# 1. 语音生成函数
async def generate_speech(text, output_file="speech_output.mp3"):
    """使用 Edge TTS 生成语音，包含错误处理"""
    VOICE = "zh-CN-YunzeNeural"
    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_file)
        return True 
    except Exception as e:
        print(f"⚠️ 语音生成失败: {e}")
        return False

# 2. 日志记录函数
def save_to_logs(user_question, ai_answer, sources):
    """将对话记录写入 Google Sheets"""
    try:
        if "gcp_service_account" not in st.secrets:
            return # 未配置则静默跳过

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # 尝试打开表格，如果找不到可能会报错，所以加 try
        sheet = client.open("南师书房日志").sheet1
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        source_str = "; ".join([f"{doc.metadata.get('source')}·{doc.metadata.get('chapter')}" for doc in sources]) if sources else "无引用"
        
        sheet.append_row([timestamp, user_question, ai_answer, source_str])
        print("✅ 日志已记录")
    except Exception as e:
        print(f"❌ 日志记录失败: {e}")

# 3. 追问生成函数
def get_suggestions(answer_text, llm_engine):
    if not llm_engine: return []
    try:
        prompt = f"基于回答：'{answer_text[:500]}...'，生成3个简短追问。只返回问题，每行一个。"
        res = llm_engine.invoke(prompt)
        return [q.strip() for q in res.content.split('\n') if q.strip()][:3]
    except: return []

# --- RAG 系统初始化 ---

@st.cache_resource
def initialize_rag():
    if "GOOGLE_API_KEY" not in st.secrets: st.error("请配置 API Key"); return None
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 定义 LLM (注意这里修正了模型名称)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview", 
        temperature=0.7, 
        google_api_key=api_key,
        safety_settings={HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    index_path = "faiss_index"
    
    vectorstore = None
    if os.path.exists(index_path):
        try: vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True); st.sidebar.success("✅ 知识库已加载")
        except: pass
    
    if vectorstore is None:
        json_path = "data/nan_books.json"
        if not os.path.exists(json_path): st.error("数据缺失"); return None
        docs = []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                docs.append(Document(page_content=item.get("text", ""), metadata={"source": item.get("source", ""), "chapter": item.get("chapter", "")}))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever()
    
    # 历史感知检索器
    context_system_prompt = "给定对话历史和最新提问，将其改写为独立问题。不要回答，只改写。"
    context_prompt = ChatPromptTemplate.from_messages([("system", context_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    history_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # 问答链
    qa_system_prompt = (
        "你是南怀瑾（南师）。语气慈悲、通俗、幽默。苏格拉底式教学。"
        "必须基于 Context 回答，Context 含书名章节可引用。"
        "\n\n参考资料 (Context):\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

    return rag_chain, llm

rag_setup = initialize_rag()
if rag_setup: rag_chain, llm_engine = rag_setup
else: rag_chain, llm_engine = None, None

# --- 聊天交互逻辑 ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "哎呀，随便坐。今天心里有什么放不下的吗？"}]

for msg in st.session_state.messages:
    avatar = "🍵" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "audio_path" in msg and os.path.exists(msg["audio_path"]):
             st.audio(msg["audio_path"], format="audio/mp3")

user_input = st.chat_input("请在此输入您与南师的对话...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"): st.markdown(user_input)

    with st.chat_message("assistant", avatar="🍵"):
        message_placeholder = st.empty()
        if rag_chain:
            with st.spinner("南师正在沉思..."):
                try:
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user": chat_history.append(HumanMessage(content=msg["content"]))
                        else: chat_history.append(AIMessage(content=msg["content"]))
                    
                    response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
                    answer = response["answer"]
                    source_documents = response["context"]
                    
                    message_placeholder.markdown(answer)

                    with st.expander("🔍 点击查看出处"):
                        if source_documents:
                            for i, doc in enumerate(source_documents):
                                book = doc.metadata.get("source", "未知")
                                chap = doc.metadata.get("chapter", "")
                                st.markdown(f"**📖 {book} · {chap}**"); st.caption(doc.page_content); st.markdown("---")
                        else: st.caption("通用智慧回答，无直接引用。")
                    
                    # 生成语音
                    audio_file = f"speech_{int(time.time())}.mp3"
                    is_audio_success = asyncio.run(generate_speech(answer[:300], audio_file))

                    # 记录日志
                    save_to_logs(user_input, answer, source_documents)
                    
                    # 存储历史
                    msg_data = {"role": "assistant", "content": answer}
                    if is_audio_success:
                        st.audio(audio_file, format="audio/mp3")
                        msg_data["audio_path"] = audio_file
                    
                    st.session_state.messages.append(msg_data)
                    
                    # 生成追问
                    suggestions = get_suggestions(answer, llm_engine)
                    st.session_state.current_suggestions = suggestions
                    
                except Exception as e:
                    message_placeholder.markdown(f"老头子糊涂了（{e}）")
        else:
            message_placeholder.markdown("API 未连接")

# 追问按钮
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if "current_suggestions" in st.session_state and st.session_state.current_suggestions:
        st.markdown("### 🤔 您可能想问：")
        cols = st.columns(3)
        for i, question in enumerate(st.session_state.current_suggestions):
            if cols[i].button(question, key=f"sugg_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.current_suggestions = []
                st.rerun()
