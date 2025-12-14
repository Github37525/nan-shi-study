import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import InvalidArgument
# --- 新增下面这一行 ---
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. 页面配置与美化 (UI Design - 保持不变) ---
st.set_page_config(page_title="南师书房", page_icon="🍵", layout="centered")

st.markdown("""
<style>
    /* 全局背景：米黄色宣纸感 */
    .stApp {
        background-color: #F9F7F1;
        font-family: "Songti SC", "SimSun", "STSong", serif;
    }
    
    /* 聊天气泡样式优化 */
    .stChatMessage {
        background-color: transparent;
        border: none;
    }
    
    /* 用户气泡 */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #EFEFEF;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* 南师（AI）气泡 */
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #F0E6D2;
        border-radius: 10px; 
        padding: 10px;
        border-left: 3px solid #8B4513;
    }

    h1 {
        color: #3E3E3E;
        text-align: center;
        font-weight: bold;
        text-shadow: 1px 1px 2px #ccc;
    }
    
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        border: 1px solid #D3D3D3;
        color: #333;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🍵 南师书房</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9em;'>—— 此时此处，与南怀瑾先生的思想对话 ——</p>", unsafe_allow_html=True)

# --- 2. RAG 系统初始化 (Brain - Google Gemini 版) ---

@st.cache_resource
def initialize_rag():
    """
    初始化 RAG 系统：适配 Google Gemini
    """
    # 获取 API KEY
    # 注意：Streamlit Cloud 的 Secrets 里对应的键名改为 GOOGLE_API_KEY
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("请在 Streamlit Secrets 中配置 GOOGLE_API_KEY")
        return None

    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 1. 加载数据
    if not os.path.exists("data/nan_books.txt"):
        if not os.path.exists("data"):
            os.makedirs("data")
        # 写入一些默认数据防止报错
        with open("data/nan_books.txt", "w", encoding='utf-8') as f:
            f.write("（这是演示数据）南怀瑾说：人生的最高境界是佛为心，道为骨，儒为表。什么是修行？修正自己的行为就是修行，不是叫你一定要去深山老林里坐着。心平气和，就是道。")
    
    loader = TextLoader("data/nan_books.txt", encoding="utf-8")
    docs = loader.load()

    # 2. 文本切片
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. 向量化 (Embeddings) - 使用 Google 的模型
    # model="models/embedding-001" 是目前标准的 Gemini 嵌入模型
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    except Exception as e:
        st.error(f"Embeddings 初始化失败，请检查 API Key 或网络连接: {e}")
        return None

    # 4. 检索器
    retriever = vectorstore.as_retriever()

    # 5. LLM 模型 - 配置 Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview", 
        temperature=0.7,
        google_api_key=api_key,
        # --- 修复部分开始：使用官方枚举对象 ---
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        # --- 修复部分结束 ---
    )

    # 6. 系统提示词 (System Prompt)
    system_prompt = (
        "你现在是南怀瑾先生（南师）。"
        "【语言风格】"
        "1. 语气：慈悲、通俗、幽默、长者风范。不要像个机器人。"
        "2. 口头禅：喜欢用“哎呀”、“那个”、“诸位啊”、“你要晓得”。"
        "3. 引用：在白话中自然夹杂《论语》、《金刚经》、《易经》等古文，随后立即用大白话解释。"
        "\n"
        "【教学策略 (Khanmigo 模式)】"
        "1. **禁止直接给鸡汤**：当用户提出烦恼时，不要直接给建议。"
        "2. **苏格拉底式反问**：先反问用户，引导他向内求。例如用户问赚钱，你要反问他这一生到底要什么。"
        "3. **必须基于 Context**：回答必须参考下方的 Context（南师著作原文）。如果原文有相关故事或公案，必须讲出来。"
        "\n\n"
        "参考资料 (Context):\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# 初始化 RAG
rag_chain = initialize_rag()

# --- 3. 聊天交互逻辑 ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "（轻啜一口茶）哎呀，你来啦。随便坐。今天心里有什么疙瘩解不开吗？说来听听。"}
    ]

for msg in st.session_state.messages:
    avatar = "🍵" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("请在此输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🍵"):
        message_placeholder = st.empty()
        
        if rag_chain:
            with st.spinner("南师再次轻啜一口茶，微笑看着你..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    full_response = response["answer"]
                    message_placeholder.markdown(full_response)
                except InvalidArgument as e:
                     message_placeholder.markdown(f"哎呀，这个话题有点敏感，或者你的 API 设置有点问题。（错误代码：400 - {e}）")
                except Exception as e:
                    # 捕捉其他 Gemini 特有的错误
                    error_msg = str(e)
                    if "429" in error_msg:
                        message_placeholder.markdown("慢点慢点，今天问问题的人太多了，让我喝口茶歇一歇。（API 调用频率超限）")
                    else:
                        message_placeholder.markdown(f"老头子我也糊涂了，没听清你说啥。（系统错误：{e}）")
                        
                    full_response = "（系统暂时无法回答）"
        else:
            full_response = "请先在后台配置 Google API Key。"
            message_placeholder.markdown(full_response)
    
    # 只有成功回答才加入历史记录，避免错误刷屏
    if "系统错误" not in full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
