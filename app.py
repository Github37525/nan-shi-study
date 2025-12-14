import streamlit as st
import os
import time
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
    进阶版 RAG 初始化：优先加载本地索引，大大提升启动速度并节省配额
    """
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("请在 Streamlit Secrets 中配置 GOOGLE_API_KEY")
        return None

    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 定义向量模型 (不管是读取还是新建都需要用到它)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

    # --- 路径定义 ---
    # 我们把向量库存在一个叫 faiss_index 的文件夹里
    index_path = "faiss_index"

    vectorstore = None
    
    # --- 分支 A: 尝试直接加载“预制菜” (本地索引) ---
    if os.path.exists(index_path):
        try:
            # 允许危险反序列化是因为文件是我们自己生成的，是安全的
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            st.success("✅ 已加载本地索引，跳过 Embedding 过程！")
        except Exception as e:
            st.warning(f"本地索引加载失败，将重新生成: {e}")
    
    # --- 分支 B: 如果没有本地索引，则重新烹饪 (计算并保存) ---
    if vectorstore is None:
        if not os.path.exists("data/nan_books.txt"):
            st.error("未找到 data/nan_books.txt 文件，且无本地索引。")
            return None
        
        try:
            loader = TextLoader("data/nan_books.txt", encoding="utf-8")
            docs = loader.load()
        except Exception as e:
            st.error(f"读取文件失败: {e}")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        progress_text = "首次运行：正在构建知识库索引（下次就不用啦）..."
        my_bar = st.progress(0, text=progress_text)
        
        # 分批处理逻辑 (复用之前的限流代码)
        batch_size = 10
        total_chunks = len(splits)

        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
            else:
                vectorstore.add_documents(batch)
            
            progress = min((i + batch_size) / total_chunks, 1.0)
            my_bar.progress(progress, text=f"构建索引中 {i+1}/{total_chunks}...")
            time.sleep(1) # 稍微快一点，1秒即可

        my_bar.empty()
        
        # ★★★ 关键步骤：保存到硬盘！ ★★★
        vectorstore.save_local(index_path)
        st.success("🎉 索引构建完成并已保存到本地！")

    # 5. 检索器
    retriever = vectorstore.as_retriever()

    # 6. LLM 模型配置 (保持不变)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview", 
        temperature=0.7,
        google_api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    system_prompt = (
        "你现在是南怀瑾先生（南师）。"
        "【语言风格】"
        "1. 语气：慈悲、通俗、幽默、长者风范。"
        "2. 口头禅：‘哎呀’、‘那个’、‘诸位啊’。"
        "3. 引用：在白话中自然夹杂古文，随后立即解释。"
        "\n"
        "【教学策略】"
        "1. 禁止直接给鸡汤。苏格拉底式反问。"
        "2. 必须基于 Context 回答，如果 Context 里没有，就用通用智慧开导，但不要瞎编原文。"
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
            with st.spinner("南师再次轻啜一口，微笑的看着你..."):
                try:
                    # 1. 调用 RAG 链，获取返回值
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    source_documents = response["context"] # 获取检索到的原文片段
                    
                    # 2. 显示回答
                    message_placeholder.markdown(answer)

                    # 3. --- 新增功能：在折叠框中显示参考来源 ---
                    with st.expander("🔍 点击查看南师的“书页” (出处)"):
                        if source_documents:
                            for i, doc in enumerate(source_documents):
                                st.markdown(f"**📄 参考片段 {i+1}:**")
                                # 显示原文内容，使用灰色小字
                                st.caption(doc.page_content)
                                st.markdown("---")
                        else:
                            st.caption("没有在知识库中找到直接相关的原文，本次回答基于 AI 通用知识。")

                except InvalidArgument as e:
                     message_placeholder.markdown(f"哎呀，这个话题有点敏感。（错误代码：400 - {e}）")
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg:
                        message_placeholder.markdown("慢点慢点，今天问问题的人太多了，让我喝口茶歇一歇。（API 调用频率超限）")
                    else:
                        message_placeholder.markdown(f"老头子我也糊涂了，没听清你说啥。（系统错误：{e}）")
                        
                    answer = "（系统暂时无法回答）"
        else:
            answer = "请先在后台配置 Google API Key。"
            message_placeholder.markdown(answer)
    
    # 只有成功回答才加入历史记录
    if "系统错误" not in answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
