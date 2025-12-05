import os
import json
import re
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from app.prompts import few_shot_examples, system_prompt_content

# --- 載入環境變數 ---
env_file = find_dotenv()
if env_file: load_dotenv(env_file)

# --- 1. 新版資料結構定義 (支援題組) ---

class SubQuestion(BaseModel):
    id: str = Field(description="子題編號，例如 (1), (2)")
    question_text: str = Field(description="題目敘述")
    options: list[str] = Field(description="選項 (A, B, C, D)")
    correct_answer: str = Field(description="正確答案")
    explanation: str = Field(description="解析")

class QuestionBlock(BaseModel):
    type: str = Field(description="類型：單題 / 題組 / 混合題組")
    difficulty: str = Field(description="難度")
    article_content: str = Field(description="閱讀測驗的文章內容 (若為單題則留空，若為題組請填入甲、乙等引文)")
    questions: list[SubQuestion] = Field(description="此區塊包含的題目列表")

class ExamPaper(BaseModel):
    main_scope_text: str = Field(description="測驗範圍說明")
    question_blocks: list[QuestionBlock] = Field(description="試題區塊列表")

# --- 2. 清洗函式 ---
def clean_and_parse_json(ai_response_text):
    try:
        text = ai_response_text.strip()
        if "```json" in text: text = text.split("```json")[1].split("```")[0]
        elif "```" in text: text = text.split("```")[1].split("```")[0]
        
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx])
        else:
            raise ValueError("無效的 JSON 結構")
    except Exception as e:
        print(f"❌ JSON 解析失敗: {e}")
        return {"main_scope_text": "解析錯誤", "question_blocks": []}

# --- 3. 檢索函式 ---
def retrieve_docs(selected_texts):
    db_path = "./data/chroma_db"
    if not os.path.exists(db_path): return "（警告：尚未建立 RAG 資料庫）"

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    
    retrieved_content = ""
    for item in selected_texts:
        title_keyword = item.split("-")[1].strip() if "-" in item else item
        print(f"🔍 [RAG] 搜尋：{title_keyword}")
        
        try:
            results = vectorstore.similarity_search(query=title_keyword, k=2, filter={"title": title_keyword})
            if results:
                for doc in results:
                    retrieved_content += f"\n\n--- 選文：{doc.metadata.get('title', '未知')} ---\n{doc.page_content}"
            else:
                retrieved_content += f"\n\n（未找到 {title_keyword} 的原文）"
        except Exception as e:
            retrieved_content += f"\n\n（搜尋錯誤：{e}）"

    return retrieved_content

# --- 4. 建立 Chain ---
def get_exam_generator_chain():
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    model_name = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not os.getenv("OPENROUTER_API_KEY"): base_url = None 

    if not api_key: raise ValueError("❌ 錯誤：找不到 API Key")

    llm = ChatOpenAI(
        model=model_name if model_name else "gpt-4o",
        api_key=api_key,
        base_url=base_url,
        temperature=0.7
    )
    
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{instruction}\n\n參考文本：\n{context}"), ("ai", "{output_json}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=few_shot_examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_content),
            few_shot_prompt,
            ("human", "{user_request}\n\n{format_instructions}"),
        ]
    )

    chain = final_prompt | llm | StrOutputParser() | clean_and_parse_json
    return chain