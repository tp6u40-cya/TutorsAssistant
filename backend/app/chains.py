import os
import sys
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from app.prompts import few_shot_examples, system_prompt_content

# --- 強制尋找並載入 .env ---
# 這樣寫可以避免 streamlit 有時候找不到檔案的問題
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    print(f"✅ 成功載入環境變數檔案: {env_file}")
else:
    print("❌ 警告：找不到 .env 檔案！請確認它在專案根目錄。")

# --- 除錯：檢查是否有讀到金鑰 ---
api_key_check = os.getenv("OPENROUTER_API_KEY")
if not api_key_check:
    # 嘗試讀取 OpenAI
    api_key_check = os.getenv("OPENAI_API_KEY")
    if not api_key_check:
        print("❌ 嚴重錯誤：程式碼讀取不到任何 API Key！")
    else:
        print(f"ℹ️ 使用 OpenAI API Key (前幾碼: {api_key_check[:5]}...)")
else:
    print(f"ℹ️ 使用 OpenRouter API Key (前幾碼: {api_key_check[:5]}...)")


# 1. 資料結構定義
class Question(BaseModel):
    id: str = Field(description="題號")
    difficulty: str = Field(description="難度")
    type: str = Field(description="題型")
    question_text: str = Field(description="題目敘述")
    options: list[str] = Field(description="選項")
    correct_answer: str = Field(description="答案")
    explanation: str = Field(description="解析")

class ExamPaper(BaseModel):
    main_text: str = Field(description="閱讀測驗的文本內容")
    questions: list[Question] = Field(description="題目列表")

# 2. 檢索函式 (使用本地模型)
def retrieve_docs(selected_texts):
    db_path = "./data/chroma_db"
    
    if not os.path.exists(db_path):
        return "（警告：尚未建立 RAG 資料庫，請先執行 rag_builder.py）"

    # 使用 HuggingFace 本地模型 (免費)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=db_path, 
        embedding_function=embedding_model
    )
    
    retrieved_content = ""
    
    for item in selected_texts:
        if "-" in item:
            title_keyword = item.split("-")[1].strip()
        else:
            title_keyword = item

        print(f"🔍 [RAG] 正在資料庫搜尋：{title_keyword}")
        
        try:
            results = vectorstore.similarity_search(
                query=title_keyword,
                k=2,
                filter={"title": title_keyword}
            )
            
            if results:
                for doc in results:
                    retrieved_content += f"\n\n--- 選文：{doc.metadata.get('title', '未知')} ---\n{doc.page_content}"
            else:
                retrieved_content += f"\n\n（未找到 {title_keyword} 的原文）"
        except Exception as e:
            print(f"⚠️ 搜尋時發生錯誤: {e}")
            retrieved_content += f"\n\n（搜尋 {title_keyword} 時發生錯誤）"

    return retrieved_content

# 3. 建立 Chain 的函式
def get_exam_generator_chain():
    # 讀取環境變數
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    model_name = os.getenv("OPENROUTER_MODEL")

    # 如果沒設定 OpenRouter，嘗試讀取 OpenAI
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        # 如果切換回 OpenAI，要清空 base_url 避免錯誤
        if not os.getenv("OPENROUTER_API_KEY"): 
            base_url = None 

    if not api_key:
        raise ValueError("❌ 錯誤：找不到 API Key，請檢查 .env 檔案設定！")

    # 初始化模型
    llm = ChatOpenAI(
        model=model_name if model_name else "gpt-4o",
        api_key=api_key,
        base_url=base_url,
        temperature=0.7
    )
    
    parser = JsonOutputParser(pydantic_object=ExamPaper)

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

    chain = final_prompt | llm | parser
    return chain