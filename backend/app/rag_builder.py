import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 改用免費的 HuggingFace 本地模型
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def build_database():
    base_path = "./data/knowledge_base"
    db_path = "./data/chroma_db"

    documents = []
    
    if not os.path.exists(base_path):
        print(f"❌ 錯誤：找不到路徑 {base_path}")
        return

    print("🚀 開始掃描知識庫...")

    # 1. 讀取檔案
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".md") or file.endswith(".txt"):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                file_name = os.path.splitext(file)[0]

                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["era"] = folder_name
                        doc.metadata["title"] = file_name
                        doc.metadata["source"] = file_name
                    
                    documents.extend(docs)
                    print(f"   ✅ 已讀取：[{folder_name}] {file_name}")
                    
                except Exception as e:
                    print(f"   ⚠️ 讀取失敗 {file_name}: {e}")

    if not documents:
        print("❌ 未發現任何文件。")
        return

    # 2. 切割文字
    print(f"📦 正在切割 {len(documents)} 份文件...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 本地模型建議切小一點，效果較好
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！"]
    )
    splits = text_splitter.split_documents(documents)

    # 3. 清除舊資料庫 (非常重要！因為模型換了，向量維度不同，必須重蓋)
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print("🧹 已清除舊的資料庫 (因為更換模型)")
        except Exception as e:
            print(f"⚠️ 無法刪除舊資料庫，請手動刪除 {db_path} 資料夾後再試: {e}")
            return

    # 4. 建立向量資料庫 (使用本地 CPU 模型)
    print(f"💾 正在使用本地模型 (HuggingFace) 建立索引... (第一次執行會下載模型，請稍候)")
    
    # 使用免費、輕量級的 all-MiniLM-L6-v2 模型
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=db_path
    )
    print(f"🎉 成功！RAG 知識庫已建立於 {db_path}")

if __name__ == "__main__":
    build_database()