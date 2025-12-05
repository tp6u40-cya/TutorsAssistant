import streamlit as st
import pandas as pd
import json
from app.chains import get_exam_generator_chain, retrieve_docs

# 設定網頁標題與排版
st.set_page_config(page_title="AI 國文家教教材生成器", page_icon="🎓", layout="wide")

st.title("🎓 國文家教教材生成器")
st.markdown("### 基於 15 篇核心古文與學測課綱 (題組加強版)")

# --- 定義 15 篇古文清單 ---
CLASSICAL_TEXTS = [
    "先秦 - 燭之武退秦師", "先秦 - 大同與小康",
    "漢魏六朝 - 諫逐客書", "漢魏六朝 - 鴻門宴", "漢魏六朝 - 桃花源記",
    "唐宋 - 出師表", "唐宋 - 師說", "唐宋 - 虯髯客傳", "唐宋 - 赤壁賦", "唐宋 - 晚遊六橋待月記",
    "明清 - 項脊軒志", "明清 - 勞山道士",
    "古典臺灣 - 勸和論", "古典臺灣 - 鹿港乘桴記", "古典臺灣 - 畫菊自序"
]

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 試卷設定")
    
    # 1. 課文選擇
    selected_texts = st.multiselect(
        "📚 請選擇測驗範圍 (可多選)",
        options=CLASSICAL_TEXTS,
        default=["唐宋 - 赤壁賦"]
    )
    
    st.divider()
    st.subheader("📊 題型與難度配分表")
    st.caption("請直接在表格中輸入題數：")

    # 2. 建立題型配置表格
    question_types = ["單選題", "多選題", "題組(閱讀)", "混合題", "素養題", "作文/問答"]
    difficulties = ["簡單", "中等", "困難"]
    
    default_data = {diff: [0]*len(question_types) for diff in difficulties}
    df = pd.DataFrame(default_data, index=question_types)
    
    # 預設值
    df.at["單選題", "簡單"] = 2
    df.at["單選題", "中等"] = 2
    
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        column_config={
            "簡單": st.column_config.NumberColumn(min_value=0, max_value=10, step=1),
            "中等": st.column_config.NumberColumn(min_value=0, max_value=10, step=1),
            "困難": st.column_config.NumberColumn(min_value=0, max_value=10, step=1),
        }
    )

    total_questions = edited_df.values.sum()
    st.info(f"📝 預計生成區塊數：{total_questions}")

    st.divider()
    
    # 3. 其他客製化要求
    st.subheader("✨ 其他出題要求")
    custom_requirements = st.text_area(
        "請輸入額外指令 (選填)",
        placeholder="例如：希望出一題與赤壁賦有關的作文、希望題目多考修辭學、希望題目結合環保議題...",
        height=100
    )

# --- 輔助函式：強健的內容渲染器 ---
def render_content_safe(content):
    """
    能處理字串、字典、列表的通用顯示函式，避免報錯
    """
    if content is None:
        return

    # 如果是字串
    if isinstance(content, str):
        if content.strip():
            st.markdown(content)
            
    # 如果是字典 (AI 雞婆把內容結構化時)
    elif isinstance(content, dict):
        # 嘗試抓取常見的 key
        text = content.get("text") or content.get("content") or content.get("body")
        if text:
            st.markdown(str(text))
        else:
            st.json(content, expanded=False)
            
    # 如果是列表
    elif isinstance(content, list):
        for item in content:
            st.markdown(f"> {item}")
    
    # 其他情況
    else:
        st.write(content)

def render_rag_text(content):
    """專門處理 RAG 回傳的內容"""
    if isinstance(content, str):
        st.markdown(content)
    else:
        render_content_safe(content)

def parse_requirements_to_string(df):
    req_list = []
    for q_type in df.index:
        for diff in df.columns:
            count = df.at[q_type, diff]
            if count > 0:
                unit = "組" if "題組" in q_type else "題"
                req_list.append(f"「{diff}」的「{q_type}」：{count} {unit}")
    return "；".join(req_list)

# --- 主程式邏輯 ---
if st.button("🚀 開始生成試卷", type="primary"):
    if not selected_texts:
        st.warning("請至少選擇一篇課文！")
    elif total_questions == 0 and not custom_requirements.strip():
        st.warning("請設定題數或輸入出題要求！")
    else:
        try:
            chain = get_exam_generator_chain()
            
            # 1. RAG 檢索
            with st.spinner("🔍 正在知識庫檢索課文原文..."):
                retrieved_context = retrieve_docs(selected_texts)
            
            # 2. 組裝指令
            text_scope = "、".join(selected_texts)
            structure_prompt = parse_requirements_to_string(edited_df)
            
            prompt_request = (
                f"請根據以下【參考課文原文】來設計試題：\n"
                f"{retrieved_context}\n\n"
                f"--------------------------------\n"
                f"【出題需求表】：\n"
                f"1. 測驗範圍：{text_scope}\n"
                f"2. 題目結構要求：{structure_prompt}\n"
                f"3. 使用者額外指定要求：{custom_requirements if custom_requirements else '無'}\n\n"
                f"【重要規則】：\n"
                f"- 若有「題組」，請生成一篇閱讀文章(甲乙文或延伸閱讀)，並在該文章下出 2-3 題子題。\n"
                f"- 若使用者要求「作文」，請將其類型標記為「作文」，並在解析欄位提供寫作引導。\n"
                f"- 請務必遵守 JSON 格式中的 QuestionBlock 結構。"
            )

            # 3. 呼叫 AI
            with st.spinner(f"🤖 AI 正在根據需求設計題目..."):
                response = chain.invoke({
                    "user_request": prompt_request,
                    "format_instructions": "請回傳完整的 JSON 格式。"
                })
                
                blocks = response.get("question_blocks", [])
                st.toast("✅ 生成完成！", icon="🎉")
                
                # --- 顯示區域 ---
                st.subheader("📖 測驗範圍 (來自 RAG 知識庫)")
                with st.expander("點擊查看 RAG 抓取的原文"):
                    render_rag_text(retrieved_context)
                
                st.divider()
                st.header("✍️ 試卷預覽")

                if not blocks:
                    st.warning("⚠️ AI 回傳內容為空，請檢查 API 額度或重試。")
                else:
                    for i, block in enumerate(blocks):
                        diff = block.get('difficulty', '一般')
                        badge_color = "green" if "簡單" in diff else "orange" if "中等" in diff else "red"
                        
                        st.markdown(f"### 第 {i+1} 部分：{block.get('type','綜合')} (:{badge_color}[{diff}])")
                        
                        # --- 修正點：使用安全的渲染函式 ---
                        article = block.get('article_content')
                        if article:
                            with st.chat_message("assistant"):
                                st.markdown("#### 閱讀材料")
                                render_content_safe(article)
                        
                        # 顯示子題
                        questions = block.get('questions', [])
                        for q in questions:
                            with st.container():
                                st.markdown(f"**{q.get('id', '●')} {q.get('question_text', '')}**")
                                
                                options = q.get('options', [])
                                if options and len(options) > 0:
                                    for opt in options:
                                        st.text(opt) 
                                
                                with st.expander("💡 解答與解析"):
                                    st.markdown(f"**答案**：{q.get('correct_answer')}")
                                    st.info(q.get('explanation'))
                                
                                st.markdown("---")

                with st.expander("🛠️ 開發者模式：查看原始 JSON"):
                    st.json(response)

        except Exception as e:
            st.error(f"❌ 發生錯誤：{str(e)}")
            if "401" in str(e):
                st.error("⚠️ 錯誤代碼 401：API Key 無效或未找到用戶。請檢查 .env 檔案中的 Key 是否正確。")
            elif "402" in str(e):
                st.error("⚠️ 錯誤代碼 402：額度不足。請檢查 OpenRouter 帳戶餘額。")