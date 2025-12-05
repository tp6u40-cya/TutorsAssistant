import streamlit as st
# 確保引用了 retrieve_docs (用於 RAG) 和 get_exam_generator_chain (用於出題)
from app.chains import get_exam_generator_chain, retrieve_docs

# 設定網頁標題與排版
st.set_page_config(page_title="AI 國文家教教材生成器", page_icon="🎓", layout="wide")

st.title("🎓 國文家教教材生成器")
st.markdown("### 基於 15 篇核心古文與學測課綱 (RAG 加強版)")

# --- 定義 15 篇古文清單 ---
CLASSICAL_TEXTS = [
    "先秦 - 燭之武退秦師",
    "先秦 - 大同與小康",
    "漢魏六朝 - 諫逐客書",
    "漢魏六朝 - 鴻門宴",
    "漢魏六朝 - 桃花源記",
    "唐宋 - 出師表",
    "唐宋 - 師說",
    "唐宋 - 虯髯客傳",
    "唐宋 - 赤壁賦",
    "唐宋 - 晚遊六橋待月記",
    "明清 - 項脊軒志",
    "明清 - 勞山道士",
    "古典臺灣 - 勸和論",
    "古典臺灣 - 鹿港乘桴記",
    "古典臺灣 - 畫菊自序"
]

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 試卷設定")
    
    # 1. 課文選擇 (多選)
    selected_texts = st.multiselect(
        "📚 請選擇測驗範圍 (可多選)",
        options=CLASSICAL_TEXTS,
        default=["唐宋 - 赤壁賦"]
    )
    
    st.divider()
    st.subheader("📊 題數與難度配置")
    
    # 2. 難度題數設定
    col1, col2, col3 = st.columns(3)
    with col1:
        num_simple = st.number_input("簡單", min_value=0, value=2, help="基礎字音字形、字義")
    with col2:
        num_medium = st.number_input("中等", min_value=0, value=2, help="文意理解、分析")
    with col3:
        num_hard = st.number_input("困難", min_value=0, value=1, help="比較閱讀、素養題")
    
    total_questions = num_simple + num_medium + num_hard
    st.info(f"📝 總題數：{total_questions} 題")
    
    st.divider()
    st.caption("💡 提示：選擇多篇課文時，AI 會嘗試設計跨文本比較的題目。")

# --- 輔助函式：美化顯示閱讀文本 ---
def render_text_content(content):
    if isinstance(content, dict):
        if "title" in content:
            st.markdown(f"#### 📑 {content['title']}")
        if "content" in content and isinstance(content["content"], list):
            for item in content["content"]:
                if "source" in item:
                    st.markdown(f"**【{item['source']}】**")
                if "text" in item:
                    st.markdown(f"> {item['text']}")
                st.markdown("---")
        else:
            st.json(content, expanded=True)
    elif isinstance(content, list):
        for item in content:
            st.markdown(f"> {item}")
            st.markdown("---")
    else:
        st.markdown(content)

# --- 主程式邏輯 ---
if st.button("🚀 開始生成試卷", type="primary"):
    # 檢查輸入
    if not selected_texts:
        st.warning("請至少選擇一篇課文！")
    elif total_questions == 0:
        st.warning("總題數不能為 0！")
    else:
        # 初始化變數，避免 'not defined' 錯誤
        response = {}
        questions = []
        
        try:
            # 取得 AI 邏輯鏈
            chain = get_exam_generator_chain()
            
            # 1. 【RAG 關鍵步驟】先去資料庫把原文抓出來
            with st.spinner("🔍 正在知識庫檢索課文原文..."):
                retrieved_context = retrieve_docs(selected_texts)
            
            # 2. 組裝指令
            text_scope = "、".join(selected_texts)
            prompt_request = (
                f"請根據以下【參考課文原文】來設計試題：\n"
                f"{retrieved_context}\n\n"
                f"--------------------------------\n"
                f"出題需求：\n"
                f"範圍：{text_scope}。\n"
                f"總題數：{total_questions} 題（簡單{num_simple}、中等{num_medium}、困難{num_hard}）。\n"
                f"請嚴格根據上述提供的原文內容出題，不要憑空捏造。"
            )

            # 3. 呼叫 AI 生成題目
            with st.spinner(f"🤖 AI 正在根據 {text_scope} 設計題目..."):
                response = chain.invoke({
                    "user_request": prompt_request,
                    "format_instructions": "請回傳完整的 JSON 格式。"
                })
                
                # 安全地取得題目列表
                questions = response.get("questions", [])

                st.toast("✅ 生成完成！", icon="🎉")
                
                # --- 區塊 1：閱讀文本 ---
                st.subheader("📖 測驗範圍 (來自 RAG 知識庫)")
                # 優先顯示 AI 回傳的 main_text，如果沒有則顯示 RAG 抓到的原文
                render_text_content(response.get("main_text", retrieved_context))
                st.divider()

                # --- 區塊 2：試題區 ---
                st.subheader(f"✍️ 試題 (共 {len(questions)} 題)")
                
                if not questions:
                    st.warning("⚠️ AI 沒有生成題目，可能是 API 回應不完整，請再試一次。")
                else:
                    # 顯示題目
                    for index, q in enumerate(questions):
                        # 根據難度標示顏色
                        diff = q.get('difficulty', '未知')
                        badge_color = "green" if "簡單" in diff else "orange" if "中等" in diff else "red"
                        
                        with st.container():
                            col_a, col_b = st.columns([1, 6])
                            with col_a:
                                st.markdown(f":{badge_color}[{diff}]")
                                st.caption(q.get('type', ''))
                            with col_b:
                                st.markdown(f"**{index+1}. {q.get('question_text', '')}**")
                            
                            if q.get('options'):
                                for opt in q['options']:
                                    st.markdown(f"- {opt}")
                            
                            with st.expander(f"👁️ 查看解答"):
                                st.markdown(f"**💡 參考答案**：{q.get('correct_answer', '')}")
                                st.info(f"**📝 解析**：\n\n{q.get('explanation', '')}")
                            
                            st.markdown("---")

                # 開發者除錯區
                with st.expander("🛠️ 開發者模式：查看原始 JSON"):
                    st.json(response)

        except Exception as e:
            # 這裡會捕捉所有錯誤，避免紅字刷屏
            st.error(f"❌ 發生錯誤：{str(e)}")
            st.markdown("---")
            st.markdown("**除錯建議：**")
            st.markdown("1. 請檢查終端機是否有 `API Key` 相關錯誤。")
            st.markdown("2. 若出現 `Insufficient credits`，代表 API 額度不足。")
            st.markdown("3. 請確認 `.md` 檔案內容是否正確讀取。")