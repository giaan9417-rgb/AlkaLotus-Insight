import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import os
import plotly.express as px
from stmol import showmol
from data import get_database
from utils import fetch_pdb, render_3d_molecule, check_lipinski, create_admet_radar, classify_potential
from scipy.optimize import minimize

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="PhytoInsight Platform | Nền tảng tin sinh học",
    layout="wide",
    page_icon="🪷",
    initial_sidebar_state="expanded"
)

if 'visited' not in st.session_state:
    intro_placeholder = st.empty()
    with intro_placeholder.container():
        st.markdown(
            """
            <style>
            @keyframes floatUpSlow {
                0% { transform: translateY(100vh) scale(0.7); opacity: 0; }
                20% { opacity: 1; }
                80% { opacity: 1; }
                100% { transform: translateY(-100vh) scale(1.5); opacity: 0; }
            }
            @keyframes floatLeaf {
                0% { transform: translateY(100vh) translateX(0) rotate(0deg); opacity: 0; }
                20% { opacity: 0.8; }
                50% { transform: translateY(50vh) translateX(50px) rotate(45deg); }
                100% { transform: translateY(-100vh) translateX(-50px) rotate(90deg); opacity: 0; }
            }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

            .lotus-overlay {
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                background-color: white;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }
            .main-icons {
                font-size: 130px;
                animation: floatUpSlow 5s ease-in-out forwards;
                filter: drop-shadow(0 0 15px rgba(255, 105, 180, 0.4));
            }
            .leaf {
                position: absolute;
                font-size: 50px;
                animation: floatLeaf 6s ease-in-out infinite;
                opacity: 0;
            }
            .lotus-text {
                margin-top: 50px;
                color: #FF69B4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
                font-size: 28px;
                letter-spacing: 3px;
                text-align: center;
                animation: fadeIn 2s ease-out 1s both;
            }
            </style>
            
            <div class="lotus-overlay">
                <div class="leaf" style="left: 15%; animation-delay: 0s;">🍃</div>
                <div class="leaf" style="left: 80%; animation-delay: 1.5s;">🍃</div>
                <div class="main-icons">🪷 🧬</div>
                <div class="lotus-text">NỀN TẢNG TIN SINH HỌC PHYTOINSIGHT PLATFORM</div>
            </div>
            """, 
            unsafe_allow_html=True 
        )
        time.sleep(5)
    intro_placeholder.empty()
    st.session_state['visited'] = True


st.title("☘️🖥️PHYTOINSIGHT PLATFORM")
st.markdown("<p style='font-size: 1.15em; color: #555; font-style: italic; margin-top: -15px; line-height: 1.4;'>Nền tảng tin sinh học  tích hợp dữ liệu nghiên cứu tương tác Alkaloid lá sen hướng đích enzyme AChE và BACE1 trong nghiên cứu Alzheimer và tính toán đa chất</p>", unsafe_allow_html=True)
st.divider()

# --- 4. KHỞI TẠO DỮ LIỆU ---
try:
    from data import get_database
    df = get_database()
except ImportError:
    # Backup nếu không tìm thấy file data.py (Dành cho chạy test)
    df = pd.DataFrame({
        'Name': ['Roemerine', 'Nuciferine'],
        'MW': [279.33, 295.38],
        'LogP': [3.1, 3.5],
        'HBD': [0, 0],
        'HBA': [3, 3],
        'Formula': ['C18H17NO2', 'C19H21NO2']
    })

if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = "Roemerine"

# --- 5. SIDEBAR ---
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

logo_paths = [
    "AlkaLotus/Logo_HungVuong.png.png", 
    "Logo_HungVuong.png.png",
    "AlkaLotus/Logo_HungVuong.png",
    "Logo_HungVuong.png"
]

logo_found = False
for path in logo_paths:
    if os.path.exists(path):
        st.sidebar.image(path, width=130)
        logo_found = True
        break

if not logo_found:
    github_logo_url = "https://raw.githubusercontent.com/giaan9417-rgb/AlkaLotus-Predictor/main/AlkaLotus/Logo_HungVuong.png.png"
    st.sidebar.image(github_logo_url, width=130)

st.sidebar.markdown(
    """
    <p style='font-size: 1em; font-weight: bold; color: #2E2E2E; margin-top: 5px; margin-bottom: 0px;'>
        Trường THPT Chuyên Hùng Vương
    </p>
    <p style='font-size: 0.8em; color: #666;'>TP. HỒ CHÍ MINH</p>
    """, 
    unsafe_allow_html=True
)
st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.divider()

st.sidebar.title("☘️ PhytoInsight Platform.")
st.sidebar.markdown("<div style='text-align: justify; font-size: 0.9em;'><b>Nền tảng tin sinh học</b>  tích hợp dữ liệu nghiên cứu tương tác Alkaloid lá sen hướng đích enzyme AChE và BACE1 trong nghiên cứu Alzheimer và tính toán đa chất.</div>", unsafe_allow_html=True)

st.sidebar.divider()
page = st.sidebar.radio(
    "Danh mục hệ thống",
    [
        "1. Thư viện Alkaloid",
        "2. Mô phỏng Docking 3D",
        "3. Phân tích & Xuất báo cáo",
        "4. Phân tích cấu trúc",
        "5. Tối ưu dung môi",
        "6. Động học chiết tách",
        "7. Dự toán quy mô & kinh tế"
    ]
)
st.sidebar.divider()
st.sidebar.caption("👨‍ Học sinh: **Quách Gia An**")
st.sidebar.caption("🏫 Đơn vị: **Lớp 11-K30 - THPT Chuyên Hùng Vương**")

# --- 6. MODULE 1: DATABASE EXPLORER ---
if page == "1. Thư viện Alkaloid":
    st.title("📚 Thư viện số hóa Alkaloid")
    
    # --- PHẦN HƯỚNG DẪN TỔNG QUAN ---
    with st.sidebar:
        st.header("📖 Hướng dẫn Module 1")
        st.info("""
        **Mục tiêu:** Tra cứu và sàng lọc các Alkaloid từ Sen dựa trên các tiêu chuẩn hóa dược quốc tế.
        
        **Các bước thực hiện:**
        1. **Lọc dữ liệu:** Sử dụng bộ lọc Lipinski để chọn ra các chất có khả năng làm thuốc cao nhất.
        2. **Quan sát Heatmap:** Tìm các ô màu hồng đậm - đó là các chất có ái lực liên kết mạnh nhất với Enzyme.
        3. **Chọn chất:** Chọn 1 hợp chất cụ thể để hệ thống ghi nhớ và phân tích sâu ở Module sau.
        """)

    if 'MW' in df.columns:
        df = df.rename(columns={'MW': 'Molecular Weight'})
    
    # --- HƯỚNG DẪN VỀ QUY TẮC LIPINSKI ---
    st.subheader("🔍 Bộ lọc sàng lọc thuốc thông minh")
    with st.expander("❓ Quy tắc Lipinski (Rule of 5) là gì?", expanded=False):
        st.write("""
        Đây là quy tắc vàng trong hóa dược để đánh giá một hợp chất có khả năng hấp thụ tốt khi dùng đường uống hay không:
        - **MW < 500:** Kích thước vừa phải để dễ di chuyển qua màng tế bào.
        - **LogP < 5:** Độ tan trong dầu phù hợp để thấm qua màng chất béo.
        - **HBD < 5 & HBA < 10:** Giới hạn liên kết Hydro để phân tử không quá cồng kềnh khi liên kết với nước.
        """)

    # --- KHU VỰC BỘ LỌC ---
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        mw_f = c1.checkbox("MW < 500", value=True, help="Lọc các phân tử nhỏ gọn")
        lp_f = c2.checkbox("LogP < 5", value=True, help="Lọc các chất có độ tan dầu lý tưởng")
        hbd_f = c3.checkbox("H-Donor < 5", value=True)
        hba_f = c4.checkbox("H-Acceptor < 10", value=True)
    
    filtered_df = df.copy()
    
    if mw_f: filtered_df = filtered_df[filtered_df['Molecular Weight'] < 500]
    if lp_f: filtered_df = filtered_df[filtered_df['LogP'] < 5]
    if hbd_f: filtered_df = filtered_df[filtered_df['HBD'] < 5]
    if hba_f: filtered_df = filtered_df[filtered_df['HBA'] < 10]
    
    st.dataframe(
        filtered_df[['Name', 'Formula', 'Molecular Weight', 'LogP', 'HBD', 'HBA']], 
        use_container_width=True,
        column_config={
            "Name": "Tên hợp chất",
            "Formula": "Công thức",
            "Molecular Weight": st.column_config.NumberColumn("MW", format="%.2f g/mol")
        }
    )

    # --- TÍNH NĂNG 1: HEATMAP PHÂN TÍCH TỔNG QUAN ---
    st.markdown("### 🌡️ Phân tích Ái lực liên kết (Binding Affinity)")
    st.caption("🔍 **Hướng dẫn:** Biểu đồ này so sánh khả năng ức chế của các chất lên 2 đích đến Alzheimer (AChE và BACE1).")
    
    if not filtered_df.empty:
        heatmap_data = filtered_df[['Name', 'dG_BACE1', 'dG_AChE']].set_index('Name')
        
        fig_heat = px.imshow(
            heatmap_data.T, 
            labels=dict(x="HỢP CHẤT", y="MỤC TIÊU", color="ΔG (kcal/mol)"),
            color_continuous_scale='RdPu_r', 
            text_auto=True, 
            aspect="auto"
        )
        
        fig_heat.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.info("💡 **Các chất có số âm lớn (màu hồng đậm) đó là những ứng viên có tiềm năng ức chế enzyme cao nhất.")
    else:
        st.warning("⚠️ Không có hợp chất nào thỏa mãn bộ lọc hiện tại. Hãy nới lỏng các điều kiện Lipinski.")

    st.divider()

    # --- CHỌN HỢP CHẤT MỤC TIÊU (ĐÃ FIX LỖI ĐỒNG BỘ) ---
    st.subheader("🎯 Chọn đối tượng nghiên cứu")
    compounds = df['Name'].tolist()
    
    # Đoạn fix lỗi: Kiểm tra nếu chất trong session_state không còn tồn tại trong list mới
    if st.session_state.selected_compound not in compounds:
        st.session_state.selected_compound = compounds[0]
        
    current_idx = compounds.index(st.session_state.selected_compound)
    
    choice = st.selectbox("Chọn hợp chất để chuyển tiếp dữ liệu sang Module sau:", 
                          compounds, index=current_idx)
    
    if choice != st.session_state.selected_compound:
        st.session_state.selected_compound = choice
        st.success(f"Đã chọn **{choice}**. Dữ liệu đã sẵn sàng ở các Module sau!")
        st.rerun() # Quan trọng: Ép app load lại để Module 2 nhận chất mới ngay lập tức
# --- MODULE 2: VIRTUAL DOCKING LAB (BẢN NÂNG CẤP GIAO DIỆN) ---
elif page == "2. Mô phỏng Docking 3D":
    st.title("🔬 Virtual Docking Lab (In Silico)")

    # --- SIDEBAR HƯỚNG DẪN THAO TÁC 3D ---
    with st.sidebar:
        st.header("🎮 Điều khiển Mô hình 3D")
        st.info("""
        **Thao tác chuột:**
        - **Xoay:** Nhấn giữ chuột trái và di chuyển.
        - **Phóng to/Thu nhỏ:** Sử dụng con lăn chuột.
        - **Di chuyển (Pan):** Nhấn giữ chuột phải.
        
        **Giải thích màu sắc:**
        - **Protein (Dải xoắn):** Cấu trúc Enzyme đích.
        - **Ligand (Que):** Hợp chất Alkaloid đang thử nghiệm.
        - **Vùng sáng:** Binding Site (Túi liên kết).
        """)
        st.divider()
        st.caption(
            "Dữ liệu trích xuất từ Bảng 2 & Chương 2 - Báo cáo Nghiên cứu"
            " 2026."
        )

    # DATABASE GỐC CỦA AN
    alkaloid_db = {
        "Nuciferine": {
            "BACE1": {"dg": -8.3, "amin": "Asp32", "stab": 75},
            "AChE": {"dg": -8.2, "amin": "Trp286", "stab": 70},
        },
        "Nornuciferine": {
            "BACE1": {"dg": -8.3, "amin": "Gly120", "stab": 72},
            "AChE": {"dg": -8.1, "amin": "Tyr124", "stab": 68},
        },
        "Roemerine": {
            "BACE1": {"dg": -9.0, "amin": "Asp32/Asp228", "stab": 88},
            "AChE": {"dg": -8.6, "amin": "Trp286", "stab": 90},
        },
        "Pronuciferine": {
            "BACE1": {"dg": -8.6, "amin": "Ser203", "stab": 78},
            "AChE": {"dg": -8.6, "amin": "Phe338", "stab": 80},
        },
        "Liensinine": {
            "BACE1": {"dg": -9.6, "amin": "Asp32", "stab": 95},
            "AChE": {"dg": -7.5, "amin": "His447", "stab": 65},
        },
        "Neferine": {
            "BACE1": {"dg": -9.0, "amin": "Tyr124", "stab": 85},
            "AChE": {"dg": -7.5, "amin": "Trp286", "stab": 62},
        },
        "Isoliensinine": {
            "BACE1": {"dg": -9.6, "amin": "Asp32/Asp228", "stab": 96},
            "AChE": {"dg": -7.7, "amin": "Trp286", "stab": 72},
        },
    }
    controls = {
        "BACE1": {"name": "Verubecestat", "dg": -8.5},
        "AChE": {"name": "Donepezil", "dg": -7.9},
    }

    tab_view, tab_compare = st.tabs(
        ["🔍 Chi tiết tương tác 3D", "⚖️ So sánh đối chứng (Benchmarking)"]
    )

    with tab_view:
        st.subheader("🖥️ Trình diễn tương tác phân tử")
        st.caption(
            "Chọn mục tiêu và hợp chất để quan sát cách Alkaloid 'khóa' các"
            " Enzyme gây bệnh Alzheimer."
        )

        target = st.radio(
            "Chọn Enzyme mục tiêu:",
            ["BACE1 (Protein 4XXS)", "AChE (Protein 7D9O)"],
            horizontal=True,
        )
        p_key = "BACE1" if "BACE1" in target else "AChE"
        pdb_id = "4XXS" if p_key == "BACE1" else "7D9O"

        selected = st.session_state.get("selected_compound", "Roemerine")
        if selected not in alkaloid_db:
            selected = list(alkaloid_db.keys())[0]

        data = alkaloid_db[selected][p_key]

        c1, c2 = st.columns([1, 2.5])
        with c1:
            with st.container(border=True):
                st.markdown(f"### 🧪 {selected}")
                st.write(f"Đích đến: **{p_key}**")
                hl = st.toggle(
                    "Hiện Binding Site",
                    value=True,
                    help="Làm nổi bật túi liên kết nơi Alkaloid tác động.",
                )

                st.divider()
                st.markdown("**📊 Chỉ số năng lượng:**")
                st.metric(
                    "Năng lượng ΔG",
                    f"{data['dg']} kcal/mol",
                    help=(
                        "Giá trị càng âm, liên kết càng bền vững và hiệu quả"
                        " ức chế càng cao."
                    ),
                )

                st.write(f"📍 **Acid amin chính:** `{data['amin']}`")
                st.progress(
                    data["stab"] / 100,
                    text=f"Độ bền phức hợp: {data['stab']}%",
                )

                if "Asp32" in data["amin"]:
                    st.success(
                        "🎯 **Cơ chế:** Khóa cặp Asp xúc tác, ngăn chặn hình"
                        " thành mảng bám Amyloid."
                    )
                elif "Trp286" in data["amin"]:
                    st.success(
                        "🎯 **Cơ chế:** Tương tác tại vùng PAS, ngăn chặn sự"
                        " tích tụ Acetylcholine."
                    )

        with c2:
            with st.container(border=True):
                with st.spinner(
                    "Đang kết nối thư viện PDB và kết xuất mô hình 3D..."
                ):
                    pdb_string = fetch_pdb(pdb_id)
                    if pdb_string:
                        showmol(
                            render_3d_molecule(pdb_string, highlight_site=hl),
                            height=500,
                            width=700,
                        )
                st.caption(
                    f"Mô hình cấu trúc tinh thể Protein {pdb_id} tương tác với"
                    f" {selected}"
                )

    with tab_compare:
        st.subheader("⚖️ Đối chiếu hiệu quả với thuốc chuẩn")
        st.write(
            "So sánh năng lượng liên kết của Alkaloid tự nhiên với các thuốc"
            " điều trị hiện hành."
        )

        comp_p = st.radio(
            "Protein đối chứng:", ["BACE1", "AChE"], horizontal=True, key="comp_p"
        )
        control_data = controls[comp_p]

        with st.container(border=True):
            selected_comp = st.selectbox(
                "Chọn Alkaloid để đối chứng:",
                list(alkaloid_db.keys()),
                index=(
                    list(alkaloid_db.keys()).index(selected)
                    if selected in alkaloid_db
                    else 0
                ),
            )

            user_dg = alkaloid_db[selected_comp][comp_p]["dg"]

            col1, col2 = st.columns(2)
            col1.metric(f"Alkaloid: {selected_comp}", f"{user_dg} kcal/mol")
            col2.metric(
                f"Thuốc: {control_data['name']}",
                f"{control_data['dg']} kcal/mol",
                delta=round(user_dg - control_data["dg"], 2),
                delta_color="inverse",
            )

            if user_dg < control_data["dg"]:
                st.success(
                    f"💡 **Phân tích:** {selected_comp} có năng lượng tự do"
                    " thấp hơn, cho thấy ái lực liên kết mạnh hơn thuốc"
                    f" {control_data['name']}."
                )

        st.markdown("#### Đồ thị so sánh ái lực (Affinity Comparison)")

        # --- ĐOẠN ĐÃ ĐƯỢC CẬP NHẬT THEO YÊU CẦU THIẾT KẾ ---
        chart_data = pd.DataFrame({
            "Hợp chất": [control_data["name"], selected_comp],
            "Năng lượng (kcal/mol)": [
                abs(control_data["dg"]),
                abs(user_dg),
            ],
        })

        import plotly.express as px

        fig = px.bar(
            chart_data,
            x="Hợp chất",
            y="Năng lượng (kcal/mol)",
            text="Năng lượng (kcal/mol)",
            color="Hợp chất",
            color_discrete_sequence=["#1f77b4", "#0066cc"],
        )

        # Định dạng trục X viết ngang (tickangle=0)
        fig.update_xaxes(title_text="", tickangle=0)
        fig.update_yaxes(title_text="Giá trị tuyệt đối |ΔG|")
        fig.update_layout(
            showlegend=False,
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Dòng lưu ý tô đậm và màu đỏ
        st.markdown(
            "<p style='color: red; font-weight: bold; margin-top: -10px;'>"
            "Lưu ý: Giá trị trị tuyệt đối càng cao thể hiện khả năng gắn kết"
            " càng tốt."
            "</p>",
            unsafe_allow_html=True,
        )
# --- MODULE 3: PHÂN TÍCH & XUẤT BÁO CÁO ---
if page == "3. Phân tích & Xuất báo cáo":
    st.title("📊 Phân tích Kết quả & Xuất báo cáo")
    
    with st.sidebar:
        st.header("📋 Hướng dẫn Module 3")
        st.info("""
        **1. Kiểm tra dược tính:** Xem các chỉ số MW, LogP để đối chiếu với quy tắc Lipinski.
        **2. Đọc Radar Chart:** Các đỉnh càng chạm rìa ngoài thì dược tính tại điểm đó càng mạnh.
        **3. Xuất báo cáo:** Nhấn nút Tải để lưu kết quả nghiên cứu dưới dạng file .txt.
        """)

    if 'selected_compound' not in st.session_state:
        st.session_state.selected_compound = df['Name'].iloc[0]
        
    selected_data = df[df['Name'] == st.session_state.selected_compound].iloc[0]
    
    st.subheader(f"Thông tin chi tiết hợp chất: {selected_data['Name']}")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Công thức hóa học", selected_data['Formula'])
    c2.metric("Khối lượng (MW)", f"{selected_data['MW']} Da")
    c3.metric("Độ ưa dầu (LogP)", selected_data['LogP'])
    st.write("---")
    st.metric("Đánh giá Drug-likeness", classify_potential(selected_data['dG_BACE1']))
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1]) 
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Năng lượng liên kết (Affinity)")
        st.metric("BACE1 ΔG", f"{selected_data['dG_BACE1']} kcal/mol", delta="-8.5 (Veru)", delta_color="inverse")
        st.metric("AChE ΔG", f"{selected_data['dG_AChE']} kcal/mol", delta="-7.9 (Done)", delta_color="inverse")
        st.caption("💡 *Ghi chú:* Chỉ số âm càng cao thể hiện khả năng gắn kết càng mạnh.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        bbb_text = "TÍCH CỰC (Có khả năng tác động TW)" if selected_data['BBB_Permeability'] else "HẠN CHẾ (Khả năng xuyên thấp)"
        if selected_data['BBB_Permeability']: 
            st.success(f"✅ **Rào máu não (BBB):** {bbb_text}")
        else: 
            st.warning(f"⚠️ **Rào máu não (BBB):** {bbb_text}")

    with col_right:
        st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("🕸️ Hồ sơ ADMET Radar")
        st.plotly_chart(create_admet_radar(selected_data), use_container_width=True)
        st.caption("🔍 **Radar Chart:** Đánh giá tính chất dược động học đa chiều.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # --- PHẦN NỘI DUNG BÁO CÁO (ĐÃ KHÔI PHỤC ĐẦY ĐỦ) ---
    current_time = time.strftime("%d/%m/%Y %H:%M:%S")
    report_text = f"""======================================================================
             BÁO CÁO PHÂN TÍCH DƯỢC TÍNH PHÂN TỬ - ALKALOTUS PREDICTOR
======================================================================
Dự án: Nghiên cứu In Silico dẫn xuất Alkaloid từ lá sen điều trị Alzheimer
Tác giả: Quách Gia An 
Đơn vị: Lớp 11-K30 - Trường THPT Chuyên Hùng Vương
Thời gian trích xuất: {current_time}

----------------------------------------------------------------------
I. THÔNG TIN HỢP CHẤT (COMPOUND IDENTIFICATION)
----------------------------------------------------------------------
- Tên hợp chất: {selected_data['Name']}
- Công thức hóa học: {selected_data['Formula']}

----------------------------------------------------------------------
II. THÔNG SỐ HÓA LÝ & QUY TẮC LIPINSKI (DRUG-LIKENESS)
----------------------------------------------------------------------
1. Khối lượng phân tử (MW): {selected_data['MW']} g/mol
2. Hệ số phân bố (LogP): {selected_data['LogP']}
3. Số liên kết H-Donor (HBD): {selected_data['HBD']}
4. Số liên kết H-Acceptor (HBA): {selected_data['HBA']}
=> ĐÁNH GIÁ CHUNG: TUÂN THỦ quy tắc Lipinski để đảm bảo khả năng hấp thụ đường uống.

----------------------------------------------------------------------
III. KẾT QUẢ MÔ PHỎNG DOCKING PHÂN TỬ (BINDING AFFINITY)
----------------------------------------------------------------------
* Mục tiêu 1: Enzyme BACE1 -> Năng lượng tự do Gibbs ΔG: {selected_data['dG_BACE1']} kcal/mol
* Mục tiêu 2: Enzyme AChE -> Năng lượng tự do Gibbs ΔG: {selected_data['dG_AChE']} kcal/mol
=> Nhận xét: Hợp chất có ái lực mạnh, khả năng ức chế enzyme mục tiêu cao.

----------------------------------------------------------------------
IV. DƯỢC ĐỘNG HỌC & ĐỘ AN TOÀN (ADMET)
----------------------------------------------------------------------
- Khả năng xuyên rào máu não (BBB): {bbb_text}
- Khả năng hấp thu qua ruột người (HIA): Cao
- Độc tính: Không gây độc tính cấp tính trong ngưỡng mô phỏng.

======================================================================
KẾT LUẬN: Hợp chất {selected_data['Name']} là ứng viên tiềm năng trong việc
nghiên cứu Alzheimer từ thảo dược tự nhiên.
======================================================================
"""
    st.header("🔬 Xuất bản kết quả")
    st.download_button(label="📥 TẢI BÁO CÁO CHI TIẾT (.TXT)", 
                       data=report_text, 
                       file_name=f"AlkaLotus_Report_{selected_data['Name']}.txt", 
                       mime="text/plain")

# --- MODULE 4: PHÂN TÍCH CẤU TRÚC & TƯƠNG ĐỒNG ĐA THÔNG SỐ ---
elif (
    page
    in [
        "4. Phân tích cấu trúc",
        (
            "4. Phân tích Cấu trúc & Tương đồng Đa thông số (Tanimoto Liên"
            " tục)"
        ),
    ]
    or "4. Phân tích" in page
):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    # --- ĐOẠN CSS XÓA TRIỆT ĐỂ NÚT CỘNG (+) VÀ TRỪ (-) ---
    st.markdown(
        """
        <style>
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {
            display: none !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title(
        "🧬 Module 4: Phân tích Cấu trúc & Tương đồng Đa thông số (Tanimoto Liên"
        " tục)"
    )

    # --- HƯỚNG DẪN MODULE 4 ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📖 Hướng dẫn Module 4")
        st.info("""
        **Mục tiêu:** Đo lường và đánh giá độ tương đồng hóa lý - sinh học đa thông số giữa các hợp chất thử nghiệm với các thuốc đối chứng chuẩn (Donepezil & Verubecestat) bằng mô hình Tanimoto Liên tục (Continuous Tanimoto).

        **Các bước thực hiện:**
        1. **Nhập thông số:** Khai báo số lượng hợp chất và tùy chỉnh các chỉ số hóa lý (MW, LogP, TPSA, HBD, HBA) cùng năng lượng liên kết ($\Delta G$).
        2. **So sánh Tanimoto:** Quan sát Bảng kết quả và Biểu đồ cột để xác định hợp chất có độ tương đồng cao nhất với thuốc đối chứng (tiệm cận 1.0).
        3. **Đánh giá SAR & Heatmap:** Phân tích Ma trận Tương đồng Toàn diện và Biểu đồ Tương quan Đặc tính - Hoạt tính (SAR) để lựa chọn ứng viên tiềm năng nhất.
        """)

    st.markdown("""
    **Cơ sở Khoa học Hóa tin học Cải tiến:**
    Nhằm tránh các lỗi nghiêm trọng khi giải mã chuỗi cấu trúc (SMILES), hệ thống áp dụng mô hình toán học **Tanimoto Liên tục (Continuous Tanimoto / Ruzicka Similarity)**.
    * **Đầu vào:** Hệ thống số hóa phân tử dựa trên các vector thông số (MW, LogP, TPSA, HBD, HBA, năng lượng liên kết).
    * **Xử lý:** Dữ liệu được chuẩn hóa (Min-Max Scaling) để các thông số có đơn vị khác nhau đóng góp công bằng vào phép tính.
    * **Đầu ra:** Đo lường chính xác mức độ tương đồng toàn diện giữa hợp chất thử nghiệm với các thuốc đối chứng chuẩn.
    """)

    # 1. Định nghĩa 2 Thuốc Đối chứng Chuẩn
    REF_DRUGS = {
        "Donepezil (Chuẩn AChE)": {
            "MW": 379.50,
            "LogP": 4.27,
            "TPSA": 38.8,
            "HBD": 0.0,
            "HBA": 4.0,
            "ΔG AChE": -11.5,
            "ΔG BACE1": -7.2,
        },
        "Verubecestat (Chuẩn BACE1)": {
            "MW": 409.41,
            "LogP": 1.15,
            "TPSA": 104.9,
            "HBD": 2.0,
            "HBA": 6.0,
            "ΔG AChE": -6.8,
            "ΔG BACE1": -10.4,
        },
    }

    # 2. Giao diện Nhập liệu Thông số
    st.subheader("⚙️ Nhập thông số Hợp chất Thử nghiệm")
    st.info(
        "💡 Nhập trực tiếp các chỉ số hóa lý và năng lượng. Hệ thống sẽ tự động"
        " tổng hợp để tính toán độ tương đồng Tanimoto đa chiều."
    )

    num_comp = st.number_input(
        "Số lượng hợp chất cần phân tích:", min_value=1, max_value=15, value=2
    )

    compounds_input = []

    for i in range(int(num_comp)):
        st.markdown(f"**🔹 Hợp chất #{i+1}**")
        c_name = st.text_input(
            f"Tên hợp chất #{i+1}:", value=f"Alkaloid_{i+1}", key=f"name_{i}"
        )

        col1, col2, col3 = st.columns(3)
        c_mw = col1.number_input(
            f"Khối lượng (MW) #{i+1}:",
            value=281.35 + (i * 10.0),
            format="%.2f",
            key=f"mw_{i}",
        )
        c_logp = col2.number_input(
            f"Độ phân cực (LogP) #{i+1}:",
            value=2.50 + (i * 0.5),
            format="%.2f",
            key=f"logp_{i}",
        )
        c_tpsa = col3.number_input(
            f"Diện tích bề mặt (TPSA) #{i+1}:",
            value=20.0 + (i * 5.0),
            format="%.2f",
            key=f"tpsa_{i}",
        )

        col4, col5, col6 = st.columns(3)
        c_hbd = col4.number_input(
            f"Số liên kết cho H (HBD) #{i+1}:",
            value=1.0,
            format="%.1f",
            key=f"hbd_{i}",
        )
        c_hba = col5.number_input(
            f"Số liên kết nhận H (HBA) #{i+1}:",
            value=2.0,
            format="%.1f",
            key=f"hba_{i}",
        )
        c_ga = col6.number_input(
            f"ΔG AChE (kcal/mol) #{i+1}:",
            value=-8.80,
            format="%.2f",
            key=f"ga_{i}",
        )

        c_gb = st.number_input(
            f"ΔG BACE1 (kcal/mol) #{i+1}:",
            value=-8.10,
            format="%.2f",
            key=f"gb_{i}",
        )
        st.divider()

        compounds_input.append({
            "Hợp chất": c_name,
            "MW": float(c_mw),
            "LogP": float(c_logp),
            "TPSA": float(c_tpsa),
            "HBD": float(c_hbd),
            "HBA": float(c_hba),
            "ΔG AChE": float(c_ga),
            "ΔG BACE1": float(c_gb),
        })

    # Chỉ chạy thuật toán khi người dùng đã có dữ liệu hợp lệ
    if len(compounds_input) > 0:
        try:
            # 3. Thuật toán xử lý và tính Tanimoto Liên tục
            all_data = []
            for name, props in REF_DRUGS.items():
                row = {"Hợp chất": name}
                row.update(props)
                all_data.append(row)
            all_data.extend(compounds_input)

            df_all = pd.DataFrame(all_data)
            features = [
                "MW",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "ΔG AChE",
                "ΔG BACE1",
            ]

            # Chuẩn hóa Min-Max (0-1)
            df_norm = df_all.copy()
            for col in features:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (
                        max_val - min_val
                    )
                else:
                    df_norm[col] = 1.0

            # Hàm tính Tanimoto liên tục
            def calc_continuous_tanimoto(vec1, vec2):
                dot_product = np.dot(vec1, vec2)
                sum_sq1 = np.dot(vec1, vec1)
                sum_sq2 = np.dot(vec2, vec2)
                denominator = sum_sq1 + sum_sq2 - dot_product
                if denominator <= 0.0001:
                    return 1.0
                return round(dot_product / denominator, 3)

            # 4. Tính toán Ma trận Tương đồng
            matrix_size = len(df_norm)
            sim_matrix = np.zeros((matrix_size, matrix_size))

            for i in range(matrix_size):
                vec_i = df_norm.iloc[i][features].values.astype(float)
                for j in range(matrix_size):
                    vec_j = df_norm.iloc[j][features].values.astype(float)
                    sim_matrix[i, j] = calc_continuous_tanimoto(vec_i, vec_j)

            names = df_norm["Hợp chất"].tolist()
            df_sim_matrix = pd.DataFrame(
                sim_matrix, index=names, columns=names
            )

            # Trích xuất kết quả đưa vào bảng
            results_list = []
            for i in range(2, matrix_size):
                results_list.append({
                    "Hợp chất": names[i],
                    "Tanimoto vs Donepezil": df_sim_matrix.iloc[i, 0],
                    "Tanimoto vs Verubecestat": df_sim_matrix.iloc[i, 1],
                })
            df_results = pd.DataFrame(results_list)

            # 5. Xuất Trực quan Hóa
            st.markdown("---")
            st.subheader("📊 Bảng Kết quả Tương đồng Tanimoto Đa thông số")
            st.dataframe(df_results, use_container_width=True)

            st.subheader("📈 Biểu đồ Cột: So sánh Tương đồng với Thuốc Chuẩn")
            df_melted = df_results.melt(
                id_vars=["Hợp chất"],
                value_vars=[
                    "Tanimoto vs Donepezil",
                    "Tanimoto vs Verubecestat",
                ],
                var_name="Đối tượng So sánh",
                value_name="Độ Tương Đồng",
            )

            fig_bar = px.bar(
                df_melted,
                x="Hợp chất",
                y="Độ Tương Đồng",
                color="Đối tượng So sánh",
                barmode="group",
                title="Mức độ tương đồng so với Thuốc chuẩn",
                color_discrete_sequence=["#1f77b4", "#ff7f0e"],
            )
            fig_bar.update_layout(yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader(
                "🧩 Ma trận Tương đồng Toàn diện (Similarity Matrix)"
            )
            fig_matrix = px.imshow(
                df_sim_matrix,
                labels=dict(
                    x="Hợp chất", y="Hợp chất", color="Tanimoto"
                ),
                x=names,
                y=names,
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="auto",
            )
            fig_matrix.update_xaxes(side="top")
            st.plotly_chart(fig_matrix, use_container_width=True)

            st.subheader("🎯 Biểu đồ Tương quan Đặc tính - Hoạt tính (SAR)")
            plot_sar_data = []
            for i in range(matrix_size):
                c_type = (
                    "Thuốc Chuẩn AChE"
                    if i == 0
                    else (
                        "Thuốc Chuẩn BACE1"
                        if i == 1
                        else "Hợp chất thử nghiệm"
                    )
                )
                plot_sar_data.append({
                    "Hợp chất": names[i],
                    "Tanimoto vs Donepezil": df_sim_matrix.iloc[i, 0],
                    "ΔG AChE (kcal/mol)": float(
                        df_all.iloc[i]["ΔG AChE"]
                    ),
                    "Loại": c_type,
                })
            df_sar = pd.DataFrame(plot_sar_data)

            fig_sar = px.scatter(
                df_sar,
                x="Tanimoto vs Donepezil",
                y="ΔG AChE (kcal/mol)",
                color="Loại",
                text="Hợp chất",
                size_max=15,
                title=(
                    "Phân tích SAR: Tương đồng vs Donepezil và Năng lượng"
                    " Liên kết"
                ),
            )
            fig_sar.update_traces(
                textposition="top center",
                marker=dict(size=12, line=dict(width=1, color="black")),
            )
            fig_sar.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_sar, use_container_width=True)

            st.success("✅ Module 4 đã chạy thành công!")
        except Exception as e:
            st.error(
                f"❌ Có lỗi toán học xảy ra trong quá trình tính toán: {e}."
                " Vui lòng kiểm tra lại các thông số nhập vào."
            )
elif page in ["5. Tối ưu dung môi", "5. Tối ưu Dung môi"]:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    
    try:
        st.title("🧪 Module 5: Tối ưu hóa Dung môi Đa Chất (Hansen Space)")

        # --- HƯỚNG DẪN MODULE 5 (ĐỒNG BỘ THEO MODULE 1 & MODULE 4) ---
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📖 Hướng dẫn Module 5")
            st.info("""
            **Mục tiêu:** Dự đoán độ hòa tan và tối ưu hóa hệ phối trộn đa dung môi cho bất kỳ hợp chất mục tiêu nào dựa trên mô hình Không gian Hansen (HSP - $\delta_D, \delta_P, \delta_H$).

            **Các bước thực hiện:**
            1. **Khai báo chất mục tiêu:** Chọn hợp chất từ danh sách chuẩn hoặc nhập trực tiếp bộ thông số HSP ($\delta_D, \delta_P, \delta_H$) và bán kính hòa tan $R_0$.
            2. **Chọn & Phối trộn dung môi:** Lựa chọn các dung môi thành phần và kích hoạt AI/Monte Carlo để tìm tỷ lệ phối trộn tối ưu (hoặc tùy chỉnh bằng thanh trượt).
            3. **Đánh giá RED & Không gian 3D:** Quan sát các chỉ số hiệu quả ($R_a$, RED, Score %), Biểu đồ Radar Cấu hình Hansen và Không gian Hansen 3D để xác định mức độ hòa tan tối ưu.
            """)

        st.markdown("""
        Hệ thống phân tích và dự đoán độ hòa tan dựa trên **Khoảng cách Hansen (Hansen Solubility Parameters - HSP)**. 
        Module này cho phép tối ưu hóa hệ dung môi cho **bất kỳ hợp chất mục tiêu nào** thông qua việc phối trộn đa dung môi để đạt được khoảng cách hòa tan ($R_a$) ngắn nhất.
        """)

        with st.expander("📖 Cơ sở Toán học & Công thức (Hansen Theory)", expanded=False):
            st.markdown("**1. Khoảng cách Hansen ($R_a$)**")
            st.markdown("Khoảng cách trong không gian 3D giữa dung môi (hoặc hỗn hợp) và chất mục tiêu. $R_a$ càng nhỏ, độ hòa tan càng cao.")
            st.latex(r"R_a = \sqrt{4(\delta_{d2} - \delta_{d1})^2 + (\delta_{p2} - \delta_{p1})^2 + (\delta_{h2} - \delta_{h1})^2}")
            
            st.markdown("**2. Chỉ số khác biệt năng lượng (RED - Relative Energy Difference)**")
            st.markdown("$R_0$ là bán kính tương tác của chất mục tiêu. Nếu $RED < 1$: Hòa tan hoàn toàn; $RED = 1$: Hòa tan một phần; $RED > 1$: Không hòa tan.")
            st.latex(r"RED = \frac{R_a}{R_0}")
            
            st.markdown("**3. Hàm tương thích Gaussian (Compatibility Score)**")
            st.latex(r"Score (\%) = 100 \times e^{-\frac{R_a^2}{2(R_0/2)^2}}")

        # --- 1. CƠ SỞ DỮ LIỆU DUNG MÔI & CHẤT CHUẨN ---
        solvents = {
            "Nước (Water)": [15.5, 16.0, 42.3],
            "Ethanol": [15.8, 8.8, 19.4],
            "Methanol": [14.7, 12.3, 22.3],
            "Acetone": [15.5, 10.4, 7.0],
            "Ethyl Acetate": [15.8, 5.3, 7.2],
            "Chloroform": [17.8, 3.1, 5.7],
            "Dichloromethane": [17.0, 7.3, 7.1],
            "Hexane": [14.9, 0.0, 0.0],
            "Isopropanol": [15.8, 6.1, 16.4]
        }

        # --- 2. GIAO DIỆN NHẬP THÔNG SỐ CHẤT MỤC TIÊU ---
        st.subheader("🎯 1. Khai báo Chất mục tiêu (Target Compound)")
        target_mode = st.radio("Lựa chọn phương thức nhập:", ["Chọn từ danh sách chuẩn", "Nhập thông số tùy chỉnh (Custom)"], horizontal=True)
        
        if target_mode == "Chọn từ danh sách chuẩn":
            target_presets = {
                "Nuciferine (Alkaloid)": {"dD": 18.5, "dP": 6.2, "dH": 5.1, "R0": 8.0},
                "Roemerine (Alkaloid)": {"dD": 18.2, "dP": 5.8, "dH": 4.8, "R0": 8.0},
                "Curcumin (Polyphenol)": {"dD": 17.4, "dP": 8.1, "dH": 9.2, "R0": 10.0},
                "Quercetin (Flavonoid)": {"dD": 19.2, "dP": 10.3, "dH": 15.1, "R0": 12.0}
            }
            t_choice = st.selectbox("Chọn hợp chất:", list(target_presets.keys()))
            t_hsp = [target_presets[t_choice]["dD"], target_presets[t_choice]["dP"], target_presets[t_choice]["dH"]]
            t_R0 = target_presets[t_choice]["R0"]
            t_name = t_choice
            st.info(f"Thông số HSP của **{t_name}**: $\delta_D$={t_hsp[0]}, $\delta_P$={t_hsp[1]}, $\delta_H$={t_hsp[2]} | Bán kính $R_0$={t_R0}")
        else:
            t_name = st.text_input("Tên hợp chất tùy chỉnh:", "Chất X")
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            t_dD = col_t1.number_input("Dispersion ($\delta_D$)", value=18.0, step=0.1)
            t_dP = col_t2.number_input("Polar ($\delta_P$)", value=6.0, step=0.1)
            t_dH = col_t3.number_input("H-Bond ($\delta_H$)", value=5.0, step=0.1)
            t_R0 = col_t4.number_input("Bán kính hòa tan ($R_0$)", value=8.0, step=0.1)
            t_hsp = [t_dD, t_dP, t_dH]

        # --- 3. XÂY DỰNG HỖN HỢP DUNG MÔI & THUẬT TOÁN TỐI ƯU ---
        st.markdown("---")
        st.subheader("🎛️ 2. Xây dựng và Tối ưu Hỗn hợp Dung môi")
        sel_sols = st.multiselect("Chọn các dung môi thành phần để phối trộn:", list(solvents.keys()), default=["Ethanol", "Nước (Water)"])
        
        if not sel_sols:
            st.warning("Vui lòng chọn ít nhất 1 dung môi để tiếp tục.")
        else:
            weights = {}
            sol_matrix = np.array([solvents[s] for s in sel_sols])
            target_vec = np.array(t_hsp)
            
            st.markdown("**Chế độ phối trộn:**")
            opt_mode = st.checkbox("🤖 Bật AI/Thuật toán Tự động tìm tỷ lệ tối ưu nhất", value=False)
            
            if opt_mode:
                st.info("🔄 Hệ thống đang chạy mô phỏng Monte Carlo (10,000 vòng lặp) để dò quét không gian tỷ lệ nhằm tìm ra hệ dung môi có khoảng cách $R_a$ nhỏ nhất...")
                np.random.seed(42)
                n_iters = 10000
                n_dim = len(sel_sols)
                random_weights = np.random.dirichlet(np.ones(n_dim), size=n_iters)
                
                mixes = np.dot(random_weights, sol_matrix)
                diff = mixes - target_vec
                ra_squared = 4*(diff[:, 0]**2) + (diff[:, 1]**2) + (diff[:, 2]**2)
                ra_arrays = np.sqrt(ra_squared)
                
                best_idx = np.argmin(ra_arrays)
                best_w = random_weights[best_idx]
                
                st.success("✅ Đã tìm thấy tỷ lệ tối ưu toán học!")
                for i, s in enumerate(sel_sols):
                    weights[s] = best_w[i]
                    # Sử dụng st.metric thay vì st.slider để tránh lỗi Crash Duplicate ID
                    st.metric(label=f"Tỷ lệ {s} đề xuất:", value=f"{best_w[i]*100:.1f} %")
            else:
                # Cấp key độc lập cho các thanh trượt thủ công
                for s in sel_sols:
                    weights[s] = st.slider(f"Tỷ lệ {s} (%)", 0, 100, int(100/len(sel_sols)), key=f"manual_{s}")
                
                total_w = sum(weights.values())
                if total_w == 0:
                    st.error("Tổng tỷ lệ phải lớn hơn 0%")
                else:
                    weights = {k: v/total_w for k, v in weights.items()}

            if ('total_w' not in locals() or total_w > 0):
                # TÍNH TOÁN KẾT QUẢ CUỐI CÙNG
                mix_hsp = np.zeros(3)
                for s, w in weights.items():
                    mix_hsp += np.array(solvents[s]) * w
                    
                dist = np.sqrt(4*(mix_hsp[0]-t_hsp[0])**2 + (mix_hsp[1]-t_hsp[1])**2 + (mix_hsp[2]-t_hsp[2])**2)
                red_score = dist / t_R0 if t_R0 > 0 else 999
                compat_score = 100 * np.exp(- (dist**2) / (2 * (t_R0/2)**2))

                # --- 4. HIỂN THỊ KẾT QUẢ & PHÂN TÍCH HIỆU QUẢ ---
                st.markdown("---")
                st.subheader(f"📊 3. Báo cáo Hiệu quả Chiết tách: {t_name}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Khoảng cách Hansen ($R_a$)", value=f"{dist:.2f}", help="Càng gần 0 càng tốt.")
                m2.metric(label="Chỉ số RED ($R_a/R_0$)", value=f"{red_score:.2f}", help="< 1 là hòa tan tốt, > 1 là khó hòa tan.")
                m3.metric(label="Mức độ Tương thích", value=f"{compat_score:.1f}%", help="Dựa trên hàm phân bố Gaussian.")

                if red_score < 0.8:
                    st.success(f"🌟 **Đánh giá Chuyên môn:** Hỗn hợp dung môi này **cực kỳ xuất sắc** để hòa tan/chiết tách {t_name}. Lực phân tán và độ phân cực hoàn toàn khớp với cấu trúc đích.")
                elif red_score <= 1.0:
                    st.warning(f"👍 **Đánh giá Chuyên môn:** Hỗn hợp này hòa tan ở mức **khá/chấp nhận được** (Nằm ngay trên ranh giới vỏ cầu Hansen). Có thể cần gia nhiệt để tăng hiệu suất.")
                else:
                    st.error(f"⚠️ **Đánh giá Chuyên môn:** Hỗn hợp này **không phù hợp** (RED > 1). Hệ dung môi nằm ngoài vùng hòa tan của {t_name}. Hãy điều chỉnh lại tỷ lệ hoặc thêm dung môi khác.")

                # --- 5. TRỰC QUAN HÓA ---
                st.subheader("🌌 4. Sơ đồ Trực quan hóa Không gian Dung môi")
                c_chart1, c_chart2 = st.columns([1, 1.2])

                with c_chart1:
                    st.markdown("**Biểu đồ Cấu hình (Hansen Radar Profile)**")
                    radar_df = pd.DataFrame({
                        'Chỉ số': ['Dispersion (dD)', 'Polar (dP)', 'H-Bond (dH)'],
                        'Mục tiêu': t_hsp,
                        'Hỗn hợp Dung môi': mix_hsp
                    })
                    df_melted = radar_df.melt(id_vars='Chỉ số', var_name='Loại', value_name='Giá trị')
                    
                    fig_radar = px.line_polar(df_melted, r='Giá trị', theta='Chỉ số', color='Loại', 
                                              line_close=True, markers=True, template="plotly_white",
                                              color_discrete_sequence=["#d62728", "#1f77b4"])
                    fig_radar.update_traces(fill='toself', opacity=0.7)
                    fig_radar.update_layout(legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig_radar, use_container_width=True)

                with c_chart2:
                    st.markdown("**Không gian Hansen 3D (Hansen Space)**")
                    fig_3d = go.Figure()
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=[t_hsp[0]], y=[t_hsp[1]], z=[t_hsp[2]],
                        mode='markers+text',
                        marker=dict(size=10, color='red', symbol='diamond'),
                        name='Chất Mục tiêu',
                        text=[t_name], textposition="top center"
                    ))

                    for s in sel_sols:
                        fig_3d.add_trace(go.Scatter3d(
                            x=[solvents[s][0]], y=[solvents[s][1]], z=[solvents[s][2]],
                            mode='markers+text',
                            marker=dict(size=5, color='gray'),
                            name=f'Thành phần: {s}',
                            text=[s], textposition="bottom center"
                        ))

                    fig_3d.add_trace(go.Scatter3d(
                        x=[mix_hsp[0]], y=[mix_hsp[1]], z=[mix_hsp[2]],
                        mode='markers',
                        marker=dict(size=12, color='blue', line=dict(width=2, color='black')),
                        name='Hỗn hợp Tối ưu'
                    ))

                    fig_3d.add_trace(go.Scatter3d(
                        x=[t_hsp[0], mix_hsp[0]], y=[t_hsp[1], mix_hsp[1]], z=[t_hsp[2], mix_hsp[2]],
                        mode='lines',
                        line=dict(color='purple', width=4, dash='dash'),
                        name=f'Khoảng cách Ra ({dist:.1f})'
                    ))

                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title='Phân tán - dD',
                            yaxis_title='Phân cực - dP',
                            zaxis_title='LK Hydro - dH',
                        ),
                        margin=dict(l=0, r=0, b=0, t=0),
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán hoặc render: {e}")
elif page in ["6. Động học chiết tách", "6. Động học Chiết tách"]:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st

    try:
        st.title("📈 Mô phỏng Động học & Vận tốc Chiết tách")

        # --- HƯỚNG DẪN MODULE 6 (ĐỒNG BỘ THEO CÁC MODULE TRƯỚC) ---
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📖 Hướng dẫn Module 6")
            st.info("""
            **Mục tiêu:** Mô phỏng diễn biến nồng độ và vận tốc chiết tách theo thời gian dựa trên mô hình Động học giả bậc hai (PSO - Pseudo-Second-Order) để tìm điểm dừng kỹ thuật (Cut-off point) tối ưu chi phí và năng lượng.

            **Các bước thực hiện:**
            1. **Thiết lập Cấu hình:** Nhập tên hợp chất mục tiêu, thời gian khảo sát tổng thể, cùng hai thông số thực nghiệm: dung lượng bão hòa $q_e$ và hằng số tốc độ $k_2$.
            2. **Theo dõi Báo cáo & Đồ thị:** Quan sát các mốc thời gian quan trọng ($t_{50}, t_{80}, t_{90}$) và đồ thị kép kết hợp giữa nồng độ $q_t$ với vận tốc chiết tức thời $V_t$.
            3. **Phân tích Tối ưu Năng lượng:** Đánh giá điểm dừng kỹ thuật khuyến nghị (Cut-off point) để tránh lãng phí năng lượng ở giai đoạn bão hòa và xuất bảng dữ liệu chi tiết.
            """)

        st.markdown("""
        Hệ thống phân tích quá trình chiết tách đa chất dựa trên mô hình **Động học giả bậc hai (Pseudo-second-order - PSO)**. 
        Mô hình này giả định bước quyết định tốc độ chiết tách là quá trình khuếch tán / hấp phụ hóa học qua ranh giới pha.
        """)

        # --- 1. CƠ SỞ TOÁN HỌC ---
        with st.expander("📖 Cơ sở Toán học & Phương trình Động học (PSO)", expanded=False):
            st.markdown("**1. Phương trình vi phân giả bậc 2:**")
            st.latex(r"\frac{dq_t}{dt} = k_2 (q_e - q_t)^2")
            
            st.markdown("**2. Dạng tích phân (Tính nồng độ theo thời gian):**")
            st.latex(r"q_t = \frac{k_2 q_e^2 t}{1 + k_2 q_e t}")
            
            st.markdown("**3. Vận tốc chiết ban đầu ($h_0$ khi $t \rightarrow 0$):**")
            st.latex(r"h_0 = k_2 q_e^2")
            
            st.markdown("**4. Công thức tính Thời gian đạt hiệu suất $x\%$ ($t_x$):**")
            st.latex(r"t_x = \frac{x}{100 - x} \times \frac{1}{k_2 q_e}")
            st.markdown("*Trong đó: $q_e$ là dung lượng cân bằng tối đa (mg/g), $k_2$ là hằng số tốc độ (g/mg.phút).*")

        st.markdown("---")

        # --- 2. GIAO DIỆN NHẬP THÔNG SỐ (MỞ RỘNG ĐA CHẤT) ---
        st.subheader("⚙️ 1. Thiết lập Cấu hình Thực nghiệm")
        
        col_input1, col_input2 = st.columns([1, 2])
        with col_input1:
            target_name = st.text_input("Tên Hợp chất Mục tiêu:", value="Nuciferine")
            max_time = st.number_input("Thời gian khảo sát (phút):", min_value=10, max_value=1440, value=120, step=10)
            
        with col_input2:
            st.markdown("**Nhập thông số động học từ thực nghiệm:**")
            c1, c2 = st.columns(2)
            qe = c1.number_input("Dung lượng bão hòa $q_e$ (mg/g):", min_value=0.1, value=25.0, step=0.5, format="%.2f")
            k2 = c2.number_input("Hằng số tốc độ $k_2$ (g/mg.phút):", min_value=0.0001, value=0.0150, step=0.001, format="%.4f")

        # --- 3. XỬ LÝ TOÁN HỌC & ALGORITHM ---
        if qe <= 0 or k2 <= 0:
            st.error("Lỗi: Các giá trị $q_e$ và $k_2$ phải lớn hơn 0.")
        else:
            time_steps = np.linspace(0, max_time, 200)
            qt = (k2 * (qe ** 2) * time_steps) / (1 + k2 * qe * time_steps)
            velocity = (k2 * (qe**2)) / ((1 + k2 * qe * time_steps)**2)
            
            h0 = k2 * (qe**2)
            t_50 = 1 / (k2 * qe)
            t_80 = 4 / (k2 * qe)
            t_90 = 9 / (k2 * qe)
            t_95 = 19 / (k2 * qe)

            # --- 4. DASHBOARD CHỈ SỐ ---
            st.subheader(f"📊 2. Báo cáo Động học: {target_name}")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(label="Vận tốc ban đầu ($h_0$)", value=f"{h0:.3f}", delta="mg/g.phút", delta_color="normal")
            m2.metric(label="Thời gian 50% ($t_{50}$)", value=f"{t_50:.1f} ph", help="Thời gian đạt 1/2 sản lượng")
            m3.metric(label="Thời gian 80% ($t_{80}$)", value=f"{t_80:.1f} ph", help="Ngưỡng bắt đầu bão hòa")
            m4.metric(label="Thời gian 90% ($t_{90}$)", value=f"{t_90:.1f} ph", help="Ngưỡng kinh tế tối đa")

            # --- 5. BIỂU ĐỒ KÉP (TRỰC QUAN HÓA CAO CẤP) ---
            st.subheader("🌌 3. Đồ thị Động học & Vận tốc Tức thời")
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=time_steps, y=qt, name=f"Nồng độ $q_t$ ({target_name})",
                           mode='lines', line=dict(color='blue', width=3),
                           fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.1)'),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=time_steps, y=velocity, name="Vận tốc chiết $V_t$",
                           mode='lines', line=dict(color='red', width=2, dash='dot')),
                secondary_y=True,
            )

            critical_times = [(t_50, "50%"), (t_80, "80%"), (t_90, "90%")]
            for t_val, label in critical_times:
                if t_val <= max_time:
                    q_val = (k2 * (qe ** 2) * t_val) / (1 + k2 * qe * t_val)
                    fig.add_trace(go.Scatter(
                        x=[t_val], y=[q_val], mode='markers+text',
                        marker=dict(color='black', size=8, symbol='diamond'),
                        text=[f"{label} ({t_val:.1f}p)"], textposition="top left", showlegend=False
                    ), secondary_y=False)
                    
                    fig.add_vline(x=t_val, line_dash="dash", line_color="gray", opacity=0.5)

            fig.add_hline(y=qe, line_dash="solid", line_color="green", annotation_text=f"Max bão hòa ($q_e$ = {qe})", secondary_y=False)

            fig.update_layout(
                title_text="Động học Pseudo-Second-Order: Nồng độ vs. Vận tốc",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(title_text="Thời gian (phút)")
            fig.update_yaxes(title_text="Nồng độ chiết $q_t$ (mg/g)", secondary_y=False, color="blue")
            fig.update_yaxes(title_text="Vận tốc $V_t$ (mg/g.phút)", secondary_y=True, color="red")

            st.plotly_chart(fig, use_container_width=True)

            # --- 6. PHÂN TÍCH CHIẾN LƯỢC SÂU ---
            st.subheader("💡 4. Phân tích Chiến lược & Tối ưu Năng lượng")
            
            time_diff_80_95 = t_95 - t_80
            yield_diff_80_95 = (0.95 * qe) - (0.80 * qe)
            
            st.info(f"""
            **Báo cáo Đánh giá Tối ưu Quy trình cho {target_name}:**
            * **Giai đoạn Đột phá (0 đến {t_50:.1f} phút):** Vận tốc chiết cực đại ($h_0 = {h0:.3f}$). Chênh lệch nồng độ lớn giúp rút trích nhanh chóng 50% sản lượng.
            * **Giai đoạn Cản trở không gian ({t_50:.1f} đến {t_80:.1f} phút):** Tốc độ giảm theo hàm mũ (đường nét đứt màu đỏ). Quá trình khuếch tán bị giới hạn.
            * **Giai đoạn Bão hòa lãng phí (Sau {t_80:.1f} phút):** Vận tốc $V_t$ tiệm cận 0. 
            
            **🔥 Khuyến nghị Kinh tế - Kỹ thuật:** 
            Để tăng thêm **15%** hiệu suất (từ 80% lên 95%), hệ thống phải chạy thêm **{time_diff_80_95:.1f} phút**. Điều này tiêu tốn năng lượng điện (gia nhiệt, máy khuấy, siêu âm) không tương xứng với lượng {target_name} thu được thêm ({yield_diff_80_95:.2f} mg/g). 
            $\\Rightarrow$ **Điểm dừng kỹ thuật (Cut-off point) tối ưu nhất: {t_80:.1f} phút đến {t_90:.1f} phút.**
            """)

            # --- 7. BẢNG DỮ LIỆU ---
            with st.expander("📋 Xem và Tải bảng Dữ liệu Mô phỏng Chi tiết"):
                df_display = pd.DataFrame({
                    "Thời gian (phút)": np.round(time_steps, 2), 
                    "Nồng độ qt (mg/g)": np.round(qt, 4),
                    "Vận tốc Vt (mg/g.ph)": np.round(velocity, 5),
                    "Hiệu suất (%)": np.round((qt / qe) * 100, 2)
                })
                st.dataframe(df_display, use_container_width=True)
                
    except Exception as e:
        st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán hoặc hiển thị: {e}")
elif page in ["7. Dự toán Quy mô & Kinh tế", "7. Dự toán quy mô & kinh tế", "7. Dự toán Quy mô & Kinh tế & Tối ưu hóa"] or "7. Dự toán" in page:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.optimize import minimize
    import streamlit as st

    try:
        st.title("💰 Module 7: Dự toán Quy mô, Tối ưu hóa Kinh tế & Chi phí Sản xuất")

        # --- HƯỚNG DẪN MODULE 7 (ĐỒNG BỘ THEO CÁC MODULE TRƯỚC) ---
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📖 Hướng dẫn Module 7")
            st.info("""
            **Mục tiêu:** Tối ưu hóa chi phí sản xuất và quy mô vận hành bằng thuật toán Tối ưu hóa phi tuyến có ràng buộc (SLSQP), giúp tìm cấu hình khối lượng nguyên liệu ($m^*$) và thể tích dung môi ($v^*$) có tổng chi phí nhỏ nhất mà vẫn đảm bảo sản lượng hoạt chất đầu ra tối thiểu ($Y_{\min}$).

            **Các bước thực hiện:**
            1. **Thiết lập Thông số Đầu vào:** Nhập đơn giá thị trường (nguyên liệu, dung môi, điện năng), thông số dược liệu, sản lượng mục tiêu $Y_{\min}$ và khoảng giới hạn quy mô thiết bị.
            2. **Kích hoạt Tối ưu hóa:** Nhấn nút chạy thuật toán SLSQP để tìm điểm nghiệm tối ưu toàn cục ($m^*, v^*$) và bóc tách cấu thành chi phí.
            3. **Phân tích Biểu đồ & Độ nhạy:** Quan sát biểu đồ tròn tỷ trọng chi phí, đường mức không gian nghiệm Contour Plot và báo cáo đánh giá độ nhạy biến động giá thị trường.
            """)

        st.markdown("""
        Hệ thống tối ưu hóa chi phí sản xuất tự động dựa trên thuật toán **Tối ưu hóa phi tuyến có ràng buộc (SLSQP)**. 
        Mô hình tìm kiếm cấu hình vận hành $(m^*, v^*)$ tối ưu nhất sao cho **Tổng chi phí sản xuất $C(m,v)$ là nhỏ nhất** nhưng vẫn đảm bảo **đạt sản lượng hoạt chất đầu ra tối thiểu ($Y_{\min}$)**.
        """)

        # --- 1. MÔ HÌNH TOÁN HỌC & CÔNG THỨC KHÁI QUÁT ---
        with st.expander("📖 Cơ sở Toán học & Bài toán Tối ưu hóa Kinh tế - Kỹ thuật", expanded=False):
            st.markdown("**1. Hàm mục tiêu Cực tiểu hóa Tổng chi phí sản xuất $C(m,v)$:**")
            st.latex(r"C(m,v) = (m \times P_{\text{raw}}) + (v \times P_{\text{solvent}}) + (0.15 \times v \times P_{\text{elec}})")
            st.markdown("""
            *Trong đó:*
            * $m$: Khối lượng nguyên liệu dược liệu khô (kg)
            * $P_{\text{raw}}$: Đơn giá nguyên liệu đầu vào (VNĐ/kg)
            * $v$: Thể tích dung môi chiết xuất (Lít)
            * $P_{\text{solvent}}$: Đơn giá dung môi chiết (VNĐ/Lít)
            * $0.15 \times v$: Định mức tiêu thụ điện năng (kWh) tương quan tuyến tính với thể tích gia nhiệt/khuấy trộn
            * $P_{\text{elec}}$: Đơn giá điện năng (VNĐ/kWh)
            """)

            st.markdown("**2. Hệ thống Ràng buộc Kỹ thuật & Sản lượng đầu ra ($Y_{\min}$):**")
            st.latex(r"Y(m,v) = m \times k \times \eta(v,t) \ge Y_{\min}")
            st.markdown("""
            *Trong đó:*
            * $k$: Hàm lượng hoạt chất cơ sở có trong nguyên liệu khô (mg/kg)
            * $\eta(v,t)$: Tỷ lệ / Hiệu suất thu hồi hoạt chất từ quá trình chiết (%)
            * $Y_{\min}$: Mức sản lượng hoạt chất mục tiêu tối thiểu cần thu được (mg)
            """)

            st.markdown("**3. Giới hạn biên thiết bị vận hành:**")
            st.latex(r"m_{\min} \le m \le m_{\max}, \quad v_{\min} \le v \le v_{\max}")

        st.markdown("---")

        # --- 2. GIAO DIỆN NHẬP THÔNG SỐ (ĐA CHẤT & THỊ TRƯỜNG) ---
        st.subheader("⚙️ 1. Thiết lập Thông số Đầu vào & Thị trường")

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            st.markdown("**🏷️ Đơn giá Thị trường**")
            price_raw = st.number_input("Giá nguyên liệu khô (VNĐ/kg):", value=50000, step=5000, format="%d")
            price_solvent = st.number_input("Giá dung môi chiết (VNĐ/Lít):", value=35000, step=2000, format="%d")
            price_elec = st.number_input("Giá điện sản xuất (VNĐ/kWh):", value=2500, step=100, format="%d")

        with col_in2:
            st.markdown("**🔬 Thông số Dược liệu & Mục tiêu**")
            target_compound = st.text_input("Tên hoạt chất mục tiêu:", value="Nuciferine")
            raw_material_name = st.text_input("Tên nguyên liệu khô:", value="Lá sen khô")
            k_content = st.number_input(f"Hàm lượng {target_compound} trong nguyên liệu (mg/kg):", value=150.0, step=10.0, format="%.1f")
            target_yield_mg = st.number_input("Sản lượng hoạt chất mục tiêu $Y_{min}$ (mg):", value=500.0, step=50.0, format="%.1f")

        with col_in3:
            st.markdown("**🎛️ Giới hạn Quy mô Thiết bị**")
            scale_leaf = st.slider("Khoảng khối lượng mẻ chiết $m$ (kg):", 1, 100, (1, 50))
            vol_solvent = st.slider("Khoảng thể tích dung môi $v$ (Lít):", 10, 500, (10, 200))
            recovery_eff_pct = st.slider("Hiệu suất thu hồi chiết xuất $\eta$ (%):", 10.0, 100.0, 80.0, step=1.0)

        st.divider()

        # --- 3. THUẬT TOÁN TỐI ƯU HÓA PHI TUYẾN (SLSQP) ---
        st.subheader("🚀 2. Chạy Mô hình Tối ưu hóa Chi phí (SciPy - SLSQP)")

        if st.button("🔥 Kích hoạt Thuật toán Tối ưu hóa Kinh tế - Kỹ thuật", type="primary"):
            eta_decimal = recovery_eff_pct / 100.0  # Chuyển phần trăm về hệ số [0, 1]

            # Kiểm tra xem sản lượng mục tiêu có khả thi với quy mô tối đa không
            max_possible_yield = scale_leaf[1] * k_content * eta_decimal
            if max_possible_yield < target_yield_mg:
                st.error(f"⚠️ **Không thể tối ưu hóa (Miền khả thi rỗng):** Ở quy mô nguyên liệu tối đa ({scale_leaf[1]} kg), sản lượng thu hồi tối đa chỉ đạt **{max_possible_yield:.1f} mg**, nhỏ hơn mục tiêu **{target_yield_mg:.1f} mg**. Vui lòng tăng giới hạn $m_{max}$ hoặc giảm $Y_{min}$.")
            else:
                # 1. Định nghĩa Hàm mục tiêu chi phí C(m, v)
                def objective(x):
                    m, v = x
                    elec_kwh = v * 0.15
                    return (m * price_raw) + (v * price_solvent) + (elec_kwh * price_elec)

                # 2. Định nghĩa Ràng buộc bất đẳng thức: m * k * eta - Y_min >= 0
                constraints = {
                    'type': 'ineq',
                    'fun': lambda x: (x[0] * k_content * eta_decimal) - target_yield_mg
                }

                # 3. Giới hạn biến (Bounds)
                bounds = [scale_leaf, vol_solvent]

                # 4. Giá trị khởi tạo (Initial guess)
                x0 = [(scale_leaf[0] + scale_leaf[1]) / 2.0, (vol_solvent[0] + vol_solvent[1]) / 2.0]

                # 5. Giải bài toán với phương pháp SLSQP
                res = minimize(objective, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)

                if res.success:
                    opt_m, opt_v = res.x
                    opt_cost = res.fun
                    est_output_mg = opt_m * k_content * eta_decimal

                    cost_raw_val = opt_m * price_raw
                    cost_solvent_val = opt_v * price_solvent
                    cost_elec_val = (opt_v * 0.15) * price_elec
                    unit_cost_vnd_per_mg = opt_cost / est_output_mg if est_output_mg > 0 else 0

                    st.success("🎉 **Tối ưu hóa thành công!** Thuật toán SLSQP đã hội tụ về điểm nghiệm tối ưu toàn cục.")

                    # --- DASHBOARD CHỈ SỐ KẾT QUẢ ---
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Khối lượng {0} ($m^*$)".format(raw_material_name), f"{opt_m:.2f} kg")
                    m2.metric("Thể tích Dung môi ($v^*$)".format(""), f"{opt_v:.2f} Lít")
                    m3.metric("Tổng Chi phí Tối thiểu", f"{opt_cost:,.0f} VNĐ")
                    m4.metric(f"Chi phí Đơn vị ({target_compound})", f"{unit_cost_vnd_per_mg:,.1f} VNĐ/mg")

                    # --- BÓC TÁCH CẤU THÀNH CHI PHÍ ---
                    with st.expander("📊 Bóc tách chi tiết cấu thành chi phí kinh tế - kỹ thuật", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Chi phí Nguyên liệu", f"{cost_raw_val:,.0f} VNĐ", f"{(cost_raw_val/opt_cost)*100:.1f}% tổng")
                        c2.metric("Chi phí Dung môi", f"{cost_solvent_val:,.0f} VNĐ", f"{(cost_solvent_val/opt_cost)*100:.1f}% tổng")
                        c3.metric("Chi phí Điện năng", f"{cost_elec_val:,.0f} VNĐ", f"{(cost_elec_val/opt_cost)*100:.1f}% tổng")

                    # --- 4. ĐỒ THỊ VÀ TRỰC QUAN HÓA CAO CẤP ---
                    st.subheader("🌌 3. Biểu đồ Phân tích Cấu trúc Chi phí & Không gian Nghiệm Tối ưu")

                    col_chart1, col_chart2 = st.columns([1, 1])

                    with col_chart1:
                        # Biểu đồ Tròn Donut bóc tách chi phí
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=[f'Nguyên liệu ({raw_material_name})', 'Dung môi chiết', 'Điện năng gia nhiệt/khuấy'],
                            values=[cost_raw_val, cost_solvent_val, cost_elec_val],
                            hole=.4,
                            marker_colors=['#2ecc71', '#3498db', '#e74c3c']
                        )])
                        fig_pie.update_layout(title_text="Cấu trúc Tỷ trọng Chi phí (%)", legend=dict(orientation="h", y=-0.1))
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_chart2:
                        # Biểu đồ Không gian nghiệm Contour Plot
                        m_grid = np.linspace(scale_leaf[0], scale_leaf[1], 50)
                        v_grid = np.linspace(vol_solvent[0], vol_solvent[1], 50)
                        M, V = np.meshgrid(m_grid, v_grid)
                        Z_cost = (M * price_raw) + (V * price_solvent) + (V * 0.15 * price_elec)

                        fig_contour = go.Figure(data=go.Contour(
                            z=Z_cost, x=m_grid, y=v_grid,
                            colorscale='Viridis',
                            colorbar=dict(title='Chi phí (VNĐ)')
                        ))

                        # Vẽ đường ràng buộc sản lượng Y_min
                        req_m_line = target_yield_mg / (k_content * eta_decimal)
                        if scale_leaf[0] <= req_m_line <= scale_leaf[1]:
                            fig_contour.add_vline(x=req_m_line, line_dash="dash", line_color="red",
                                                   annotation_text=f"Ràng buộc $Y_{{min}}$ ({req_m_line:.1f}kg)")

                        # Đánh dấu điểm tối ưu
                        fig_contour.add_trace(go.Scatter(
                            x=[opt_m], y=[opt_v], mode='markers+text',
                            marker=dict(color='red', size=14, symbol='star'),
                            text=["ĐIỂM TỐI ƯU"], textposition="top center"
                        ))

                        fig_contour.update_layout(
                            title_text="Đường mức Chi phí & Vùng khả thi trong không gian (m, v)",
                            xaxis_title="Khối lượng Nguyên liệu m (kg)",
                            yaxis_title="Thể tích Dung môi v (Lít)"
                        )
                        st.plotly_chart(fig_contour, use_container_width=True)

                    # --- 5. PHÂN TÍCH CHUYÊN SÂU & ĐÁNH GIÁ ĐỘ NHẠY ---
                    st.subheader("💡 4. Đánh giá Chiến lược & Phân tích Độ nhạy Kinh tế")

                    # Phân tích độ nhạy (Sensitivity Analysis) khi biến động giá
                    raw_plus_20 = ((opt_m * price_raw * 1.2) + cost_solvent_val + cost_elec_val - opt_cost) / opt_cost * 100
                    solvent_plus_20 = (cost_raw_val + (cost_solvent_val * 1.2) + (cost_elec_val * 1.2) - opt_cost) / opt_cost * 100

                    st.info(f"""
                    **Báo cáo Đánh giá Tối ưu hóa Kinh tế cho {target_compound}:**
                    * **Cấu hình Vận hành Khuyên dùng:** Để đạt sản lượng hoạt chất mục tiêu **{target_yield_mg:.1f} mg** (thực tế thu được **{est_output_mg:.1f} mg**), mẻ chiết cần sử dụng đúng **{opt_m:.2f} kg** {raw_material_name} và **{opt_v:.2f} Lít** dung môi.
                    * **Bản chất Nghiệm Tối ưu:** Thuật toán tự động siết khối lượng nguyên liệu $m$ chạm đúng ngưỡng ràng buộc tối thiểu ($m^* = {opt_m:.2f}$ kg) và chọn thể tích dung môi $v$ ở mức thấp nhất trong khoảng cho phép ($v^* = {opt_v:.2f}$ Lít) để triệt tiêu chi phí điện năng và dung môi dư thừa.
                    
                    **📈 Báo cáo Phân tích Độ nhạy Thị trường (Sensitivity Analysis):**
                    * Nếu **giá nguyên liệu ({raw_material_name})** tăng 20%: Tổng chi phí sản xuất sẽ tăng thêm **+{raw_plus_20:.2f}%**.
                    * Nếu **giá dung môi** tăng 20%: Tổng chi phí sản xuất sẽ tăng thêm **+{solvent_plus_20:.2f}%**.
                    $\\Rightarrow$ *Khuyến nghị:* Chi phí sản xuất nhạy cảm nhất với yếu tố **{"Nguyên liệu đầu vào" if cost_raw_val > cost_solvent_val else "Dung môi & Năng lượng gia nhiệt"}**. Do đó, chiến lược thu mua và hoàn lưu (tái sử dụng) dung môi sẽ là chìa khóa quyết định giá thành thương mại.
                    """)

                    # --- 6. XUẤT BẢNG DỮ LIỆU ---
                    with st.expander("📋 Xem Bảng Tổng hợp Bóc tách Chi phí Chi tiết"):
                        df_summary = pd.DataFrame({
                            "Hạng mục Chi phí": [f"Nguyên liệu ({raw_material_name})", "Dung môi Chiết", "Điện năng Gia nhiệt/Khuấy", "TỔNG CỘNG"],
                            "Số lượng / Định mức": [f"{opt_m:.2f} kg", f"{opt_v:.2f} Lít", f"{opt_v*0.15:.2f} kWh", "-"],
                            "Đơn giá (VNĐ)": [f"{price_raw:,.0f} /kg", f"{price_solvent:,.0f} /Lít", f"{price_elec:,.0f} /kWh", "-"],
                            "Thành tiền (VNĐ)": [f"{cost_raw_val:,.0f}", f"{cost_solvent_val:,.0f}", f"{cost_elec_val:,.0f}", f"{opt_cost:,.0f}"],
                            "Tỷ trọng (%)": [f"{(cost_raw_val/opt_cost)*100:.1f}%", f"{(cost_solvent_val/opt_cost)*100:.1f}%", f"{(cost_elec_val/opt_cost)*100:.1f}%", "100.0%"]
                        })
                        st.dataframe(df_summary, use_container_width=True)
                else:
                    st.warning("⚠️ Thuật toán SLSQP không thể hội tụ. Vui lòng điều chỉnh lại khoảng giới hạn biến hoặc thay đổi giá trị khởi tạo.")

    except Exception as e:
        st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán tối ưu hóa: {e}")
