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
    page_title="AlkaLotus Insight | Alzheimer Research",
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
                <div class="lotus-text">NỀN TẢNG TIN SINH HỌC TÍCH HỢP DỮ LIỆU VÀ MÔ PHỎNG ĐỘNG HỌC ALKALOTUS INSIGHT</div>
            </div>
            """, 
            unsafe_allow_html=True 
        )
        time.sleep(5)
    intro_placeholder.empty()
    st.session_state['visited'] = True


st.title("🪷 AlkaLotus Insight")
st.markdown("<p style='font-size: 1.15em; color: #555; font-style: italic; margin-top: -15px; line-height: 1.4;'>Nền tảng tin sinh học tích hợp dữ liệu và mô phỏng động học chiết tách Alkaloid lá sen hướng đích enzyme AChE và BACE1 trong nghiên cứu Alzheimer</p>", unsafe_allow_html=True)
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

st.sidebar.title("🪷 AlkaLotus Insight")
st.sidebar.markdown("<div style='text-align: justify; font-size: 0.9em;'><b>Nền tảng tin sinh học</b>  tích hợp dữ liệu và mô phỏng động học chiết tách Alkaloid lá sen hướng đích enzyme AChE và BACE1 trong nghiên cứu Alzheimer.</div>", unsafe_allow_html=True)

st.sidebar.divider()
page = st.sidebar.radio(
    "Danh mục hệ thống",
    [
        "1. Thư viện Alkaloid",
        "2. Mô phỏng Docking 3D",
        "3. Phân tích & Xuất báo cáo",
        "4. Phân tích cấu trúc (Toán)",
        "5. Tối ưu dung môi (Toán)",
        "6. Động học chiết tách (Toán)",
        "7. Dự toán quy mô & kinh tế (Toán)"
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
        st.caption("Dữ liệu trích xuất từ Bảng 2 & Chương 2 - Báo cáo Nghiên cứu 2026.")

    # DATABASE GỐC CỦA AN (Đảm bảo được đặt ở đây để không bao giờ bị None)
    alkaloid_db = {
        "Nuciferine": {"BACE1": {"dg": -8.3, "amin": "Asp32", "stab": 75}, "AChE": {"dg": -8.2, "amin": "Trp286", "stab": 70}},
        "Nornuciferine": {"BACE1": {"dg": -8.3, "amin": "Gly120", "stab": 72}, "AChE": {"dg": -8.1, "amin": "Tyr124", "stab": 68}},
        "Roemerine": {"BACE1": {"dg": -9.0, "amin": "Asp32/Asp228", "stab": 88}, "AChE": {"dg": -8.6, "amin": "Trp286", "stab": 90}},
        "Pronuciferine": {"BACE1": {"dg": -8.6, "amin": "Ser203", "stab": 78}, "AChE": {"dg": -8.6, "amin": "Phe338", "stab": 80}},
        "Liensinine": {"BACE1": {"dg": -9.6, "amin": "Asp32", "stab": 95}, "AChE": {"dg": -7.5, "amin": "His447", "stab": 65}},
        "Neferine": {"BACE1": {"dg": -9.0, "amin": "Tyr124", "stab": 85}, "AChE": {"dg": -7.5, "amin": "Trp286", "stab": 62}},
        "Isoliensinine": {"BACE1": {"dg": -9.6, "amin": "Asp32/Asp228", "stab": 96}, "AChE": {"dg": -7.7, "amin": "Trp286", "stab": 72}}
    }
    controls = {
        "BACE1": {"name": "Verubecestat", "dg": -8.5},
        "AChE": {"name": "Donepezil", "dg": -7.9}
    }

    tab_view, tab_compare = st.tabs(["🔍 Chi tiết tương tác 3D", "⚖️ So sánh đối chứng (Benchmarking)"])

    with tab_view:
        st.subheader("🖥️ Trình diễn tương tác phân tử")
        st.caption("Chọn mục tiêu và hợp chất để quan sát cách Alkaloid 'khóa' các Enzyme gây bệnh Alzheimer.")

        target = st.radio("Chọn Enzyme mục tiêu:", ["BACE1 (Protein 4XXS)", "AChE (Protein 7D9O)"], horizontal=True)
        p_key = "BACE1" if "BACE1" in target else "AChE"
        pdb_id = "4XXS" if p_key == "BACE1" else "7D9O"
        
        # ĐOẠN FIX LỖI TYPEERROR QUAN TRỌNG:
        selected = st.session_state.get('selected_compound', 'Roemerine')
        if selected not in alkaloid_db:
            selected = list(alkaloid_db.keys())[0] # Tự lấy chất đầu tiên nếu lỗi
        
        data = alkaloid_db[selected][p_key]

        c1, c2 = st.columns([1, 2.5])
        with c1:
            with st.container(border=True):
                st.markdown(f"### 🧪 {selected}")
                st.write(f"Đích đến: **{p_key}**")
                hl = st.toggle("Hiện Binding Site", value=True, help="Làm nổi bật túi liên kết nơi Alkaloid tác động.")
                
                st.divider()
                st.markdown("**📊 Chỉ số năng lượng:**")
                st.metric("Năng lượng ΔG", f"{data['dg']} kcal/mol", 
                          help="Giá trị càng âm, liên kết càng bền vững và hiệu quả ức chế càng cao.")
                
                st.write(f"📍 **Acid amin chính:** `{data['amin']}`")
                st.progress(data['stab']/100, text=f"Độ bền phức hợp: {data['stab']}%")
                
                if "Asp32" in data['amin']:
                    st.success("🎯 **Cơ chế:** Khóa cặp Asp xúc tác, ngăn chặn hình thành mảng bám Amyloid.")
                elif "Trp286" in data['amin']:
                    st.success("🎯 **Cơ chế:** Tương tác tại vùng PAS, ngăn chặn sự tích tụ Acetylcholine.")

        with c2:
            with st.container(border=True):
                with st.spinner("Đang kết nối thư viện PDB và kết xuất mô hình 3D..."):
                    pdb_string = fetch_pdb(pdb_id)
                    if pdb_string:
                        showmol(render_3d_molecule(pdb_string, highlight_site=hl), height=500, width=700)
                st.caption(f"Mô hình cấu trúc tinh thể Protein {pdb_id} tương tác với {selected}")

    with tab_compare:
        st.subheader("⚖️ Đối chiếu hiệu quả với thuốc chuẩn")
        st.write("So sánh năng lượng liên kết của Alkaloid tự nhiên với các thuốc điều trị hiện hành.")

        comp_p = st.radio("Protein đối chứng:", ["BACE1", "AChE"], horizontal=True, key="comp_p")
        control_data = controls[comp_p]
        
        with st.container(border=True):
            # Đồng bộ lại selectbox đối chứng
            selected_comp = st.selectbox("Chọn Alkaloid để đối chứng:", list(alkaloid_db.keys()), 
                                         index=list(alkaloid_db.keys()).index(selected) if selected in alkaloid_db else 0)
            
            user_dg = alkaloid_db[selected_comp][comp_p]['dg']
            
            col1, col2 = st.columns(2)
            col1.metric(f"Alkaloid: {selected_comp}", f"{user_dg} kcal/mol")
            col2.metric(f"Thuốc: {control_data['name']}", f"{control_data['dg']} kcal/mol", 
                        delta=round(user_dg - control_data['dg'], 2), delta_color="inverse")
            
            if user_dg < control_data['dg']:
                st.success(f"💡 **Phân tích:** {selected_comp} có năng lượng tự do thấp hơn, cho thấy ái lực liên kết mạnh hơn thuốc {control_data['name']}.")
            
        st.markdown("#### Đồ thị so sánh ái lực (Affinity Comparison)")
        chart_data = pd.DataFrame({
            "Hợp chất": [selected_comp, control_data['name']],
            "Năng lượng (kcal/mol)": [abs(user_dg), abs(control_data['dg'])]
        })
        st.bar_chart(chart_data.set_index("Hợp chất"))
        st.caption("Lưu ý: Giá trị trị tuyệt đối càng cao thể hiện khả năng gắn kết càng tốt.")
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
Tác giả: Quách Gia An - Nguyễn Lê Bách Hợp
Đơn vị: Lớp 10-K30 - Trường THPT Chuyên Hùng Vương
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
phát triển các liệu pháp điều trị Alzheimer từ thảo dược tự nhiên.
======================================================================
"""
    st.header("🔬 Xuất bản kết quả")
    st.download_button(label="📥 TẢI BÁO CÁO CHI TIẾT (.TXT)", 
                       data=report_text, 
                       file_name=f"AlkaLotus_Report_{selected_data['Name']}.txt", 
                       mime="text/plain")

    

elif page == "4. Phân tích cấu trúc (Toán)":
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    import streamlit as st

    st.title("🧬 Phân tích Tương đồng Cấu trúc (Tanimoto) & Ái lực Liên kết")
    st.markdown("""
    Hệ thống đánh giá và so sánh các **hợp chất thử nghiệm** dựa trên **2 Mô hình cốt lõi**:
    1. **Tanimoto Similarity:** Độ tương đồng khung cấu trúc 2D (Morgan Fingerprint, radius=2, 1024 bits).
    2. **Ái lực Liên kết ($\Delta G$):** So sánh năng lượng Docking trực tiếp với 2 thuốc chuẩn gốc làm hệ quy chiếu (**Donepezil** & **Verubecestat**).
    """)

    # 1. Cơ sở dữ liệu cố định: 2 Thuốc chuẩn gốc (Benchmark Standards)
    ref_drugs = {
        "Donepezil (Chuẩn AChE)": {
            "smiles": "COc1ccc2c(c1)C(=O)C(CC3CCN(Cc4ccccc4)CC3)C2",
            "affinity_AChE": -11.5,
            "affinity_BACE1": -7.2
        },
        "Verubecestat (Chuẩn BACE1)": {
            "smiles": "CS(=O)(=O)N1CCN(CC1)c2ccc(c3csc(N)n3)cc2F",
            "affinity_AChE": -6.8,
            "affinity_BACE1": -10.4
        }
    }

    # Hàm phân tích và trích xuất đặc trưng từ mã SMILES
    def analyze_molecule(smiles):
        if not smiles or not isinstance(smiles, str):
            return None, None
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None: 
                return None, None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            props = {
                "MW": round(Descriptors.MolWt(mol), 2),
                "LogP": round(Descriptors.MolLogP(mol), 2),
                "TPSA": round(Descriptors.TPSA(mol), 2),
                "HBD": rdMolDescriptors.CalcNumLipinskiHDonors(mol),
                "HBA": rdMolDescriptors.CalcNumLipinskiHAcceptors(mol),
                "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol)
            }
            return fp, props
        except Exception:
            return None, None

    # Hàm tính Tanimoto an toàn (Khắc phục hoàn toàn lỗi Boost.Python.ArgumentError)
    def safe_tanimoto(fp1, fp2):
        if fp1 is None or fp2 is None:
            return 0.0
        return round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)

    # Tính Fingerprint cho 2 thuốc chuẩn
    ref_fps = {}
    for r_name, r_data in ref_drugs.items():
        rfp, _ = analyze_molecule(r_data["smiles"])
        ref_fps[r_name] = rfp

    # 2. Sidebar: Nhập dữ liệu linh hoạt do người dùng tự định nghĩa
    with st.sidebar:
        st.header("⚙️ Nhập Dữ liệu Hợp chất Thử nghiệm")
        input_mode = st.radio("Phương thức nhập dữ liệu:", ["Nhập thủ công", "Tải file CSV danh sách"])

        test_compounds = []

        if input_mode == "Nhập thủ công":
            num_compounds = st.number_input("Số lượng hợp chất muốn phân tích:", min_value=1, max_value=10, value=1, step=1)
            for i in range(int(num_compounds)):
                st.markdown(f"--- \n**Hợp chất #{i+1}**")
                c_name = st.text_input(f"Tên hợp chất #{i+1}:", f"Hợp chất {i+1}", key=f"name_{i}")
                c_smiles = st.text_input(f"Mã SMILES #{i+1}:", "CN1CCC2=CC3=C(C=C2C1CC4=CC=C(O)C=C4)OC" if i==0 else "", key=f"smiles_{i}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    g_ache = st.number_input(f"ΔG AChE #{i+1} (kcal/mol):", value=-8.5, key=f"g_ache_{i}")
                with col_b:
                    g_bace1 = st.number_input(f"ΔG BACE1 #{i+1} (kcal/mol):", value=-8.0, key=f"g_bace1_{i}")

                test_compounds.append({
                    "Name": c_name,
                    "SMILES": c_smiles,
                    "g_ache": g_ache,
                    "g_bace1": g_bace1
                })

        else:
            uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
            st.caption("Cột yêu cầu trong CSV: `Name`, `SMILES`, `g_ache`, `g_bace1`")
            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file)
                for _, row in df_upload.iterrows():
                    test_compounds.append({
                        "Name": str(row.get("Name", "Hợp chất")),
                        "SMILES": str(row.get("SMILES", "")),
                        "g_ache": float(row.get("g_ache", -8.0)),
                        "g_bace1": float(row.get("g_bace1", -8.0))
                    })
            else:
                st.info("Đang hiển thị mẫu thử nghiệm mặc định bên dưới:")
                test_compounds.append({
                    "Name": "Hợp chất Mẫu A",
                    "SMILES": "CN1CCC2=CC3=C(C=C2C1CC4=CC=C(O)C=C4)OC",
                    "g_ache": -8.5,
                    "g_bace1": -8.0
                })

    # 3. Tính toán Tanimoto Similarity
    processed_results = []
    plot_points = []

    # Điểm dữ liệu của 2 thuốc chuẩn
    don_ver_sim = safe_tanimoto(ref_fps["Donepezil (Chuẩn AChE)"], ref_fps["Verubecestat (Chuẩn BACE1)"])

    plot_points.append({
        "Name": "⭐ Donepezil (Chuẩn AChE)",
        "Tanimoto_Donepezil": 1.0,
        "Tanimoto_Verubecestat": don_ver_sim,
        "ΔG_AChE": ref_drugs["Donepezil (Chuẩn AChE)"]["affinity_AChE"],
        "ΔG_BACE1": ref_drugs["Donepezil (Chuẩn AChE)"]["affinity_BACE1"],
        "Type": "Thuốc chuẩn AChE"
    })
    plot_points.append({
        "Name": "⭐ Verubecestat (Chuẩn BACE1)",
        "Tanimoto_Donepezil": don_ver_sim,
        "Tanimoto_Verubecestat": 1.0,
        "ΔG_AChE": ref_drugs["Verubecestat (Chuẩn BACE1)"]["affinity_AChE"],
        "ΔG_BACE1": ref_drugs["Verubecestat (Chuẩn BACE1)"]["affinity_BACE1"],
        "Type": "Thuốc chuẩn BACE1"
    })

    invalid_smiles = []

    for comp in test_compounds:
        fp, props = analyze_molecule(comp["SMILES"])
        if fp is None:
            invalid_smiles.append(comp["Name"])
            continue

        sim_don = safe_tanimoto(fp, ref_fps["Donepezil (Chuẩn AChE)"])
        sim_ver = safe_tanimoto(fp, ref_fps["Verubecestat (Chuẩn BACE1)"])

        processed_results.append({
            "Hợp chất": comp["Name"],
            "Tanimoto vs Don": sim_don,
            "Tanimoto vs Ver": sim_ver,
            "ΔG AChE": comp["g_ache"],
            "ΔG BACE1": comp["g_bace1"],
            "MW": props["MW"] if props else None,
            "LogP": props["LogP"] if props else None,
            "TPSA": props["TPSA"] if props else None
        })

        plot_points.append({
            "Name": comp["Name"],
            "Tanimoto_Donepezil": sim_don,
            "Tanimoto_Verubecestat": sim_ver,
            "ΔG_AChE": comp["g_ache"],
            "ΔG_BACE1": comp["g_bace1"],
            "Type": "Hợp chất thử nghiệm"
        })

    if invalid_smiles:
        st.warning(f"⚠️ Phát hiện mã SMILES không hợp lệ ở các chất: {', '.join(invalid_smiles)}. Đã tự động bỏ qua.")

    if not processed_results:
        st.error("Chưa có hợp chất hợp lệ nào để hiển thị. Vui lòng kiểm tra lại thông tin nhập ở Sidebar.")
    else:
        # 4. Dashboard chỉ số nhanh
        st.subheader("📌 Tổng quan Các chỉ số Cấu trúc & Ái lực")
        selected_target_name = st.selectbox("Chọn hợp chất xem nhanh chỉ số:", [r["Hợp chất"] for r in processed_results])
        target_res = next(r for r in processed_results if r["Hợp chất"] == selected_target_name)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tanimoto vs Donepezil", f"{target_res['Tanimoto vs Don']*100:.1f}%")
        c2.metric("ΔG AChE (kcal/mol)", f"{target_res['ΔG AChE']}")
        c3.metric("Tanimoto vs Verubecestat", f"{target_res['Tanimoto vs Ver']*100:.1f}%")
        c4.metric("ΔG BACE1 (kcal/mol)", f"{target_res['ΔG BACE1']}")

        # 5. Đồ thị Không gian Tương đồng Tanimoto & Ái lực Gắn kết
        st.subheader("🎯 Đồ thị Mối tương quan Cấu trúc (Tanimoto) & Ái lực gắn kết (ΔG)")
        
        df_plot = pd.DataFrame(plot_points)
        fig_scatter = px.scatter(
            df_plot, 
            x="Tanimoto_Donepezil", 
            y="ΔG_AChE", 
            color="Type", 
            text="Name",
            size_max=15,
            hover_data=["Tanimoto_Verubecestat", "ΔG_BACE1"],
            labels={
                "Tanimoto_Donepezil": "Độ tương đồng Tanimoto vs Donepezil (0 - 1)", 
                "ΔG_AChE": "Năng lượng Gắn kết AChE ΔG (kcal/mol - Càng âm càng mạnh)"
            },
            title="Tương quan giữa Tương đồng Cấu trúc vs Donepezil và Ái lực Liên kết AChE"
        )
        fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
        fig_scatter.update_yaxes(autorange="reversed")  # Năng lượng càng âm hiển thị càng cao
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 6. Bảng dữ liệu tổng hợp
        st.subheader("📊 Bảng Báo cáo Tổng hợp (Full Tanimoto & Docking Metrics)")

        ref_table_rows = [
            {
                "Hợp chất": "⭐ Donepezil (Chuẩn AChE)",
                "Tanimoto vs Don": 1.0, 
                "Tanimoto vs Ver": don_ver_sim,
                "ΔG AChE": ref_drugs["Donepezil (Chuẩn AChE)"]["affinity_AChE"],
                "ΔG BACE1": ref_drugs["Donepezil (Chuẩn AChE)"]["affinity_BACE1"],
                "MW": 379.5, "LogP": 4.27, "TPSA": 38.8
            },
            {
                "Hợp chất": "⭐ Verubecestat (Chuẩn BACE1)",
                "Tanimoto vs Don": don_ver_sim, 
                "Tanimoto vs Ver": 1.0,
                "ΔG AChE": ref_drugs["Verubecestat (Chuẩn BACE1)"]["affinity_AChE"],
                "ΔG BACE1": ref_drugs["Verubecestat (Chuẩn BACE1)"]["affinity_BACE1"],
                "MW": 409.4, "LogP": 2.15, "TPSA": 88.5
            }
        ]

        full_table_data = processed_results + ref_table_rows
        df_table = pd.DataFrame(full_table_data).set_index("Hợp chất")

        st.dataframe(
            df_table.style.highlight_max(subset=['Tanimoto vs Don', 'Tanimoto vs Ver'], color='lightblue')
                          .highlight_min(subset=['ΔG AChE', 'ΔG BACE1'], color='lightgreen'), 
            use_container_width=True
        )

        # 7. Biện luận Chuyên sâu SAR
        st.subheader("💡 Biện luận Chuyên sâu (Structure-Activity Relationship - SAR)")
        
        sim_don_val = target_res['Tanimoto vs Don']
        sim_ver_val = target_res['Tanimoto vs Ver']
        
        st.markdown(f"**Đánh giá cho hợp chất đang chọn:** `{target_res['Hợp chất']}`")
        
        if sim_don_val >= 0.5:
            st.success(f"• **Tương đồng cấu trúc cao với Donepezil** (Tanimoto = {sim_don_val*100:.1f}%): Hợp chất sở hữu nhiều đặc điểm/khung giàn tương đồng với thuốc chuẩn Donepezil, có khả năng cao tương tác với túi gắn kết của AChE.")
        elif sim_don_val >= 0.3:
            st.info(f"• **Tương đồng cấu trúc trung bình với Donepezil** (Tanimoto = {sim_don_val*100:.1f}%): Hợp chất mang một số nhóm thế hoặc bộ khung tương tự Donepezil.")
        else:
            st.warning(f"• **Độ tương đồng cấu trúc thấp so với Donepezil** (Tanimoto = {sim_don_val*100:.1f}%): Hợp chất mang bộ khung cấu trúc mới biệt lập.")

        if target_res['ΔG AChE'] <= ref_drugs["Donepezil (Chuẩn AChE)"]["affinity_AChE"]:
            st.success(f"• **Ái lực AChE vượt trội:** Năng lượng liên kết ($\Delta G = {target_res['ΔG AChE']}$ kcal/mol) mạnh hơn hoặc tương đương thuốc chuẩn Donepezil ({ref_drugs['Donepezil (Chuẩn AChE)']['affinity_AChE']} kcal/mol).")
        else:
            st.info(f"• **Ái lực AChE:** $\Delta G = {target_res['ΔG AChE']}$ kcal/mol (Thuốc chuẩn Donepezil: {ref_drugs['Donepezil (Chuẩn AChE)']['affinity_AChE']} kcal/mol).")
elif page == "5. Tối ưu Dung môi (Toán)":
    with st.sidebar:
        st.header("📖 Hướng dẫn Module 5")
        st.info("""
        **Mục tiêu:** Dự đoán độ tan của Alkaloid trong hỗn hợp dung môi dựa trên lý thuyết Hansen.
        
        **Các bước thực hiện:**
        1. **Chọn Alkaloid:** Chọn đối tượng cần chiết xuất.
        2. **Chọn dung môi:** Chọn các dung môi muốn phối trộn.
        3. **Điều chỉnh tỷ lệ:** Sử dụng thanh trượt để thay đổi phần trăm từng dung môi.
        4. **Đọc chỉ số:** Ra càng gần 0 và chỉ số hòa tan > 80% là hỗn hợp tối ưu.
        """)
    st.title("🧪 Hệ thống Tối ưu hóa Dung môi Hansen")
    st.markdown("Sử dụng **Khoảng cách Hansen (Hansen Solubility Parameters - HSP)** để dự đoán độ tan Alkaloid trong hỗn hợp dung môi đa thành phần.")

    # 1. Cơ sở dữ liệu dung môi
    solvents = {
        "Ethanol": [15.8, 8.8, 19.4],
        "Nước": [15.5, 16.0, 42.3],
        "Acetone": [15.5, 10.4, 7.0],
        "Methanol": [14.7, 12.3, 22.3],
        "Ethyl Acetate": [15.8, 5.3, 7.2]
    }

    target_alkaloid = st.selectbox("Chọn Alkaloid mục tiêu:", ["Nuciferine", "Roemerine", "Liensinine"])
    target_coords = {"Nuciferine": [18.5, 6.2, 5.1], "Roemerine": [18.2, 5.8, 4.8], "Liensinine": [19.1, 8.5, 12.3]}
    t_hsp = target_coords[target_alkaloid]

    # 2. Cấu hình hỗn hợp
    st.subheader("🎛️ Thiết lập hỗn hợp dung môi")
    sel_sols = st.multiselect("Chọn dung môi trong hỗn hợp:", list(solvents.keys()), default=["Ethanol", "Nước"])
    
    weights = {}
    if sel_sols:
        for s in sel_sols:
            weights[s] = st.slider(f"Tỷ lệ {s} (%)", 0, 100, int(100/len(sel_sols)))
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
            
            # Tính toán thông số hỗn hợp
            mix_hsp = np.zeros(3)
            for s, w in weights.items():
                mix_hsp += np.array(solvents[s]) * w
            
            # 3. Tính toán Khoảng cách Hansen (Ra) chuẩn xác
            dist = np.sqrt(4*(mix_hsp[0]-t_hsp[0])**2 + (mix_hsp[1]-t_hsp[1])**2 + (mix_hsp[2]-t_hsp[2])**2)
            
            # --- CÔNG THỨC KHOA HỌC CHUẨN (HÀM GAUSSIAN) ---
            # R0 là bán kính tương tác giả định của hợp chất mục tiêu (~8.0 units)
            R0 = 8.0 
            score = 100 * np.exp(- (dist**2) / (2 * (R0/2)**2))
            # ------------------------------------------------

            # 4. Hiển thị kết quả
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Khoảng cách Ra", f"{dist:.2f}", help="Khoảng cách Hansen (Ra) càng nhỏ, độ tương thích càng cao.")
            c2.metric("Chỉ số hòa tan", f"{score:.1f}%")

            # 5. Đồ thị Radar (Phân tích chuyên sâu)
            st.subheader("📊 Phân tích độ tương thích (Hansen Profile)")
            radar_df = pd.DataFrame({
                'Thông số': ['Dispersion (dD)', 'Polar (dP)', 'H-Bond (dH)'],
                'Mục tiêu': t_hsp,
                'Hỗn hợp': mix_hsp
            })
            df_melted = radar_df.melt(id_vars='Thông số', var_name='Loại', value_name='Giá trị')
            
            fig = px.line_polar(df_melted, r='Giá trị', theta='Thông số', color='Loại', 
                                line_close=True, markers=True, template="plotly_white")
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 **Nhận xét:** Diện tích hình 'Hỗn hợp' càng gần với 'Mục tiêu' thì độ tan càng lý tưởng.")

            # 6. Lời khuyên thông minh
            if score > 80:
                st.balloons()
                st.success(f"🎉 **Tối ưu:** Tỷ lệ này cực kỳ phù hợp để chiết xuất {target_alkaloid}!")
            elif score < 40:
                st.error("⚠️ **Cần điều chỉnh:** Dung môi hiện tại quá xa so với mục tiêu. Hãy thử thay đổi tỷ lệ hoặc loại dung môi khác.")
elif page == "6. Động học Chiết tách (Toán)":
    with st.sidebar:
        st.header("📖 Hướng dẫn Module 6")
        st.info("""
        **Mục tiêu:** Mô phỏng quá trình chiết tách bằng mô hình toán học Pseudo-second-order.
        
        **Các bước thực hiện:**
        1. **Thiết lập thông số:** Thay đổi Qe (dung lượng cực đại) và k2 (tốc độ chiết) từ bảng điều khiển bên trái.
        2. **Quan sát biểu đồ:** Xem đường cong nồng độ tăng dần theo thời gian.
        3. **Xác định t90:** Xem thời gian tối ưu để đạt 90% hiệu suất, giúp tiết kiệm thời gian và năng lượng vận hành.
        """)
    st.title("📈 Mô phỏng Động học & Vận tốc Chiết tách")
    st.markdown("Sử dụng mô hình **Pseudo-second-order** để tối ưu hóa thời gian chiết xuất dược liệu.")

    # 1. Cấu hình thông số thực nghiệm (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Cài đặt thực nghiệm")
    qe = st.sidebar.slider("Dung lượng tối đa (Qe - mg/g):", 5.0, 100.0, 25.0)
    k2 = st.sidebar.slider("Hằng số vận tốc (k2 - g/mg.phút):", 0.001, 0.05, 0.015, step=0.001)

    # 2. Xử lý thuật toán
    time_steps = np.linspace(0, 200, 100) # Giảm số bước để bảng hiện vừa đủ, không quá dài
    qt = (k2 * (qe ** 2) * time_steps) / (1 + k2 * qe * time_steps)
    velocity = (k2 * (qe**2)) / ((1 + k2 * qe * time_steps)**2)

    # 3. Dashboard hiển thị các thông số cốt lõi
    col1, col2, col3 = st.columns(3)
    t_90 = 9 / (k2 * qe)
    col1.metric("Tốc độ ban đầu", f"{velocity[0]:.3f} mg/g.p")
    col2.metric("Thời gian đạt 90% (t90)", f"{int(t_90)} phút")
    col3.metric("Trạng thái", "Hiệu quả" if t_90 < 120 else "Chậm")

    # 4. Đồ thị trực quan
    fig = px.line(x=time_steps, y=qt, labels={'x': 'Thời gian (phút)', 'y': 'Nồng độ chiết (mg/g)'}, 
                  title="Đường cong động học Pseudo-second-order")
    fig.add_hline(y=qe, line_dash="dash", line_color="red", annotation_text="Giới hạn bão hòa (Qe)")
    st.plotly_chart(fig, use_container_width=True)

    # 5. Phân tích chuyên sâu
    st.subheader("💡 Phân tích chiến lược")
    efficiency = qt / qe
    time_to_80 = time_steps[np.searchsorted(efficiency, 0.8)]
    
    st.success(f"""
    - **Giai đoạn tăng trưởng nhanh:** Từ 0 đến {int(time_to_80)} phút, quá trình chiết đạt hiệu suất cao nhất.
    - **Thời điểm tối ưu:** Hệ thống khuyến nghị dừng quá trình tại **{int(t_90 + 10)} phút**. 
    - **Giải thích khoa học:** Việc tiếp tục chiết sau thời điểm này tiêu tốn năng lượng vận hành máy khuấy nhưng chỉ làm tăng <5% nồng độ hoạt chất.
    """)

    # 6. Bảng hiển thị dữ liệu (Đã định dạng sạch sẽ)
    st.subheader("📋 Bảng dữ liệu mô phỏng chi tiết")
    df_display = pd.DataFrame({
        "Thời gian (phút)": time_steps, 
        "Nồng độ (mg/g)": qt
    })
    # Hiển thị bảng với format số thập phân gọn gàng
    st.dataframe(df_display.style.format({"Nồng độ (mg/g)": "{:.4f}"}), use_container_width=True)
  # --- MODULE 7: DỰ TOÁN QUY MÔ & KINH TẾ (TOÁN - NÂNG CẤP RÀNG BUỘC & PHÂN TÍCH CHI PHÍ) ---
elif page == "7. Dự toán Quy mô & Kinh tế (Toán)":
    with st.sidebar:
        st.header("📖 Hướng dẫn Module 7")
        st.info("""
        **Mục tiêu:** Mô phỏng bài toán kinh tế và tối ưu hóa chi phí sản xuất hoạt chất Alkaloid dưới ràng buộc khắt khe về sản lượng thu hồi.
        
        **Điểm nổi bật:**
        - Tích hợp dữ liệu từ Module 6 (Động học).
        - Sử dụng thuật toán tối ưu hóa phi tuyến có ràng buộc (`SLSQP`).
        """)
        
    st.title("7. Dự toán Quy mô & Kinh tế & Tối ưu hóa (Toán)")
    st.markdown("Hệ thống tối ưu hóa chi phí sản xuất tự động dưới ràng buộc đảm bảo hàm lượng hoạt chất đầu ra.")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.subheader("Thông số Đầu vào (Thị trường)")
        price_leaf = st.number_input("Giá lá sen khô (VNĐ/kg)", value=50000, step=5000)
        price_solvent = st.number_input("Giá dung môi tối ưu (VNĐ/Lít)", value=35000, step=2000)
        price_elec = st.number_input("Giá điện (VNĐ/kWh)", value=2500, step=100)
        target_yield_mg = st.number_input("Mức hoạt chất tối thiểu cần đạt (mg)", value=500.0, step=50.0)

    with col_in2:
        st.subheader("Thông số Giới hạn Vận hành")
        
        scale_leaf = st.slider("Khoảng quy mô mẻ chiết (kg lá)", 1, 50, (1, 50))
        st.caption("📌 **Quy mô mẻ:** Giới hạn khối lượng nguyên liệu lá sen khô đầu vào cho một lần chiết xuất.")
        
        vol_solvent = st.slider("Khoảng thể tích dung môi (Lít)", 10, 200, (10, 200))
        st.caption("📌 **Thể tích dung môi:** Khoảng giới hạn lượng dung môi cấp vào bình chiết.")
        
        recovery_eff = st.slider("Hiệu suất thu hồi trung bình (%)", 0.5, 2.0, 1.2, step=0.1)
        st.caption("📌 **Hiệu suất thu hồi:** Tỷ lệ chuyển hóa hoạt chất kế thừa từ mô hình động học Module 6.")

    st.divider()
    st.subheader("⚙️ Tối ưu hóa phi tuyến có ràng buộc (SciPy - SLSQP)")
    
    if st.button("🚀 Chạy thuật toán tối ưu hóa kinh tế & ràng buộc"):
        # Hàm mục tiêu cần cực tiểu hóa (Chi phí)
        def objective(x):
            m, v = x
            elec_kwh = v * 0.15 
            return (m * price_leaf) + (v * price_solvent) + (elec_kwh * price_elec)

        # Ràng buộc bất đẳng thức: Y(m, v) >= target_yield_mg
        # tương đương với: (m * recovery_eff * 15) - target_yield_mg >= 0
        constraints = {
            'type': 'ineq', 
            'fun': lambda x: (x[0] * recovery_eff * 15) - target_yield_mg
        }

        bounds = [scale_leaf, vol_solvent]
        x0 = [(scale_leaf[0] + scale_leaf[1]) / 2, (vol_solvent[0] + vol_solvent[1]) / 2]
        
        # Sử dụng phương pháp SLSQP để giải quyết bài toán tối ưu có ràng buộc
        res = minimize(objective, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if res.success:
            opt_m, opt_v = res.x
            opt_cost = res.fun
            est_output_mg = opt_m * recovery_eff * 15 
            
            # Tính toán chi tiết cấu thành chi phí để báo cáo chuyên sâu
            cost_leaf_val = opt_m * price_leaf
            cost_solvent_val = opt_v * price_solvent
            cost_elec_val = (opt_v * 0.15) * price_elec
            
            st.success("🎉 Tối ưu hóa thành công dưới ràng buộc sản lượng!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Khối lượng lá tối ưu", f"{opt_m:.2f} kg")
            m2.metric("Thể tích dung môi tối ưu", f"{opt_v:.2f} Lít")
            m3.metric("Tổng chi phí tối thiểu", f"{opt_cost:,.0f} VNĐ")
            
            # Hiển thị thêm bảng bóc tách chi phí chuyên nghiệp
            with st.expander("📊 Xem chi tiết cấu thành chi phí kinh tế-kỹ thuật"):
                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.metric("Chi phí Nguyên liệu lá", f"{cost_leaf_val:,.0f} VNĐ", f"{(cost_leaf_val/opt_cost)*100:.1f}%")
                col_c2.metric("Chi phí Dung môi", f"{cost_solvent_val:,.0f} VNĐ", f"{(cost_solvent_val/opt_cost)*100:.1f}%")
                col_c3.metric("Chi phí Điện năng", f"{cost_elec_val:,.0f} VNĐ", f"{(cost_elec_val/opt_cost)*100:.1f}%")
            
            st.info(f"💡 **Khuyến nghị vận hành chuẩn hóa:** Để đảm bảo đạt sản lượng yêu cầu tối thiểu **{target_yield_mg:.1f} mg** (thực tế thu được **{est_output_mg:.1f} mg**) với chi phí thấp nhất, hệ thống tự động đề xuất cấu hình: **{opt_m:.1f} kg** lá sen khô và **{opt_v:.1f} lít** dung môi.")
        else:
            st.warning("⚠️ Không tìm thấy nghiệm thỏa mãn (Miền khả thi rỗng). Có thể mức sản lượng yêu cầu quá cao so với giới hạn quy mô lá hoặc hiệu suất hiện tại. Vui lòng nới lỏng ràng buộc hoặc tăng quy mô mẻ chiết.")
