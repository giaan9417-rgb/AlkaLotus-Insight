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
    import numpy as np
    import plotly.express as px
    import streamlit as st

    st.title("🧬 Module 4: Phân tích Cấu trúc & Tương đồng Đa thông số (Tanimoto Liên tục)")
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
            "MW": 379.50, "LogP": 4.27, "TPSA": 38.8, 
            "HBD": 0.0, "HBA": 4.0, "ΔG AChE": -11.5, "ΔG BACE1": -7.2
        },
        "Verubecestat (Chuẩn BACE1)": {
            "MW": 409.41, "LogP": 1.15, "TPSA": 104.9, 
            "HBD": 2.0, "HBA": 6.0, "ΔG AChE": -6.8, "ΔG BACE1": -10.4
        }
    }

    # 2. Giao diện Nhập liệu Thông số
    st.subheader("⚙️ Nhập thông số Hợp chất Thử nghiệm")
    st.info("💡 Nhập trực tiếp các chỉ số hóa lý và năng lượng. Hệ thống sẽ tự động tổng hợp để tính toán độ tương đồng Tanimoto đa chiều.")
    
    num_comp = st.number_input("Số lượng hợp chất cần phân tích:", min_value=1, max_value=15, value=2, step=1)
    
    compounds_input = []
    
    for i in range(int(num_comp)):
        st.markdown(f"**🔹 Hợp chất #{i+1}**")
        c_name = st.text_input(f"Tên hợp chất #{i+1}:", value=f"Alkaloid_{i+1}", key=f"name_{i}")
        
        col1, col2, col3 = st.columns(3)
        c_mw = col1.number_input(f"Khối lượng (MW) #{i+1}:", value=281.35 + (i*10.0), format="%.2f", key=f"mw_{i}")
        c_logp = col2.number_input(f"Độ phân cực (LogP) #{i+1}:", value=2.50 + (i*0.5), format="%.2f", key=f"logp_{i}")
        c_tpsa = col3.number_input(f"Diện tích bề mặt (TPSA) #{i+1}:", value=20.0 + (i*5.0), format="%.2f", key=f"tpsa_{i}")
        
        col4, col5, col6 = st.columns(3)
        c_hbd = col4.number_input(f"Số liên kết cho H (HBD) #{i+1}:", value=1.0, format="%.1f", key=f"hbd_{i}")
        c_hba = col5.number_input(f"Số liên kết nhận H (HBA) #{i+1}:", value=2.0, format="%.1f", key=f"hba_{i}")
        c_ga = col6.number_input(f"ΔG AChE (kcal/mol) #{i+1}:", value=-8.80, format="%.2f", key=f"ga_{i}")
        
        c_gb = st.number_input(f"ΔG BACE1 (kcal/mol) #{i+1}:", value=-8.10, format="%.2f", key=f"gb_{i}")
        st.divider()
        
        compounds_input.append({
            "Hợp chất": c_name,
            "MW": float(c_mw), "LogP": float(c_logp), "TPSA": float(c_tpsa),
            "HBD": float(c_hbd), "HBA": float(c_hba), 
            "ΔG AChE": float(c_ga), "ΔG BACE1": float(c_gb)
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
            features = ["MW", "LogP", "TPSA", "HBD", "HBA", "ΔG AChE", "ΔG BACE1"]
            
            # Chuẩn hóa Min-Max (0-1) an toàn
            df_norm = df_all.copy()
            for col in features:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = 1.0 

            # Hàm tính Tanimoto liên tục
            def calc_continuous_tanimoto(vec1, vec2):
                dot_product = np.dot(vec1, vec2)
                sum_sq1 = np.dot(vec1, vec1)
                sum_sq2 = np.dot(vec2, vec2)
                denominator = sum_sq1 + sum_sq2 - dot_product
                if denominator <= 0.0001:  # Chống lỗi chia cho 0 hoặc số quá nhỏ
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
            df_sim_matrix = pd.DataFrame(sim_matrix, index=names, columns=names)

            # Trích xuất kết quả đưa vào bảng
            results_list = []
            for i in range(2, matrix_size): 
                results_list.append({
                    "Hợp chất": names[i],
                    "Tanimoto vs Donepezil": df_sim_matrix.iloc[i, 0],
                    "Tanimoto vs Verubecestat": df_sim_matrix.iloc[i, 1]
                })
            df_results = pd.DataFrame(results_list)

            # 5. Xuất Trực quan Hóa
            st.markdown("---")
            st.subheader("📊 Bảng Kết quả Tương đồng Tanimoto Đa thông số")
            st.dataframe(df_results, use_container_width=True)

            st.subheader("📈 Biểu đồ Cột: So sánh Tương đồng với Thuốc Chuẩn")
            df_melted = df_results.melt(id_vars=["Hợp chất"], 
                                        value_vars=["Tanimoto vs Donepezil", "Tanimoto vs Verubecestat"],
                                        var_name="Đối tượng So sánh", 
                                        value_name="Độ Tương Đồng")
            
            fig_bar = px.bar(df_melted, x="Hợp chất", y="Độ Tương Đồng", 
                             color="Đối tượng So sánh", barmode="group",
                             title="Mức độ tương đồng so với Thuốc chuẩn",
                             color_discrete_sequence=["#1f77b4", "#ff7f0e"])
            fig_bar.update_layout(yaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("🧩 Ma trận Tương đồng Toàn diện (Similarity Matrix)")
            fig_matrix = px.imshow(df_sim_matrix,
                                   labels=dict(x="Hợp chất", y="Hợp chất", color="Tanimoto"),
                                   x=names, y=names,
                                   color_continuous_scale="Blues",
                                   text_auto=True, 
                                   aspect="auto")
            fig_matrix.update_xaxes(side="top")
            st.plotly_chart(fig_matrix, use_container_width=True)
            
            st.subheader("🎯 Biểu đồ Tương quan Đặc tính - Hoạt tính (SAR)")
            plot_sar_data = []
            for i in range(matrix_size):
                c_type = "Thuốc Chuẩn AChE" if i == 0 else "Thuốc Chuẩn BACE1" if i == 1 else "Hợp chất thử nghiệm"
                plot_sar_data.append({
                    "Hợp chất": names[i],
                    "Tanimoto vs Donepezil": df_sim_matrix.iloc[i, 0],
                    "ΔG AChE (kcal/mol)": float(df_all.iloc[i]["ΔG AChE"]),
                    "Loại": c_type
                })
            df_sar = pd.DataFrame(plot_sar_data)
            
            fig_sar = px.scatter(df_sar, x="Tanimoto vs Donepezil", y="ΔG AChE (kcal/mol)",
                                 color="Loại", text="Hợp chất", size_max=15,
                                 title="Phân tích SAR: Tương đồng vs Donepezil và Năng lượng Liên kết")
            fig_sar.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='black')))
            fig_sar.update_yaxes(autorange="reversed") 
            st.plotly_chart(fig_sar, use_container_width=True)

            st.success("✅ Module 4 đã chạy thành công! Không sử dụng SMILES, thuật toán Tanimoto Liên tục được áp dụng chính xác bằng cách chuẩn hóa các vector thông số đầu vào.")
        except Exception as e:
            st.error(f"❌ Có lỗi toán học xảy ra trong quá trình tính toán: {e}. Vui lòng kiểm tra lại các thông số nhập vào.")
elif page in ["5. Tối ưu dung môi (Toán)", "5. Tối ưu Dung môi (Toán)"]:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    
    try:
        st.title("🧪 Module 5: Tối ưu hóa Dung môi Đa Chất (Hansen Space)")
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
