import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ App Config ------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------ Theming Helpers ------------------------
def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --radius: 12px;
                --card-bg: #0e1117;
                --muted: #8b949e;
                --border: #2d333b;
                --accent: #3b82f6; /* blue-500 */
            }
            .app-header h1 {
                font-size: 1.9rem;
                margin-bottom: 0.25rem;
            }
            .app-subtitle {
                color: var(--muted);
                margin-bottom: 1.25rem;
            }
            .card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 18px;
            }
            .metric-card {
                background: linear-gradient(180deg, rgba(59,130,246,0.12), rgba(59,130,246,0.06));
                border: 1px solid rgba(59,130,246,0.25);
            }
            .footer-note {
                color: var(--muted);
                font-size: 0.9rem;
            }
            .hint {
                color: var(--muted);
                font-size: 0.85rem;
                margin-top: -6px;
                margin-bottom: 8px;
            }
            .divider {
                border-top: 1px dashed var(--border);
                margin: 12px 0 18px 0;
            }
            .label-compact .st-emotion-cache-10trblm, 
            .label-compact .st-emotion-cache-1djdyxw {
                margin-bottom: 2px !important;
            }
            .success-callout {
                padding: 12px 14px;
                border-radius: 10px;
                border: 1px solid rgba(34,197,94,0.25);
                background: rgba(34,197,94,0.08);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_css()

# ------------------------ Load Artifacts ------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_encoders():
    model = pickle.load(open("laptop_model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    return model, encoders

try:
    model, encoders = load_model_and_encoders()
except Exception as e:
    st.error("Failed to load model or encoders. Ensure 'laptop_model.pkl' and 'encoders.pkl' exist.")
    st.exception(e)
    st.stop()

# ------------------------ Header ------------------------
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.title("💻 Laptop Price Predictor")
st.markdown('<div class="app-subtitle">Estimate price from detailed specifications, with a clean and focused UI.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.header("🔧 Specifications")
    st.caption("Fill in key attributes. Hover labels for hints.")

    # Compact labels
    label_kw = {"help": "Select from the trained categories."}

    company = st.selectbox("Company", encoders['Company'].classes_, **label_kw)
    typename = st.selectbox("Type", encoders['TypeName'].classes_, help="Form factor/type (Ultrabook, Gaming, etc.)")
    opsys = st.selectbox("Operating System", encoders['OpSys'].classes_, help="OS family used for training categories.")
    cpu_model = st.selectbox("CPU Model", encoders['CpuModel'].classes_, help="Exact CPU model category.")
    gpu_model = st.selectbox("GPU Model", encoders['GpuModel'].classes_, help="Exact GPU model category.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        inches = st.number_input("Screen Size (inches)", 10.0, 20.0, step=0.1, help="Diagonal screen size.")
        weight = st.number_input("Weight (kg)", 0.5, 5.0, step=0.1, help="Weight in kilograms.")
        ram = st.number_input("RAM (GB)", 2.0, 64.0, step=2.0, help="System RAM.")
    with c2:
        cpu_speed = st.number_input("CPU Speed (GHz)", 0.9, 4.5, step=0.1, help="Base or typical turbo-range reference.")
        memory_gb = st.number_input("Primary Storage (GB)", 8.0, 2048.0, step=8.0, help="Main drive capacity.")
        extra_memory_gb = st.number_input("Extra Storage (GB)", 0.0, 2048.0, step=128.0, help="Secondary drive capacity.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    screen = st.selectbox("Panel Type", ["IPS Panel", "IPS Panel Retina Display"], help="Panel family used in training.")
    screen_quality = st.selectbox("Screen Quality", ["Full HD", "4K Ultra HD", "Quad HD+"], help="Resolution class.")
    total_pixels = st.selectbox(
        "Resolution (total pixels)",
        [2073600,1049088,8294400,5760000,3686400,1440000,3317760,4096000,3393024,2304000,1296000,5184000,3840000,3110400,4990464],
        help="H×W pixel count; select closest match."
    )
    is_touchscreen = st.selectbox("Touchscreen", [0, 1], help="0 = No, 1 = Yes")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    memory_type = st.selectbox("Primary Storage Type", ["SSD","HDD","Hybrid","Flash"])
    extra_memory_type = st.selectbox("Extra Storage Type", ["HDD","SSD","Hybrid","None"], help="Type of second drive.")
    gpu_brand = st.selectbox("GPU Brand", ["Intel","Nvidia","AMD","ARM"])
    cpu_brand = st.selectbox("CPU Brand", ["Intel","AMD","Samsung"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    show_demo_charts = st.toggle("Show demo visuals", value=False, help="Enables example charts with dummy data.")

    predict_clicked = st.button("💰 Predict Price", use_container_width=True)

# ------------------------ Encodings ------------------------
extra_memory_type_map = {'None':0,"HDD":1, "SSD":2, "Hybrid":3}
screen_panel_map = {'IPS Panel Retina Display':1, 'IPS Panel':2}
memory_type_map = {'HDD':1, 'SSD':2, 'Hybrid':3,'Flash':4}
gpu_brand_map = {'Intel':1, 'AMD':2, 'Nvidia':4,'ARM':5}
cpu_brand_map = {"Intel":1, "AMD":2, "Samsung":3}
screen_quality_map = {'Full HD':1, 'Quad HD+':2, '4K Ultra HD':3}

company_encoded = encoders['Company'].transform([company])[0]
typename_encoded = encoders['TypeName'].transform([typename])[0]
opsys_encoded = encoders['OpSys'].transform([opsys])[0]
cpu_model_encoded = encoders['CpuModel'].transform([cpu_model])[0]
gpu_model_encoded = encoders['GpuModel'].transform([gpu_model])[0]

extra_memory_type_encoded = extra_memory_type_map[extra_memory_type]
screen_encoded = screen_panel_map[screen]
memory_type_encoded = memory_type_map[memory_type]
gpu_brand_encoded = gpu_brand_map[gpu_brand]
cpu_brand_encoded = cpu_brand_map[cpu_brand]
screen_quality_encoded = screen_quality_map[screen_quality]

# ------------------------ Input Validation ------------------------
def validate_inputs():
    errors = []
    if memory_gb == 0 and extra_memory_gb == 0:
        errors.append("Storage cannot be 0 for both primary and extra.")
    if inches < 10 or inches > 20:
        errors.append("Screen size out of supported range.")
    if ram % 2 != 0:
        errors.append("RAM should be an even number in GB (2,4,8,16,...).")
    return errors

# ------------------------ Prediction ------------------------
if predict_clicked:
    errs = validate_inputs()
    if errs:
        for e in errs:
            st.warning(e)
    else:
        features = np.array([[
            company_encoded, typename_encoded, inches, opsys_encoded,
            extra_memory_type_encoded, extra_memory_gb, memory_gb, screen_encoded,
            weight, ram, memory_type_encoded, gpu_brand_encoded, cpu_brand_encoded,
            cpu_speed, total_pixels, gpu_model_encoded, cpu_model_encoded,
            is_touchscreen, screen_quality_encoded
        ]])

        prediction = model.predict(features)[0]

        # Layout: result + breakdown
        r1, r2 = st.columns([1,1])
        with r1:
            st.markdown("### Result")
            st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="success-callout">
                    <div style="font-size: 0.95rem; color:#16a34a; margin-bottom:6px;">Prediction successful</div>
                    <div style="font-size: 2rem; font-weight: 700;">€{prediction:.2f}</div>
                    <div style="color:#9ca3af; font-size: 0.9rem;">Estimated laptop price</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown("### Quick Summary")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Company", company)
            k2.metric("Type", typename)
            k3.metric("OS", opsys)
            k4, k5, k6 = st.columns(3)
            k4.metric("CPU Brand", cpu_brand)
            k5.metric("GPU Brand", gpu_brand)
            k6.metric("RAM", f"{int(ram)} GB")
            k7, k8, k9 = st.columns(3)
            k7.metric("Storage", f"{int(memory_gb)} GB {memory_type}")
            k8.metric("Extra", f"{int(extra_memory_gb)} GB {extra_memory_type}" if extra_memory_gb > 0 else "—")
            k9.metric("Display", f"{screen_quality} • {int(inches)}\"")
            st.markdown('</div>', unsafe_allow_html=True)

        # Optional demo visuals (clearly labeled)
    if show_demo_charts:
        st.markdown("### Visuals (demo)")
        st.caption("These charts use example data for visualization only — not from your trained dataset.")

        # Compact layout: 2 columns, then 1 full-width
        c1, c2 = st.columns([1,1])

        # 1) Predicted vs Avg (Bar) — rainbow colors + smaller figure
        with c1:
            avg_price_company = np.random.randint(600, 2000)
            data = pd.DataFrame({
                "Category":["Predicted Price", f"Average Price ({company})"],
                "Price":[prediction, avg_price_company]
            })

            # Use matplotlib for tighter control instead of st.bar_chart
            fig_bar, ax_bar = plt.subplots(figsize=(3.6, 2.4), dpi=150)  # smaller size
            categories = data["Category"].values
            values = data["Price"].values

            # Rainbow colors (HSV across N items)
            N = len(categories)
            rainbow = [plt.cm.hsv(i / max(N, 1)) for i in range(N)]

            ax_bar.bar(categories, values, color=rainbow)
            ax_bar.set_ylabel("Price (€)")
            ax_bar.set_title("Predicted vs Company Avg", fontsize=10)
            ax_bar.tick_params(axis='x', labelrotation=15)
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=False)

        # 2) RAM Distribution (Pie) — rainbow + smaller
        with c2:
            ram_dist = pd.DataFrame({
                "RAM":[4,8,16,32],
                "Count":[120,300,200,50]
            })
            counts = ram_dist["Count"].values
            labels = ram_dist["RAM"].astype(str).values

            N = len(labels)
            rainbow = [plt.cm.hsv(i / max(N, 1)) for i in range(N)]

            fig_pie, ax_pie = plt.subplots(figsize=(3.6, 2.4), dpi=150)  # smaller
            wedges, texts, autotexts = ax_pie.pie(
                counts,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=rainbow,
                textprops={"fontsize":8}
            )
            ax_pie.set_title("RAM Distribution (demo)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_pie, use_container_width=False)

        # 3) CPU Speed vs Price (Line) — rainbow gradient along line + smaller
        cpu_speed_data = pd.DataFrame({
            "CPU Speed (GHz)":[1.5,2.0,2.5,3.0,3.5,4.0],
            "Avg Price":[400,600,900,1400,2000,2500]
        })

        x = cpu_speed_data["CPU Speed (GHz)"].values
        y = cpu_speed_data["Avg Price"].values

        fig_line, ax_line = plt.subplots(figsize=(7.6, 2.6), dpi=150)  # compact wide
        # Create segment-wise rainbow colors
        from matplotlib.collections import LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color each segment using HSV gradient
        cmap = plt.cm.hsv
        colors = cmap(np.linspace(0, 1, len(segments)))
        lc = LineCollection(segments, colors=colors, linewidths=2)
        ax_line.add_collection(lc)
        ax_line.set_xlim(x.min(), x.max())
        ax_line.set_ylim(min(y)*0.95, max(y)*1.05)
        ax_line.set_xlabel("CPU Speed (GHz)")
        ax_line.set_ylabel("Avg Price (€)")
        ax_line.set_title("CPU Speed vs Price (demo)", fontsize=10)
        ax_line.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_line, use_container_width=False)


# ------------------------ Footer ------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="footer-note">Tip: Use the toggle to enable demo visuals. Replace them with real aggregates from your dataset for production.</div>', unsafe_allow_html=True)
