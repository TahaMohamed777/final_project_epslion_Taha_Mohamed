import streamlit as st
import pandas as pd
import joblib
import base64
from sklearn.preprocessing import FunctionTransformer
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# ================= Background Image Function =================
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #fff;
        font-family: 'Segoe UI', sans-serif;
    }}
    
    /* Overlay for better text readability */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }}
    
    .block-container {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 25px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(44, 62, 80, 0.9), rgba(76, 161, 175, 0.9));
        color: white;
        backdrop-filter: blur(10px);
    }}
    
    .big-title {{ 
        font-size: 42px; 
        text-align:center; 
        font-weight:bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }}
    
    .sub-title {{ 
        text-align:center; 
        font-size:18px; 
        color:#ddd; 
        margin-bottom:30px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ================= Feature Engineering =================
def feature_engineering(df):
    df = df.copy()
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,30,50,100], labels=['Young','Adult','Senior'])
    df['Tenure_per_Age'] = df['Tenure'] / (df['Age'] + 1)
    df['Balance_per_Product'] = df['Balance'] / (df['NumOfProducts'] + 1)
    df['HighSalary'] = (df['EstimatedSalary'] > df['EstimatedSalary'].median()).astype(int)
    return df

feature_engineer = FunctionTransformer(feature_engineering)

# ================= Page Config =================
st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

# Set background image
set_png_as_page_bg("C:\\Users\\Taha mohamed\\OneDrive - Egyptian Chinese University (ECU)\\Epslion\\machine learning\\project3\\background\\photo-1565638459249-c85cbb2faaa8.jpeg")

# باقي الكود يبقى نفسه...
# ================= Load Data & Model =================
df = pd.read_csv("cleaned_data.csv")
model = joblib.load("stacking_model.pkl")

# ================= NAVIGATION SYSTEM =================
pages = ["🏠 Home","🔮 Prediction","ℹ️ About Data","📊 Analysis","🧩 EDA","💡 Recommendations"]

# Initialize session state if not exists
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

# ==== Custom CSS for Sidebar Style ====
# ==== Custom Title CSS ====
st.markdown("""
<style>
.big-title {
    font-size: 42px; 
    text-align:center; 
    font-weight:normal; 
    background: linear-gradient(135deg,#00c6ff,#0072ff); 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align:center; 
    font-size:20px;
    font-weight:normal;
    margin-bottom:20px; 
    background: linear-gradient(135deg,#f12711,#f5af19); 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subheadings */
h2, .css-10trblm, .css-1d391kg {
    color: #76EEC6 !important;  /* Light Cyan */
}
</style>
""", unsafe_allow_html=True)

# ==== Sidebar content ====
st.sidebar.markdown("## 📌 Navigation")
choice = st.sidebar.radio("", pages, index=pages.index(st.session_state["page"]))
st.session_state["page"] = choice  # Sync sidebar with session_state

# ==== Extra Info in Sidebar ====
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.write(f"**Total Customers:** {len(df)}")
st.sidebar.write(f"**Churn Rate:** {df['Exited'].mean()*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 Last Prediction")
if "last_pred" in st.session_state:
    if st.session_state["last_pred"] == 1:
        st.sidebar.error(f"⚠️ Will Churn ({st.session_state['last_proba']:.0%})")
    else:
        st.sidebar.success(f"✅ Stay ({(1-st.session_state['last_proba']):.0%})")
else:
    st.sidebar.info("No prediction yet.")
# ================= HOME PAGE =================
if st.session_state["page"] == "🏠 Home":
    # ====== Titles with Gradient Color ======
    st.markdown("""
    <style>
    .big-title {
        font-size: 42px; 
        text-align:center; 
        font-weight:normal; 
        background: linear-gradient(135deg,#f12711,#f5af19);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-title {
        text-align:center; 
        font-size:18px; 
        margin-bottom:30px; 
        background: linear-gradient(135deg,#f12711,#f5af19);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='big-title'>Bank Churn Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Predict • Analyze • Explore • Recommend</div>", unsafe_allow_html=True)

    # ✅ مركز الصورة + تصغير الحجم
    st.markdown("""
    <div style='display:flex; justify-content:center; margin:20px 0;'>
        <img src="https://cdn.dribbble.com/userupload/42177612/file/original-6b80f731604a3ee61165320f9acaf931.gif" 
             width="900">
    </div>
    """, unsafe_allow_html=True)

    # ==== CSS for Colored Buttons ====
    st.markdown("""
    <style>
    .stButton > button {
        padding: 18px 30px;
        margin: 10px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        min-width: 180px;
        text-align: center;
    }
    .stButton > button:hover {
        transform: scale(1.07);
        box-shadow: 0px 8px 18px rgba(0,0,0,0.4);
    }
    /* Assign colors by order */
    div.stButton:nth-of-type(1) > button {background: linear-gradient(135deg,#f12711,#f5af19);} /* Prediction */
    div.stButton:nth-of-type(2) > button {background: linear-gradient(135deg,#00b09b,#96c93d);} /* About */
    div.stButton:nth-of-type(3) > button {background: linear-gradient(135deg,#2193b0,#6dd5ed);} /* Analysis */
    div.stButton:nth-of-type(4) > button {background: linear-gradient(135deg,#cc2b5e,#753a88);} /* EDA */
    div.stButton:nth-of-type(5) > button {background: linear-gradient(135deg,#42275a,#734b6d);} /* Recommendations */
    </style>
    """, unsafe_allow_html=True)

    # ==== Buttons Row ====
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🔮 Prediction"):
            st.session_state["page"] = "🔮 Prediction"

    with col2:
        if st.button("ℹ️ About Data"):
            st.session_state["page"] = "ℹ️ About Data"

    with col3:
        if st.button("📊 Analysis"):
            st.session_state["page"] = "📊 Analysis"

    with col4:
        if st.button("🧩 EDA"):
            st.session_state["page"] = "🧩 EDA"

    with col5:
        if st.button("💡 Recommendations"):
            st.session_state["page"] = "💡 Recommendations"
# ================= PREDICTION =================
# ================= PREDICTION =================
elif choice == "🔮 Prediction":
    st.title("🔮 Predict Customer Churn")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            CreditScore = st.number_input("Credit Score", int(df.CreditScore.min()), int(df.CreditScore.max()), int(df.CreditScore.median()))
            Age = st.number_input("Age", int(df.Age.min()), int(df.Age.max()), int(df.Age.median()))
            Tenure = st.number_input("Tenure", int(df.Tenure.min()), int(df.Tenure.max()), int(df.Tenure.median()))
            Balance = st.number_input("Balance", float(df.Balance.min()), float(df.Balance.max()), float(df.Balance.median()))
        with col2:
            NumOfProducts = st.number_input("Num Of Products", int(df.NumOfProducts.min()), int(df.NumOfProducts.max()), int(df.NumOfProducts.median()))
            HasCrCard = st.selectbox("Has Credit Card?", [0,1])
            IsActiveMember = st.selectbox("Active Member?", [0,1])
            EstimatedSalary = st.number_input("Estimated Salary", float(df.EstimatedSalary.min()), float(df.EstimatedSalary.max()), float(df.EstimatedSalary.median()))
            Geography = st.selectbox("Geography", df.Geography.unique())
            Gender = st.selectbox("Gender", df.Gender.unique())
        
        submit = st.form_submit_button("🔮 Predict Churn")

    if submit:
        # 1️⃣ Prepare input data
        input_data = pd.DataFrame([{
            "CreditScore": CreditScore, 
            "Age": Age, 
            "Tenure": Tenure,
            "Balance": Balance, 
            "NumOfProducts": NumOfProducts,
            "HasCrCard": HasCrCard, 
            "IsActiveMember": IsActiveMember,
            "EstimatedSalary": EstimatedSalary, 
            "Geography": Geography,
            "Gender": Gender
        }])

        # 2️⃣ Apply Feature Engineering (للعرض فقط)
        engineered_data = feature_engineering(input_data)

        # 3️⃣ Prediction (الموديل مدرب على الأعمدة الأصلية فقط)
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        # 4️⃣ Show Results
        st.success("✅ Prediction Completed!")
        if prediction:
            st.error("😟 Prediction: Customer Will Churn")
        else:
            st.success("😃 Prediction: Customer Will Stay")

        colA, colB = st.columns(2)
        with colA:
            st.metric("Probability of Churn", f"{proba:.2%}")
        with colB:
            st.metric("Confidence", f"{(1-proba if prediction==0 else proba):.2%}")

        # 5️⃣ Show Engineered Features للعميل
        st.subheader("🧩 Engineered Features for this Customer")
        st.dataframe(engineered_data)

        # 6️⃣ Save result to session_state
        st.session_state["last_pred"] = prediction
        st.session_state["last_proba"] = proba
        st.session_state["last_input"] = engineered_data
# ================= ABOUT DATA =================
elif choice == "ℹ️ About Data":
    st.markdown("<div class='big-title'>Bank Churn Dashboard</div>", unsafe_allow_html=True)

    # ✅ مركز الصورة
    st.markdown("""
    <div style='display:flex; justify-content:center; margin:20px 0;'>
        <img src="https://hacheemaster.github.io/assets/images/churn.jpeg" 
             width="800">
    </div>
    """, unsafe_allow_html=True)

    st.title("ℹ️ Dataset Information")

    # ------------------- Column Description -------------------
    st.subheader("📋 Column Descriptions")
    desc = {
        "CreditScore":"Credit score of the customer",
        "Age":"Age of customer",
        "Tenure":"Years with bank",
        "Balance":"Account balance",
        "NumOfProducts":"Number of bank products the customer is using",
        "HasCrCard":"Whether customer has a credit card (1=yes,0=no)",
        "IsActiveMember":"Whether customer is active (1=yes,0=no)",
        "EstimatedSalary":"Estimated salary",
        "Geography":"Country of customer",
        "Gender":"Gender of customer",
        "Exited":"Target: 1=Churn, 0=Stay"
    }
    st.table(pd.DataFrame(desc.items(), columns=["Column","Description"]))

    # ------------------- Dataset Filters -------------------
    st.subheader("🔎 Data Filters")

    col1, col2 = st.columns(2)
    with col1:
        geo_filter = st.multiselect("🌍 Select Geography", options=df["Geography"].unique())
    with col2:
        gender_filter = st.multiselect("👤 Select Gender", options=df["Gender"].unique())

    filtered_df = df.copy()
    if geo_filter:
        filtered_df = filtered_df[filtered_df["Geography"].isin(geo_filter)]
    if gender_filter:
        filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]

    n_rows = st.slider("📌 Number of rows to display", 5, 50, 10)
    st.dataframe(filtered_df.head(n_rows))

    # ------------------- Dataset Overview -------------------
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(filtered_df))
    with col2:
        st.metric("Features", filtered_df.shape[1])
    with col3:
        st.metric("Churn Rate", f"{filtered_df['Exited'].mean()*100:.2f}%")

    # ------------------- Target Distribution -------------------
    st.subheader("🎯 Churn Distribution")
    fig_target = px.pie(filtered_df, names="Exited", title="Churn vs Stay",
                        color="Exited", color_discrete_map={0:"green", 1:"red"})
    st.plotly_chart(fig_target, use_container_width=True)
# ================= ANALYSIS =================
# ================= ANALYSIS =================
elif choice == "📊 Analysis":
    st.markdown("<div class='big-title'>Customer Churn Analysis</div>", unsafe_allow_html=True)

    # ✅ مركز الصورة + تصغير الحجم
    st.markdown("""
    <div style='display:flex; justify-content:center; margin:20px 0;'>
        <img src="https://ece.emory.edu/_includes/images/sections/programs/Data-Analytics-Intro.jpg" 
             width="1000">
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Geography", color="Exited", barmode="group", title="Churn by Geography")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df, x="NumOfProducts", color="Exited", barmode="group", title="Churn by Number of Products")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(df, names="Gender", title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df, x="IsActiveMember", color="Exited", barmode="group", title="Churn vs Active Membership")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Age Groups vs Churn")
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,30,40,50,60,100], labels=['<30','30-40','40-50','50-60','60+'])
    fig_age = px.histogram(df, x="AgeGroup", color="Exited", barmode="group", title="Churn by Age Group")
    st.plotly_chart(fig_age, use_container_width=True)
    fig = px.histogram(df, x="Age", color="Exited", barmode="group", title="Churn by Age")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(df, x="Tenure", color="Exited", barmode="group", title="Churn by Tenure")
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("💰 Balance Distribution by Churn")
    fig_balance = px.histogram(df, x="Balance", color="Exited", nbins=40, title="Balance Distribution by Churn")
    st.plotly_chart(fig_balance, use_container_width=True)

# ================= EDA =================
# ================= EDA =================

 
elif choice == "🧩 EDA":
    st.markdown("<div class='big-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    # ✅ مركز الصورة + تصغير الحجم
    st.markdown("""
    <div style='display:flex; justify-content:center; margin:20px 0;'>
        <img src="https://media2.giphy.com/media/v1.Y2lkPTZjMDliOTUyd3Zqa3AydDdqZzJtZTE4bjdnNXMxczF3NW9jYXF2dWVmbnF3amRnYyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/xT9C25UNTwfZuk85WP/200w.gif" 
             width="1000">
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔎 Summary Statistics")
    st.write(df.describe())

    st.subheader("🛑 Missing Values")
    st.write(df.isnull().sum())

    # KDE / Distribution plots
    st.subheader("📊 Distribution of Continuous Features")
    for feature in ["Age","CreditScore","Balance","EstimatedSalary"]:
        fig = px.histogram(df, x=feature, color="Exited", nbins=30, opacity=0.7,
                           title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📌 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Boxplots for outlier detection
    st.subheader("📦 Outlier Analysis with Boxplots")
    features_box = ["Age","CreditScore","Balance","EstimatedSalary"]
    for f in features_box:
        fig_box = px.box(df, y=f, color="Exited", title=f"{f} Distribution by Churn")
        st.plotly_chart(fig_box, use_container_width=True)

    # Scatter Matrix
    st.subheader("🔗 Scatter Matrix (Sample 300 Customers)")
    fig_scatter_matrix = px.scatter_matrix(df.sample(300), dimensions=["Age","Balance","CreditScore","EstimatedSalary"], color="Exited")
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    
# ================= RECOMMENDATIONS =================
elif choice == "💡 Recommendations":
    st.title("💡 Smart Recommendations")

    # check if prediction was made
    if "last_pred" not in st.session_state or "last_input" not in st.session_state:
        st.warning("⚠️ Please make a prediction first")
    else:
        pred = st.session_state["last_pred"]
        prob = st.session_state["last_proba"]
        input_data = st.session_state["last_input"]

        import plotly.graph_objects as go
        # ===== Gauge Visualization =====
        st.subheader("📊 Churn Probability Gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {"color": "red" if pred==1 else "green"},
                'steps': [
                    {'range':[0,40],'color':'lightgreen'},
                    {'range':[40,70],'color':'yellow'},
                    {'range':[70,100],'color':'tomato'}
                ]
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # ===== Progress Bar for Churn %
        st.subheader("📊 Probability Progress Bar")
        st.progress(int(prob*100))

        # ===== Customer Insights =====
        st.subheader("🔎 Customer Insights")
        avg_values = df.mean(numeric_only=True)
        st.write(f"- **Age**: {int(input_data['Age'].values[0])} years compared to the average {avg_values['Age']:.0f}")
        st.write(f"- **Balance**: {input_data['Balance'].values[0]:,.0f} compared to the average {avg_values['Balance']:.0f}")
        st.write(f"- **Products Owned**: {input_data['NumOfProducts'].values[0]} compared to the average {avg_values['NumOfProducts']:.1f}")
        st.write(f"- **Estimated Salary**: {input_data['EstimatedSalary'].values[0]:,.0f} compared to the average {avg_values['EstimatedSalary']:.0f}")

        # ===== Similar Customers Analysis =====
        st.subheader("📊 Similar Customers Behavior")
        similar = df[
            (df['Age'].between(int(input_data['Age'].values[0])-5, int(input_data['Age'].values[0])+5)) &
            (df['NumOfProducts'] == input_data['NumOfProducts'].values[0])
        ]
        if not similar.empty:
            fig_sim = px.pie(similar, names="Exited", title="Churn vs Stay among Similar Customers")
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("💡 No enough similar customers found in dataset")

           # ===== What-if Analysis =====
            st.subheader("🧪 What-if Analysis")

            new_balance = st.slider(
                "Modify Balance",
                0,
                int(df["Balance"].max()),
                int(input_data["Balance"].values[0])
            )

            # نعمل نسخة ونغير الـ Balance فيها
            whatif_data = input_data.copy()
            whatif_data["Balance"] = new_balance

            # اعادة التنبؤ
            new_proba = model.predict_proba(whatif_data)[0][1]
            new_pred = model.predict(whatif_data)[0]   # 👈 هنا عرفناه قبل الاستخدام

            st.write(f"🔮 Churn Probability after modifying balance = {new_proba:.2%}")

            
            # ===== Recommendations Based on Prediction =====
        if pred==1:
            st.error(f"🚨 High Risk of Churn ({prob:.0%})")
            st.markdown("### Suggested Retention Actions:")
            st.write("- 📞 Personal call with relationship manager")
            st.write("- 💳 Offer better credit card benefits")
            st.write("- 💰 Tailored loan/financial offers")
            st.write("- 🎁 Loyalty rewards program")
        else:
            st.success(f"✅ Customer Likely to Stay ({prob:.0%})")
            st.markdown("### Suggested Opportunities:")
            st.write("- 🛍️ Cross-sell investment plans")
            st.write("- 🤝 Promote loyalty schemes")
            st.write("- 📲 Push digital banking adoption")
            st.write("- 🎯 Personalized marketing")
        # Boxplots
        st.subheader("📊 Boxplots of Key Features vs Churn")
        features_box = ["Age","CreditScore","Balance","EstimatedSalary"]
        for f in features_box:
            fig_box = px.box(df, x="Exited", y=f, color="Exited", title=f"{f} vs Churn")
            st.plotly_chart(fig_box, use_container_width=True)

        # Bar chart for categorical features
        st.subheader("📊 Categorical Features vs Churn")
        for cat in ["Geography","Gender","NumOfProducts","HasCrCard","IsActiveMember"]:
            fig_cat = px.histogram(df, x=cat, color="Exited", barmode="group", title=f"{cat} vs Churn")
            st.plotly_chart(fig_cat, use_container_width=True)
