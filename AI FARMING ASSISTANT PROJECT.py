import os
from datetime import date

import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f4faf4 0%, #f8fbf7 48%, #ffffff 100%);
        color: #183b2a;
    }
    [data-testid="stSidebar"] {
        background: #123c2a;
    }
    [data-testid="stSidebar"] * {
        color: #f1f8f1 !important;
    }
    .hero {
        background: linear-gradient(120deg, #123c2a 0%, #1e6842 60%, #398653 100%);
        padding: 2.1rem 2.3rem;
        border-radius: 24px;
        color: white;
        margin: 0.4rem 0 1.5rem 0;
        box-shadow: 0 14px 35px rgba(22, 76, 47, .14);
    }
    .hero h1 {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 800;
        margin: 0.3rem 0 0.45rem 0;
        letter-spacing: -1.2px;
    }
    .hero p { color: #e0f0e4; font-size: 1.04rem; margin: 0; }
    .eyebrow {
        color: #b9e4c5;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: .75rem;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        border: 1px solid #e1eee3;
        border-radius: 17px;
        padding: 1rem 1.1rem;
        min-height: 112px;
        box-shadow: 0 5px 18px rgba(24, 59, 42, .045);
    }
    .metric-label { color: #6b8272; font-size: .82rem; font-weight: 600; }
    .metric-value { color: #1b5535; font-family: 'Manrope'; font-size: 1.22rem; font-weight: 800; margin-top: .35rem; }
    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #174a30;
        font-size: 1.35rem;
        font-weight: 800;
        margin: .35rem 0 .2rem 0;
    }
    .section-sub { color: #6b8272; margin-bottom: 1rem; }
    .report-card {
        background: #ffffff;
        border: 1px solid #dfecdf;
        border-left: 5px solid #398653;
        border-radius: 15px;
        padding: 1.1rem 1.3rem;
        margin: .7rem 0;
        box-shadow: 0 5px 18px rgba(24, 59, 42, .04);
    }
    div.stButton > button[kind="primary"], .stFormSubmitButton > button {
        background: #237447;
        border: 0;
        border-radius: 11px;
        color: white;
        font-weight: 700;
        padding: .65rem 1.2rem;
    }
    div.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
        background: #185b35;
        color: white;
    }
    .small-note { color: #738778; font-size: .82rem; }
    [data-testid="stForm"] {
        background: rgba(255,255,255,.82);
        border: 1px solid #e1eee3;
        border-radius: 18px;
        padding: 1.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Gemini client and prompts
# ---------------------------------------------------------
def get_gemini_client():
    """Read the API key from Streamlit secrets or an environment variable."""
    api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def build_validation_prompt(details):
    return f"""
You are an input validation system for an AI Farming Assistant.

Check whether the farmer's answers are meaningful, realistic and relevant.

Farming Style: {details['farming_style']}
Crop / Trees: {details['crop']}
Soil Description: {details['soil_status']}
Water Drainage: {details['water_drainage']}
Location: {details['location']}
Water Availability: {details['water_availability']}
Sunlight: {details['sunlight']}
Irrigation Source: {details['water_source']}
Other Concerns: {details['extra_concerns']}

Rules:
1. Reject random words, fictional places, jokes and unrelated answers.
2. Accept "I don't know" when the farmer genuinely doesn't know.
3. Accept "None" or "No problems" for concerns.
4. Accept unusual but potentially valid farming information.
5. Reject fictional irrigation sources such as chocolate rivers.
6. Do not reject answers just because they are unfamiliar.

Output only:
VALID
OR
INVALID: [Brief explanation]
"""


def build_report_prompt(details):
    today = date.today().strftime("%d %B %Y")
    return f"""
You are a professional agricultural advisor. Analyse the farmer's information
and generate a short, crisp and practical farming report.

FARMER DETAILS:
Farming Style: {details['farming_style']}
Previous Crop / Existing Trees: {details['crop']}
Soil Colour and Texture: {details['soil_status']}
Water Drainage: {details['water_drainage']}
Irrigation Source: {details['water_source']}
Water Availability: {details['water_availability']}
Sunlight: {details['sunlight']}
Location: {details['location']}
Other Concerns: {details['extra_concerns']}
Date: {today}

IMPORTANT:
- Consider the location and current agricultural season.
- The farmer does not have a soil testing kit.
- Never claim exact pH or nutrient levels without testing.
- Do not invent weather data or statistics.
- Give practical and affordable recommendations.
- Use simple English and short bullet points only.
- Give a maximum of 3 bullet points under each section.
- Keep each bullet to one or two short sentences.
- If information is insufficient, clearly mention it.

Use exactly these sections:
🌱 1. SOIL HEALTH ANALYSIS
- Likely soil characteristics.
- Possible soil concerns.
- Suggested improvement.

🌾 2. NEXT FARMING STRATEGY
- Recommended crop or tree-management strategy.
- Why it is suitable.
- Approximate growing duration, if known.

🚜 3. CULTIVATION PROCEDURES
- Land preparation and sowing.
- Water management.
- Crop maintenance and harvesting.

🍃 4. NATURAL SOLUTIONS
- Organic manure and compost.
- Mulching and soil improvement.
- Natural pest management.

⚠️ 5. FARMER ALERTS
- Important risks.
- Precautions.
- Additional tests or information needed.

Use headings and bullets. No lengthy introduction or conclusion.
"""


def generate_ai_report(details):
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml "
            "or set it as an environment variable."
        )

    validation_prompt = build_validation_prompt(details)
    validation_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=validation_prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    validation = (validation_response.text or "").strip()

    if validation != "VALID":
        if validation.startswith("INVALID:"):
            raise ValueError(validation.replace("INVALID:", "", 1).strip())
        raise RuntimeError("The information could not be verified. Please review your answers.")

    report_response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=build_report_prompt(details),
        config=types.GenerateContentConfig(temperature=0.4),
    )
    report = (report_response.text or "").strip()
    if not report:
        raise RuntimeError("Gemini did not return a report. Please try again.")
    return report


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌱 FarmAssist AI")
    st.caption("Smart guidance for better farming")
    st.markdown("---")
    st.markdown("### Your workspace")
    st.markdown("🧑‍🌾 Farmer profile")
    st.markdown("🌾 Crop planning")
    st.markdown("💧 Soil & water insights")
    st.markdown("📋 AI farming report")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Enter your farm details\n"
        "2. AI checks your answers\n"
        "3. Get practical recommendations"
    )
    st.markdown("---")
    st.caption("Powered by Gemini AI")
    st.caption("Advice is a starting point; confirm high-impact decisions with a local agricultural officer.")

# ---------------------------------------------------------
# Header and summary cards
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Intelligent agriculture • Simple decisions</div>
        <h1>AI Farming Assistant 🌿</h1>
        <p>Understand your farm better and get practical, affordable recommendations powered by AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
for col, icon, label, value in [
    (c1, "🌱", "FARMING SUPPORT", "Crop & tree guidance"),
    (c2, "💧", "RESOURCE AWARENESS", "Soil, water & sunlight"),
    (c3, "🤖", "AI-POWERED", "Personalised farm report"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{icon} {label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.markdown('<div class="section-heading">Tell us about your farm</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Fill in the details below. If you are unsure about an answer, you can write “I don’t know”.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Farmer form
# ---------------------------------------------------------
with st.form("farmer_details_form"):
    st.markdown("#### 01  Farming profile")
    farming_style = st.radio(
        "Select your farming style",
        ["Seasonal Crops", "Permanent Trees"],
        horizontal=True,
        help="Choose seasonal crops or trees that remain on your farm for several years.",
    )
    crop_question = (
        "What crop did you grow previously?"
        if farming_style == "Seasonal Crops"
        else "What type of trees are growing on your farm?"
    )
    crop = st.text_input(crop_question, placeholder="e.g. Ragi, paddy, mango trees")

    st.markdown("---")
    st.markdown("#### 02  Soil and water")
    left, right = st.columns(2)
    with left:
        soil_status = st.text_input(
            "Soil colour and texture",
            placeholder="e.g. Red clay, black loose soil",
        )
        water_drainage = st.selectbox(
            "How does water drain from your soil?",
            ["Select an option", "Quickly", "Slowly", "Water remains on the soil", "I don't know"],
        )
        water_source = st.text_input(
            "Main irrigation water source",
            placeholder="e.g. Borewell, canal, rainwater",
        )
    with right:
        water_availability = st.selectbox(
            "Water availability",
            [
                "Select an option",
                "Throughout the year",
                "Only during certain seasons",
                "Limited / irregular",
                "I don't know",
            ],
        )
        sunlight = st.selectbox(
            "Sunlight received by your farm",
            ["Select an option", "Full sunlight", "Partial sunlight", "Mostly shade", "I don't know"],
        )
        location = st.text_input(
            "Village, city or region",
            placeholder="e.g. Mysuru, Karnataka",
        )

    st.markdown("---")
    st.markdown("#### 03  Other farming concerns")
    extra_concerns = st.text_area(
        "Any pests, diseases or other concerns?",
        placeholder="Describe the problem, or enter None if there are no concerns.",
        height=95,
    )

    submitted = st.form_submit_button("🌿 Analyse my farm", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Validation and report
# ---------------------------------------------------------
if submitted:
    required_text = {
        "Crop / trees": crop,
        "Soil description": soil_status,
        "Irrigation source": water_source,
        "Location": location,
        "Other concerns": extra_concerns,
    }
    missing = [name for name, value in required_text.items() if not value.strip()]
    if water_drainage == "Select an option":
        missing.append("Water drainage")
    if water_availability == "Select an option":
        missing.append("Water availability")
    if sunlight == "Select an option":
        missing.append("Sunlight")

    if missing:
        st.error("Please complete these fields: " + ", ".join(missing) + ".")
    else:
        details = {
            "farming_style": farming_style,
            "crop": crop.strip(),
            "soil_status": soil_status.strip(),
            "water_drainage": water_drainage,
            "location": location.strip(),
            "water_availability": water_availability,
            "sunlight": sunlight,
            "water_source": water_source.strip(),
            "extra_concerns": extra_concerns.strip(),
        }
        st.session_state["farm_details"] = details
        st.session_state.pop("farm_report", None)
        st.session_state.pop("farm_error", None)

        with st.status("Checking your farm information and preparing recommendations...", expanded=True) as status:
            try:
                st.write("🔎 Checking that the information is meaningful...")
                report = generate_ai_report(details)
                st.session_state["farm_report"] = report
                status.update(label="Your farming report is ready!", state="complete", expanded=False)
            except ValueError as exc:
                st.session_state["farm_error"] = f"Your information needs a quick review: {exc}"
                status.update(label="Please review your farm information", state="error", expanded=True)
            except Exception as exc:
                st.session_state["farm_error"] = str(exc)
                status.update(label="Could not generate the report", state="error", expanded=True)

if st.session_state.get("farm_error"):
    st.error(st.session_state["farm_error"])

if st.session_state.get("farm_report"):
    st.write("")
    st.markdown('<div class="section-heading">Your AI agronomist report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Recommendations based on the details you provided.</div>',
        unsafe_allow_html=True,
    )
    details = st.session_state.get("farm_details", {})
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">FARM TYPE</div>'
            f'<div class="metric-value">{details.get("farming_style", "—")}</div></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">CROP / TREES</div>'
            f'<div class="metric-value">{details.get("crop", "—")}</div></div>',
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">LOCATION</div>'
            f'<div class="metric-value">{details.get("location", "—")}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(st.session_state["farm_report"])
    st.download_button(
        "⬇️ Download farming report",
        data=st.session_state["farm_report"],
        file_name="ai_farming_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if st.button("Start a new farm analysis", use_container_width=True):
        st.session_state.pop("farm_report", None)
        st.session_state.pop("farm_error", None)
        st.rerun()

st.markdown("---")
st.markdown(
    '<div class="small-note">🌱 AI Farming Assistant • Built with Streamlit and Gemini. '
    'AI suggestions are general guidance and should be checked with local agricultural experts.</div>',
    unsafe_allow_html=True,
)
