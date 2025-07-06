import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📧 Email Spam Classifier")
st.markdown("### Analyze and classify emails as spam or non-spam using AI")
st.markdown("---")

# Sidebar for API key input
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input(
        "Enter your GROQ API Key:",
        type="password",
        help="Get your API key from https://console.groq.com"
    )
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Enter your GROQ API key in the sidebar
    2. Paste the email content you want to analyze
    3. Click 'Classify Email' to get results
    4. View the classification, sentiment, and detailed analysis
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Email Content")
    email_content = st.text_area(
        "Paste your email content here:",
        height=300,
        placeholder="Enter the email content you want to classify...",
        help="Paste the complete email content including headers, body, and signature"
    )

with col2:
    st.header("🎯 Quick Actions")
    
    # Sample emails for testing
    st.subheader("Sample Emails")
    
    sample_legitimate = """Good morning,

I would need a ppt on the topic I had sent yesterday

Please make it fast.

Thank you,
Kind Regards
Divyam Maskeri"""
    
    sample_suspicious = """Good morning,

Please send us the invoice which was not attached in the previous email, along with your debit card number and expiry date, CVV.

Thank you,
Kind Regards
Divyam Maskeri"""
    
    sample_marketing = """Register today for an exclusive webinar on how AI + Automation maximizes time to value and business impact!

Bringing you a few of the highlights from AAI's Partner Summit and Imagine conference with a focus on AI + Automation and AI Agents, we are hosting a webinar with the Sr. Global Director of GenAI Solutions, Luis Barcenas, and other GenAI Solutions experts to explain how to manage and monetize complex AI operations with AI Agents.
We will also go over new content in the partner portal and new incentives with our Sr. Director of Partner Programs & Incentives, Frances Fortanely.
Please register for a time that works for you.
AMER Aug 27, 2024 08:00 AM PST
APJ Aug 28, 2024 10:30 AM IST (Aug 27 10:00 PM PST)
We are excited to help you learn, get enabled, and sell Automation & AI solutions to your customers!
If you have any questions, please email pinnacle@automationanywhere.com.
Thank you for your partnership. Go be great!
Pinnacle Partner Program Team"""
    
    if st.button("📋 Load Legitimate Email", use_container_width=True):
        st.session_state.email_content = sample_legitimate
        st.rerun()
    
    if st.button("⚠️ Load Suspicious Email", use_container_width=True):
        st.session_state.email_content = sample_suspicious
        st.rerun()
    
    if st.button("📢 Load Marketing Email", use_container_width=True):
        st.session_state.email_content = sample_marketing
        st.rerun()

# Update email content if sample was selected
if 'email_content' in st.session_state:
    email_content = st.session_state.email_content

# Classification section
st.markdown("---")
st.header("🔍 Email Classification")

# Initialize session state for results
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

# Classification button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_button = st.button(
        "🚀 Classify Email",
        use_container_width=True,
        type="primary"
    )

# Classification logic
if classify_button:
    if not groq_api_key:
        st.error("❌ Please enter your GROQ API key in the sidebar")
    elif not email_content.strip():
        st.error("❌ Please enter email content to classify")
    else:
        try:
            with st.spinner("🔄 Analyzing email..."):
                # Set up the environment
                os.environ["GROQ_API_KEY"] = groq_api_key
                
                # Initialize the model
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct'
                )
                
                # Create the prompt template
                template = """
As an expert spam email classifier you are supposed to analyze and categorize any email as either spam or non-spam. Post evaluation, you need to generate a comprehensive and concise report
explaining the details of the classification outcome with proper justification. Also adjudge the sentiment of the email. Also, please pick the name of the sender of the mail and also any other important customer details, if present. Classify the intent of the email as per the issue and show it in one line.

email: {email}
"""
                
                prompt = PromptTemplate.from_template(template=template)
                parser = StrOutputParser()
                chain = prompt | llm | parser
                
                # Get classification result
                result = chain.invoke({"email": email_content})
                st.session_state.classification_result = result
                
        except Exception as e:
            st.error(f"❌ Error during classification: {str(e)}")

# Display results
if st.session_state.classification_result:
    st.markdown("---")
    st.header("📊 Classification Results")
    
    # Display the result in a nice format
    st.markdown("### 📋 Detailed Analysis")
    st.text_area(
        "Classification Report:",
        value=st.session_state.classification_result,
        height=300,
        disabled=True
    )
    
    # Additional features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💾 Export Results")
        if st.button("📥 Download Report", use_container_width=True):
            st.download_button(
                label="Download Classification Report",
                data=st.session_state.classification_result,
                file_name="email_classification_report.txt",
                mime="text/plain"
            )
    
    with col2:
        st.markdown("### 🔄 Actions")
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.classification_result = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("### 🛠️ About")
st.markdown("""
This email classifier uses advanced AI to analyze email content and determine whether it's spam or legitimate. 
The model considers various factors including:
- **Content Analysis**: Examines the email text for spam indicators
- **Sender Information**: Analyzes sender details and authenticity
- **Sentiment Analysis**: Determines the emotional tone of the email
- **Intent Classification**: Identifies the purpose of the email
""")

st.markdown("---")
st.markdown("*Built with Streamlit and LangChain | Powered by GROQ*")
