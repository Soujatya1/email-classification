import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📧 Email Spam Classifier")
st.markdown("### Analyze and classify emails as spam or non-spam using AI")
st.markdown("---")

with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input(
        "Enter your GROQ API Key:",
        type="password",
        help="Get your API key from https://console.groq.com"
    )

st.header("📝 Email Content")
email_content = st.text_area(
    "Paste your email content here:",
    height=300,
    placeholder="Enter the email content you want to classify...",
    help="Paste the complete email content including headers, body, and signature"
)

st.markdown("---")
st.header("🔍 Email Classification")

if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_button = st.button(
        "🚀 Classify Email",
        use_container_width=True,
        type="primary"
    )

if classify_button:
    if not groq_api_key:
        st.error("❌ Please enter your GROQ API key in the sidebar")
    elif not email_content.strip():
        st.error("❌ Please enter email content to classify")
    else:
        try:
            with st.spinner("🔄 Analyzing email..."):
                os.environ["GROQ_API_KEY"] = groq_api_key
                
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name='meta-llama/llama-4-scout-17b-16e-instruct'
                )
                
                template = """
As an expert email classifier you are supposed to analyze and categorize any email as either Genuine or Fraud. Post evaluation, you need to generate a comprehensive and concise report
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

if st.session_state.classification_result:
    st.markdown("---")
    st.header("📊 Classification Results")
    
    st.markdown("### 📋 Detailed Analysis")
    st.text_area(
        "Classification Report:",
        value=st.session_state.classification_result,
        height=300,
        disabled=True
    )
    
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
