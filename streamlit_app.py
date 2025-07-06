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
