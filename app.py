import streamlit as st
import pickle
import PyPDF2
import re
import torch
from io import BytesIO
import os
from journal_recommendation_system import JournalRecommender

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'journal_recommender.pkl')
    try:
        device = torch.device('cpu')
        
        # Load with CPU mapping
        with open(model_path, 'rb') as f:
            model_components = torch.load(f, map_location=device)
            
        # Initialize recommender
        recommender = JournalRecommender()
        recommender.model.load_state_dict(model_components["model_state"])
        recommender.tokenizer = model_components["tokenizer"]
        recommender.df = model_components["df"]
        recommender.df_embeds = model_components["df_embeds"]
        recommender.model.to(device)
        
        return recommender
        
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {str(e)}")

def extract_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    first_page = pdf_reader.pages[0].extract_text()
    
    # Extract title (usually first line)
    title = first_page.split('\n')[0].strip()
    
    # Extract abstract (typically starts with "Abstract")
    abstract_match = re.search(r'Abstract[:\s]*(.*?)(?=\n\n|\n[A-Z]{2,}|$)', 
                             first_page, 
                             re.DOTALL | re.IGNORECASE)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    return title, abstract

def main():
    st.title("Journal Recommendation System")
    
    # Load model
    try:
        recommender = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload your paper (PDF)", type="pdf")
    
    if uploaded_file:
        try:
            # Extract text
            title, abstract = extract_from_pdf(BytesIO(uploaded_file.read()))
            
            # Display extracted text
            st.subheader("Extracted Information")
            st.write("**Title:**", title)
            st.write("**Abstract:**", abstract)
            
            # Get recommendations
            if st.button("Get Recommendations"):
                with st.spinner("Generating recommendations..."):
                    recs = recommender.recommend(title, abstract, top_k=10)
                
                # Display recommendations
                st.subheader("Recommended Journals")
                for i, rec in enumerate(recs, 1):
                    st.markdown(f"""
                    **{i}. {rec['journal_name']}**
                    - Similarity Score: {rec['similarity']:.3f}
                    - Journal Score: {rec['journal_score']:.3f}
                    """)
                    
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()