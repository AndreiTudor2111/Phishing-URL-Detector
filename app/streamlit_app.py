"""
Phishing URL Detection - Streamlit Web Application

A user-friendly interface for detecting phishing URLs using machine learning.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path so we can import src package
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from src.models.predict import PhishingDetector
from src.utils.config import Config


# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .phishing {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .legitimate {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🔐 Phishing URL Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect malicious URLs using machine learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("🤖 Model Selection")
        
        # Check which models are available
        traditional_ml_available = (
            Config.MODEL_RANDOM_FOREST.exists() and 
            Config.MODEL_XGBOOST.exists()
        )
        
        neural_networks_available = (
            (Config.MODEL_PRODUCTION_DIR / "neural_network_lstm.pt").exists() and
            (Config.MODEL_PRODUCTION_DIR / "neural_network_scaler.pkl").exists()
        )
        
        # Determine available options
        model_category_options = []
        if traditional_ml_available:
            model_category_options.append("Traditional ML")
        if neural_networks_available:
            model_category_options.append("Neural Networks")
        
        # If no models available, show error
        if not model_category_options:
            st.error("⚠️ No models found! Please train models first.")
            st.stop()
        
        # Model category selection (only show if both are available)
        if len(model_category_options) > 1:
            model_category = st.radio(
                "Model Type:",
                options=model_category_options,
                help="Choose between traditional machine learning or deep learning models"
            )
        else:
            model_category = model_category_options[0]
            st.info(f"**Available:** {model_category} models only")
        
        # Model selection based on category
        if model_category == "Traditional ML":
            model_options = {
                'XGBoost': 'xgboost',
                'Random Forest': 'random_forest'
            }
            
            model_display = st.selectbox(
                "Select Model:",
                options=list(model_options.keys()),
                index=0,
                help="Traditional ML models with >93% accuracy"
            )
            
            model_type = model_options[model_display]
            use_neural = False
            
            # Model info
            if model_display == 'XGBoost':
                st.info("**XGBoost**\n- Accuracy: 93.3%\n- Speed: ⚡ Very Fast\n- Size: 4.8 MB")
            else:
                st.info("**Random Forest**\n- Accuracy: 93.5%\n- Speed: ⚡ Fast\n- Size: 223 MB")
        
        else:  # Neural Networks
            model_options = {
                'FeedForward NN': 'feedforward',
                'LSTM Network': 'lstm',
                'GRU Network': 'gru',
                'CNN Network': 'cnn'
            }
            
            model_display = st.selectbox(
                "Select Model:",
                options=list(model_options.keys()),
                index=1,  # Default to LSTM
                help="Deep learning models trained with PyTorch"
            )
            
            model_type = model_options[model_display]
            use_neural = True
            
            # Model info
            model_info = {
                'FeedForward NN': "**FeedForward**\n- Size: 243 KB\n- Speed: ⚡ Fastest\n- Type: Dense Network",
                'LSTM Network': "**LSTM**\n- Size: 954 KB\n- Speed: 🔄 Fast\n- Type: Sequential",
                'GRU Network': "**GRU**\n- Size: 725 KB\n- Speed: 🔄 Fast\n- Type: Sequential",
                'CNN Network': "**CNN**\n- Size: 1.12 MB\n- Speed: 🔄 Medium\n- Type: Convolutional"
            }
            st.info(model_info[model_display])
        
        st.markdown("---")
        
        st.header("📊 About")
        st.info(f"""
        Currently using: **{model_display}**
        
        This application uses advanced machine learning to detect phishing URLs.
        
        **Features:**
        - 60 URL characteristics analyzed
        - Statistical pattern recognition
        - Real-time predictions
        - Multiple model options
        """)
        
        st.markdown("---")
        st.caption("Developed by Ostache Andrei Tudor")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single URL", "📁 Batch Analysis", "ℹ️ Information"])
    
    # Tab 1: Single URL Detection
    with tab1:
        st.header("Analyze Single URL")
        
        url_input = st.text_input(
            "Enter URL to check:",
            placeholder="https://example.com/login",
            help="Enter the complete URL including http:// or https://"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            analyze_button = st.button("🔍 Analyze URL", use_container_width=True, type="primary")
        
        if analyze_button and url_input:
            with st.spinner("Analyzing URL..."):
                try:
                    # Load appropriate detector
                    if use_neural:
                        from src.models.predict_neural import NeuralNetworkDetector
                        detector = NeuralNetworkDetector(model_type=model_type)
                    else:
                        detector = PhishingDetector(model_type=model_type)
                    
                    # Make prediction
                    prediction, probability = detector.predict(url_input)
                    
                    # Display result with color coding
                    if prediction == "Phishing":
                        st.error(f"""
### ⚠️ PHISHING DETECTED

**This URL appears to be malicious!**

🔴 **Confidence:** {probability*100:.1f}%

⚠️ **Warning:** Avoid clicking or entering credentials on this site.
                        """)
                    else:
                        st.success(f"""
### ✅ LEGITIMATE

**This URL appears to be safe.**

🟢 **Confidence:** {(1-probability)*100:.1f}%

✓ **Status:** No phishing indicators detected.
                        """)
                    
                    # Show URL details
                    with st.expander("🔬 View Technical Details"):
                        from src.feature_extraction import extract_features
                        features = extract_features(url_input)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("URL Length", features['length_url'])
                            st.metric("Number of Dots", features['nb_dots'])
                            st.metric("Number of Digits", features['nb_digits'])
                            st.metric("URL Entropy", f"{features['entropy_url']:.2f}")
                        
                        with col2:
                            st.metric("Contains IP", "Yes" if features['contains_ip'] else "No")
                            st.metric("Suspicious Keywords", features['number_suspicious_words'])
                            st.metric("Subdomains", features['count_subdomains'])
                            st.metric("HTTPS", "Yes" if features['https_token'] else "No")
                
                except Exception as e:
                    st.error(f"Error analyzing URL: {str(e)}")
        
        elif analyze_button:
            st.warning("⚠️ Please enter a URL to analyze")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch URL Analysis")
        
        st.write("Upload a CSV file containing URLs to analyze multiple URLs at once.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should contain a column with URLs"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.write(f"Loaded {len(df)} rows")
                
                # Column selection
                url_column = st.selectbox(
                    "Select the column containing URLs:",
                    options=df.columns.tolist()
                )
                
                if st.button("🔍 Analyze All URLs", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} URLs..."):
                        # Load appropriate detector
                        if use_neural:
                            from src.models.predict_neural import NeuralNetworkDetector
                            detector = NeuralNetworkDetector(model_type=model_type)
                        else:
                            detector = PhishingDetector(model_type=model_type)
                        
                        predictions = []
                        probabilities = []
                        
                        progress_bar = st.progress(0)
                        
                        for idx, url in enumerate(df[url_column]):
                            pred, prob = detector.predict(str(url))
                            predictions.append(pred)
                            probabilities.append(prob)
                            progress_bar.progress((idx + 1) / len(df))
                        
                        df['Prediction'] = predictions
                        df['Phishing_Probability'] = [f"{p*100:.1f}%" for p in probabilities]
                        
                        # Display results
                        st.success("✅ Analysis complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            phishing_count = sum(1 for p in predictions if p == "Phishing")
                            st.metric("🚨 Phishing URLs", phishing_count)
                        
                        with col2:
                            legitimate_count = sum(1 for p in predictions if p == "Legitimate")
                            st.metric("✅ Legitimate URLs", legitimate_count)
                        
                        with col3:
                            st.metric("📊 Total Analyzed", len(df))
                        
                        # Show results table
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="phishing_detection_results.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Tab 3: Information
    with tab3:
        st.header("How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Detection Features")
            st.markdown("""
            Our system analyzes **60 characteristics** of each URL:
            
            - **Structural Analysis**
              - URL length and complexity
              - Domain and subdomain patterns
              - Path and query parameters
            
            - **Character Analysis**
              - Special character frequency
              - Digit ratios
              - Suspicious symbols
            
            - **Statistical Analysis**
              - Entropy calculation
              - Character frequency distribution
              - KL divergence from normal patterns
            
            - **Pattern Recognition**
              - IP addresses in URLs
              - Suspicious keywords
              - Known phishing indicators
              - URL shortening services
            """)
        
        with col2:
            st.subheader("🤖 Machine Learning Models")
            st.markdown("""
            **Traditional ML (Gradient Boosting/Ensembles)**
            
            **XGBoost Classifier**
            - Accuracy: 93.3%
            - Size: 4.8 MB
            - Speed: ⚡ Very Fast
            
            **Random Forest Classifier**
            - Accuracy: 93.5%
            - Size: 223 MB
            - Speed: ⚡ Fast
            
            ---
            
            **Neural Networks (Deep Learning)**
            
            **FeedForward NN**
            - Size: 243 KB
            - Speed: ⚡ Fastest
            - Architecture: Dense layers
            
            **LSTM Network**
            - Size: 954 KB
            - Speed: 🔄 Fast
            - Architecture: Recurrent
            
            **GRU Network**
            - Size: 725 KB
            - Speed: 🔄 Fast
            - Architecture: Recurrent
            
            **CNN Network**
            - Size: 1.12 MB
            - Speed: 🔄 Medium
            - Architecture: Convolutional
            
            All models trained on **662K+ URLs** with **60 features** each.
            """)
        
        st.markdown("---")
        
        st.subheader("⚠️ Important Notes")
        st.warning("""
        - This tool provides predictions based on URL characteristics only
        - Always exercise caution with suspicious links
        - No detection system is 100% accurate
        - When in doubt, verify the source independently
        - Never enter credentials on suspicious websites
        """)


if __name__ == "__main__":
    main()
