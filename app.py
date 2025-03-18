import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import streamlit as st
import time
import asyncio
import sys

from huggingface_hub import hf_hub_download

os.environ["STREAMLIT_WATCHED_FILES"] = ""
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# # or simply:
# torch.classes.__path__ = []

# Page configuration
st.set_page_config(
    page_title="Mehfil-e-Sukhan",
    page_icon="📜",
    layout="wide"  # Using wide layout for side-by-side content
)


# Custom spinner implementation
def custom_spinner():
    spinner_html = """
    <style>
        .custom-spinner {
            margin-left: 60px;
            margin-top: 20px;
            display: flex;
            align-items: center;
        }
        .custom-spinner-circle {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 3px solid rgba(230, 74, 74, 0.3);
            border-top: 3px solid #E64A4A;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        .custom-spinner-text {
            font-family: 'Source Sans Pro', sans-serif;
            color: #F5F5F5;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <div class="custom-spinner">
        <div class="custom-spinner-circle"></div>
        <div class="custom-spinner-text">Creating your poetry...</div>
    </div>
    """
    return spinner_html

# Apply custom CSS for a professional UI with true vertical split
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background-color: #0D0D0D;
        color: #F5F5F5;
    }
    
    /* Split screen layout - true vertical split with gap */
    .main-layout {
        display: flex;
        width: 100%;
    }

    .left-column {
        width: 30%;
        background-color: #141414;
        padding: 2rem;
        box-sizing: border-box;
    }

    .right-column {
        width: 60%;
        background-color: #1C1C1C;
        padding: 2rem;
        box-sizing: border-box;
        margin-left: 10%;
    }
    
    /* Title styling */
    .title {
        font-family: 'Playfair Display', serif;
        color: #E64A4A;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 5px;
        letter-spacing: 1.2px;
    }
    
    .subtitle {
        font-family: 'Playfair Display', serif;
        color: #F5F5F5;
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 40px;
        font-style: italic;
        letter-spacing: 0.8px;
        opacity: 0.85;
    }
    
    /* Panel titles */
    .panel-title {
        font-family: 'Playfair Display', serif;
        color: #E64A4A;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 25px;
        letter-spacing: 0.8px;
        border-bottom: 1px solid rgba(230, 74, 74, 0.3);
        padding-bottom: 10px;
        margin-left: 5px;
        margin-right: 40px;
        width: fit-content;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #E64A4A;
        color: white;
        width: auto !important;
        transition: all 0.3s ease;
        border: none;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        letter-spacing: 0.6px;
        height: 50px;
        border-radius: 4px;
        margin-left: 60px;
        padding-left: 30px;
        padding-right: 30px;
        display: inline-block;
    }
    
    .stButton > button:hover {
        background-color: #C52E2E;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
  
    /* Input wrapper to handle container margins */
    .stTextInput > div {
        margin-left: 60px;
        margin-right: 60px;  /* Changed from 100px to 60px to match poetry box */
        width: calc(100% - 120px);  /* Added width calculation */
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 4px;
        border: 1px solid rgba(245, 245, 245, 0.2);
        background-color: rgba(245, 245, 245, 0.05);
        color: #000000;
        font-family: 'Source Sans Pro', sans-serif;
        height: 50px;
        width: 100%;  /* Make it use full width of container */
    }

    
    /* Poetry output box */
    .poetry-box {
        background-color: rgba(245, 245, 245, 0.03);
        border-radius: 8px;
        padding: 35px;
        border: 1px solid rgba(245, 245, 245, 0.08);
        border-left: 4px solid #E64A4A;
        min-height: 250px;
        width: calc(100% - 160px);
        font-size: 1.3rem;
        line-height: 2;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.5s ease;
        margin-top: 20px;
        animation: fadeIn 0.6s ease-out forwards;
        margin-left: 60px;
        margin-right: 10px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Control labels */
    .control-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        color: #E64A4A;
        margin-bottom: 5px;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        margin-left: 5px;
        margin-right: 40px;
    }
    
    /* Description text */
    .description-text {
        font-family: 'Source Sans Pro', sans-serif;
        color: rgba(245, 245, 245, 0.7);
        font-size: 0.9rem;
        margin-bottom: 5px;
        font-style: italic;
        margin-left: 5px;
        margin-right: 40px;
    }
    
    /* Slider styling */
    .stSlider div[data-baseweb="slider"] div {
        background-color: transparent;
        border: none;
        box-shadow: none;
        margin-left: 5px;
        margin-right: 40px;
    }
    
    .stSlider div[data-baseweb="slider"] div div div {
        background-color: transparent;
        border: none;
        box-shadow: none;
    }
    
    .stSlider div[data-baseweb="slider"] > div > div {
        background-color: transparent !important;
    }
    
    .stSlider div[data-baseweb="slider"] div[role="slider"] {
        background: #E64A4A;
        border: none;
        box-shadow: none;
    }
    
    .stSlider [data-testid="stTickBarMin"], 
    .stSlider [data-testid="stTickBarMax"] {
        color: #F5F5F5;
        background: none !important;
        border: none;
        box-shadow: none;
    }
    
    /* Select boxes */
    .stSelectbox {
        margin-bottom: 5px;
        margin-left: 5px;
        margin-right: 40px;
    }
    
    .stSelectbox div {
        background-color: rgba(245, 245, 245, 0.05);
        color: #F5F5F5;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(245, 245, 245, 0.05);
        border-color: rgba(245, 245, 245, 0.2);
    }
    
    /* Input container */
    .input-container {
        margin-bottom: 30px;
        margin-left: 60px;
        margin-right: 100px;
    }
    
    /* Starting word label with left margin */
    .starting-word-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        color: #E64A4A;
        margin-bottom: 5px;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        margin-left: 60px;
        margin-right: 40px;
        margin-top: 50px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        width: 100%;
        margin-top: 40px;
        padding: 20px;
        color: rgba(245, 245, 245, 0.6);
        font-style: italic;
        border-top: 1px solid rgba(245, 245, 245, 0.08);
        font-family: 'Playfair Display', serif;
    }
    
    /* Indented lines */
    .indented-line {
        padding-left: 35px;
    }
    
    /* Control items */
    .control-item {
        margin-bottom: 5px;
        margin-left: 5px;
        margin-right: 40px;
    }
    
    /* Poetry text */
    .poetry-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        line-height: 2;
        white-space: pre-wrap;
        color: #F5F5F5;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Right column top margin */
    .right-column-content {
        margin-top: 30px;
    }
    
    /* Button container for width control */
    .button-container {
        text-align: left;
        margin-left: 60px;
    }
            
    
    
    /* Hide elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. Model Definition
# -------------------------------
class BiLSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=768, num_layers=3, dropout=0.2):
        super(BiLSTMLanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

# -------------------------------
# 2. Poetry Generation Function
# -------------------------------
@st.cache_resource
def load_models():
    # Define your model repository
    repo_id = "zaiffi/Mehfil-e-Sukhan"  # Replace with your actual username/model-name
    
    # Download SentencePiece Model
    try:
        sp_model_path = hf_hub_download(repo_id=repo_id, filename="urdu_sp.model")
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
    except Exception as e:
        st.error(f"Error loading SentencePiece model: {e}")
        return None, None
    
    # Initialize & Load the Trained Model
    vocab_size = sp.get_piece_size()
    model = BiLSTMLanguageModel(vocab_size, embed_dim=512, hidden_dim=768, num_layers=3, dropout=0.2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename="model_weights.pth")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return sp, None
    
    return sp, model

def generate_poetry_nucleus(model, sp, start_word, num_words=12, temperature=1.2, top_p=0.85):
    """
    Generates poetry using nucleus (top-p) sampling.
    'num_words' means 1 starting word + (num_words - 1) generated tokens.
    Output is formatted to 6 words per line.
    """
    device = next(model.parameters()).device
    start_ids = sp.encode_as_ids(start_word)
    input_ids = [2] + start_ids  # BOS token=2
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, hidden = model(input_tensor)

    generated_ids = input_ids[:]

    for _ in range(num_words - 1):  # generate one less token than num_words
        last_logits = logits[:, -1, :]
        scaled_logits = last_logits / temperature

        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filtered_indices = cumulative_probs > top_p

        if torch.all(filtered_indices):
            filtered_indices[-1] = False

        sorted_indices = sorted_indices[~filtered_indices]
        sorted_logits = sorted_logits[~filtered_indices]

        if len(sorted_indices) > 0:
            next_token_id = sorted_indices[torch.multinomial(F.softmax(sorted_logits, dim=-1), 1).item()].item()
        else:
            next_token_id = torch.argmax(last_logits).item()

        generated_ids.append(next_token_id)

        next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        logits, hidden = model(next_input, hidden)

    # Decode & format with indentation for every second line
    generated_text = sp.decode_ids(generated_ids[1:])  # skip BOS
    words = generated_text.split()
    
    # Group words into lines of 6 words each
    lines = []
    for i in range(0, len(words), 6):
        line = " ".join(words[i:i+6])
        lines.append(line)
    
    # Format lines with indentation for even-numbered lines (0-indexed)
    formatted_lines = []
    for i, line in enumerate(lines):
        if i % 2 == 1:  # Every second line (1, 3, 5...)
            formatted_lines.append(f'<div class="indented-line">{line}</div>')
        else:
            formatted_lines.append(f'<div>{line}</div>')
    
    formatted_text = "\n".join(formatted_lines)
    return formatted_text

# -------------------------------
# 3. Main Application
# -------------------------------
def main():
    # Load models
    sp, model = load_models()
    
    # Setup session state for storing poetry
    if 'poetry' not in st.session_state:
        st.session_state.poetry = ""
    
    # Define spinner_placeholder at the function level, before it's used
    spinner_placeholder = st.empty()  # Define it here, at the main function level
    
    # Title and subtitle
    st.markdown('<h1 class="title">Mehfil-e-Sukhan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Har Lafz Ek Mehfil</p>', unsafe_allow_html=True)
    
    # Create the true vertical split
    col1, col2 = st.columns([3, 7])  # 30% left, 70% right
    
    # Left Column (30%) - Control Settings
    with col1:
        
        st.markdown('<h2 class="panel-title">Control Settings</h2>', unsafe_allow_html=True)
        
        # Number of Words Control
        st.markdown('<div class="control-item">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Number of Words</div>', unsafe_allow_html=True)
        st.markdown('<div class="description-text">Total words in the generated poetry</div>', unsafe_allow_html=True)
        # Number of Words Control
        num_words = st.selectbox("Number of Words", options=[12, 18, 24, 30, 36, 42, 48], index=0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Creativity (Temperature) Control
        st.markdown('<div class="control-item">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Creativity</div>', unsafe_allow_html=True)
        st.markdown('<div class="description-text">Higher values generate more unique poetry</div>', unsafe_allow_html=True)
        # Creativity (Temperature) Control
        temperature = st.slider("Creativity", 0.5, 2.0, 1.2, 0.1, key="creativity", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Focus (Top-p) Control
        st.markdown('<div class="control-item">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Focus</div>', unsafe_allow_html=True)
        st.markdown('<div class="description-text">Higher focus makes the AI stick to probable words</div>', unsafe_allow_html=True)
        # Focus (Top-p) Control
        top_p = st.slider("Focus", 0.5, 1.0, 0.85, 0.05, key="focus", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right Column (70%) - Input and Output
    with col2:
        # Add a container with top margin
        st.markdown('<div class="right-column-content">', unsafe_allow_html=True)
        
        # Starting Word Input with proper left margin
        st.markdown('<div class="starting-word-label">Starting Word/Phrase</div>', unsafe_allow_html=True)
        # Starting Word Input
        start_word = st.text_input("Starting Word/Phrase", value="ishq", placeholder="Enter a Roman Urdu word", label_visibility="collapsed")
        
        # Generate Button with custom width
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        generate_button = st.button("Generate Poetry")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # And then in your button click handler:
        if generate_button:
            if not sp or not model:
                st.error("Models not properly loaded. Please check the model files.")
            else:
                # Create a placeholder for the spinner
                spinner_placeholder = st.empty()
                
                # Show custom spinner
                spinner_placeholder.markdown(custom_spinner(), unsafe_allow_html=True)
                
                # Generate poetry
                time.sleep(0.1)  # Add slight delay for smooth transition
                st.session_state.poetry = generate_poetry_nucleus(
                    model,
                    sp,
                    start_word,
                    num_words=num_words,
                    temperature=temperature,
                    top_p=top_p
                )
        
        # Clear the spinner once poetry is generated
        spinner_placeholder.empty()
        
        # Display generated poetry
        if st.session_state.poetry:
            st.markdown(f"""
            <div class="poetry-box">
                <div class="poetry-text">
                {st.session_state.poetry}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with quote
    st.markdown("""
    <div class="footer">
        "Poetry is the rhythmical creation of beauty in words" - Edgar Allan Poe
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()