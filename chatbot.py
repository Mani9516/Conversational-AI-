!pip install transformers
!pip install streamlit

import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

# Download Microsoft's DialoGPT model and tokenizer
checkpoint = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# A ChatBot class with emoji support
class ChatBot():
    def __init__(self):
        self.chat_history_ids = None
        self.bot_input_ids = None

    def user_input(self, text):
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            return '👋 See you soon! Bye!'

        self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1)
        else:
            self.bot_input_ids = self.new_user_input_ids

        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        if response == "":
            response = self.random_response()

        return "🤖: " + response + " 😊"
        
    def random_response(self):
        i = -1
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], skip_special_tokens=True)
        while response == '':
            i = i-1
            response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], skip_special_tokens=True)
        
        if response.strip() == '?':
            reply = np.random.choice(["🤔 I don't know", "😕 I am not sure"])
        else:
            reply = np.random.choice(["👍 Great", "👌 Fine. What's up?", "😊 Okay"])
        return reply

# Build a ChatBot object
bot = ChatBot()

# Streamlit interface
st.title("📱 Chat with Me")
st.write("Chat with an AI-powered bot using DialoGPT. Type a message and let's chat! 🚀")

# Input from user
user_input = st.text_input("You: ", "")

# Display the conversation
if user_input:
    bot_response = bot.user_input(user_input)
    st.text_area("ChatBot:", value=bot_response, height=100, max_chars=None, key=None)
    st.text("")  # Space for new message input

# Custom styling for Snapchat-like feel
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #1c1c1c;
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    input[type="text"] {
        border-radius: 20px;
        padding: 10px;
        border: 1px solid #ffffff;
        color: #ffffff;
        background-color: #2a2a2a;
    }
    .stTextArea {
        background-color: #2a2a2a;
        border-radius: 20px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #444;
    }
    </style>
    """,
    unsafe_allow_html=True
)
