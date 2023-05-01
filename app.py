# Automated GPT prompt to describe topics and learn them
# @since April 30th 2023

# imports
import os 
from apikey import apikey
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# import api key from apikey.py
os.environ['OPENAI_API_KEY'] = apikey

# framework
st.title('GPT++ for learning')
prompt = st.text_input('Prompt topic:')

# template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='Identify the 20% of {topic} that will yield 80% of the desired results and provide a focused learning plan to master it.' +
    'Create a separate output for this prompt: Explain {topic} in the simplest terms possible as if teaching it to a complete beginner. Identify gaps in my understanding and suggest resources to fill them.'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)


# Items on screen
if prompt: 
    title = title_chain.run(prompt)

    st.write(title)

    with st.expander('History'):
        st.info(title_memory.buffer)