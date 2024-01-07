import streamlit as st
# from streamlit_chat import message
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()
st.cache_resource.clear()
st.cache_data.clear()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Circular Economy Business Ideas: How good are they? Are they any good? Let's find out!", page_icon=":robot:")
st.header("Circular Economy Business Ideas")
st.subheader("How good are they? Are they any good? Let's find out!")


model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name='gpt-4', temperature=0.2)
problem_validation = ChatPromptTemplate.from_template("I am trying to solve this problem: {problem}. Evaluate whether this problem is important for sustainability, whether it already has a solution, and whether it is a problem that is likely to be solved in the future. Respond in under 100 words.")
problem_validation_chain = problem_validation | model

solution_validation = ChatPromptTemplate.from_template("I am trying to solve this problem: {problem}. This is my proposed solution: {solution}. Evaluate if my solution addresses the problem, how sustainable it is, if it is a good solution, and if it is a solution that is possible to implement.")
solution_validation_chain = solution_validation | model

qa = ChatPromptTemplate.from_template("I am trying to solve this problem: {problem}. This is my proposed solution: {solution}. I have the following question: {question}")
qa_chain = qa | model

chains = [problem_validation_chain, solution_validation_chain, qa_chain]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_state" not in st.session_state:
    st.session_state.input_state = 0

if "problem" not in st.session_state:
    st.session_state.problem = "no problem specified"
if "solution" not in st.session_state:
    st.session_state.solution = "no solution specified"
if "question" not in st.session_state:
    st.session_state.question = "no question specified"

if st.button("New session"):
    st.session_state.messages = []
    st.session_state.input_state = 0
    st.session_state.problem = "no problem specified"
    st.session_state.solution = "no solution specified"
    st.session_state.question = "no question specified"


if st.session_state.messages == []:
    st.session_state.messages.append({"role": "assistant", "content": "Enter a real world problem you could solve with the help of your idea and let's get started!"})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Message the assistant..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    if st.session_state.input_state == 0:
        st.session_state.problem = user_input
    elif st.session_state.input_state == 1:
        st.session_state.solution = user_input
    else:
        st.session_state.question = user_input
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # print(st.session_state.input_state, st.session_state.problem, st.session_state.solution, st.session_state.question)
        for response in chains[st.session_state.input_state].stream({"problem": st.session_state.problem, "solution": st.session_state.solution, "question": st.session_state.question}):
            full_response += (response.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    if st.session_state.input_state == 0:
        with st.chat_message("assistant"):
            full_response = "Now enter your idea"
            st.markdown("Now enter your idea")
            st.session_state.input_state = 1
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    elif st.session_state.input_state == 1:
        with st.chat_message("assistant"):
            full_response = "Enter any questions you have about this solution"
            st.markdown("Enter any questions you have about this solution")
            st.session_state.input_state = 2
            st.session_state.messages.append({"role": "assistant", "content": full_response})


