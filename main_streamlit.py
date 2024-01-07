import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
            ChatPromptTemplate,
            SystemMessagePromptTemplate,
            AIMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )
import os
from dotenv import load_dotenv
load_dotenv()
st.cache_resource.clear()
st.cache_data.clear()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
from langchain_openai import ChatOpenAI

template = """

        The following is a friendly conversation between a human and an AI. 
        The AI is talkative and provides lots of specific details from its context. 
        If the AI does not know the answer to a question, it truthfully says it does
        not know.

        Current conversation:
        Human: {input}
        AI Assistant:"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

example_human_history = HumanMessagePromptTemplate.from_template("Hi")
example_ai_history = AIMessagePromptTemplate.from_template("hello, how are you today?")

human_template="{input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human_history, example_ai_history, human_message_prompt])

chat = ChatOpenAI(api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.2)


# From here down is all the StreamLit UI.
st.set_page_config(page_title="Cicular Economy Business Ideas: How good are they? Are they moonshots? Let's find out!", page_icon=":robot:")
st.header("Cicular Economy Business Ideas")
st.subheader("How good are they? Are they moonshots? Let's find out!")

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    chain = ConversationChain(
        chat=chat
    )
    return chain


if "chain" not in st.session_state:
    st.session_state["chain"] = load_chain()

chain = st.session_state["chain"]

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    print(user_input)
    output = chat.generate() #????
    print(output)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["past"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")