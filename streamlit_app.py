import streamlit as st
import pandas as pd
import textwrap
import numpy as np
import google.generativeai as genai
#from streamlit_gsheets import GSheetsConnection

########################################################
def read_db(file):
    df = pd.read_excel(file)
    df = df.drop(columns = ['Unnamed: 0'])
    df['Embeddings'] = df['Embeddings'].apply(lambda x: [float(i) for i in x.replace('[', '').replace(']', '').split(', ')])
    return df


def find_best_passage(query, dataframe, top_n=1, model='models/embedding-001'):
    """
    Compute the cosine similarity between the query and each document in the dataframe
    using the dot product and normalization.
    """
    query_embedding = genai.embed_content(model=model, content=query, task_type="retrieval_query")["embedding"]

    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = np.stack(dataframe['Embeddings']) / np.linalg.norm(np.stack(dataframe['Embeddings']), axis=1, keepdims=True)

    cosine_similarities = np.dot(embeddings_norm, query_embedding_norm)

    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    top_scores = cosine_similarities[top_indices]

    dataframe['SCORE'] = cosine_similarities
    return dataframe.iloc[top_indices]#, top_scores

def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ").replace("\\", " ")
    prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passages included below. \
    Be sure to respond in a complete sentence, being comprehensive, be exhaustive including all relevant background information. \
    Write the file, title and section you are using to respond at the end of the answer. \
    When necessary use bullet points to list the relevant information. \
    If the passage is irrelevant to the answer, you may ignore it. \
    Break down complicated concepts. \

    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

      ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

##############################################################
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = read_db('data_embeddings.xlsx')

genai.configure(api_key='AIzaSyAyqAdDi1utTQRZQuzbeqVjy6V8BbUzihk')
model_nlp = genai.GenerativeModel('models/gemini-1.5-pro-latest')



##########################################################
st.set_page_config(page_title="Biomicrobot", page_icon=":space_invader:", layout="wide") #robot_face

st.image('logo2.png', width=200, use_container_width=True)

# = st.connection("gsheets", type=GSheetsConnection)

#existing_data = conn.read(worksheet="Hoja 1", usecols=list(range(2)), ttl=5)
#existing_data = existing_data.dropna(how="all")

st.markdown("""
    <div style="display: flex; align-items: center;">
        <div style="margin-right: 20px;">
            <!-- Contenedor para la imagen -->
        </div>
        <h1 style="color: #C27E06; margin: 0;">Welcome to Biomicrobot</h1>
    </div>
""", unsafe_allow_html=True) #FFAD1D

st.write("\n" * 20)

# Descripción
st.markdown("""
    <div style="text-align: center; font-size: 18px;">
        Ask me something:
    </div>
""", unsafe_allow_html=True)

user_input = st.text_input("", "")

chat_container = st.container()

if user_input:
    with chat_container:
        # Mostrar pregunta del usuario
        st.markdown(f"**You:** {user_input}")
        
        # Respuesta simple del chatbot (esto se puede integrar con un modelo de NLP más avanzado)
        # query = "Explain all the materials I should use for the fabrication and testing of a MOX type sensor" 
        passage = find_best_passage(user_input, df, top_n=5)[['TITLE','FILE','SECTION','CONTENT']]
        prompt = make_prompt(user_input, passage)
        answer = model_nlp.generate_content(prompt)

        response = f"**Biomicrobot:** {answer.text}"
        st.markdown(response)
        
    # Botón de reset
    if st.button("Clean chat"):
        st.experimental_rerun()
        
   
