#Libraries
import streamlit as st
import pandas as pd
import textwrap
import numpy as np
import google.generativeai as genai

#Functions
def read_db(file):
    df = pd.read_excel(file)
    df = df.drop(columns = ['Unnamed: 0'])
    df['Embeddings'] = df['Embeddings'].apply(lambda x: [float(i) for i in x.replace('[', '').replace(']', '').split(', ')])
    return df


def find_best_passage(query, dataframe, top_n=1, model='models/text-embedding-004'):
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

def make_prompt(query, relevant_passage, ms_prompt_):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ").replace("\\", " ")
    prompt = textwrap.dedent(ms_prompt_ + """

    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

      ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

def show_wellcome_ms (wellcome_ms_='Bienvendio a Biomicrobot'):
    st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="margin-right: 20px;">
                <!-- Contenedor para la imagen -->
            </div>
            <h1 style="color: #00000; margin: 0;">Welcome to Biomicrobot</h1>
        </div>
    """, unsafe_allow_html=True)
    st.write("\n" * 20)
    return None
    

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

genai.configure(api_key=api_key_)
model_nlp = genai.GenerativeModel('models/gemini-2.0-flash') 
#models/gemini-2.5-flash-preview-04-17
#models/gemini-2.0-flash       
#models/gemini-2.5-pro-exp-03-25


st.set_page_config(page_title="Biomicrobot", page_icon=":space_invader:", layout="wide") #robot_face

st.image('logo2.png', width=200, use_container_width=True)

show_wellcome_ms()


idioma = st.selectbox("Selecciona tu idioma / Select your language", ["Español", "English"])

# Acción dependiente del idioma
if idioma == "Español":
    file = 'data_embeddings_es.xlsx'
    ms = 'Me puedes preguntar sobre los protocolos de Biomicrosystems:'
    ms_prompt = """Eres un bot útil e informativo que responde preguntas utilizando el texto de los pasajes de referencia incluidos a continuación.\
        Asegúrate de responder en una oración completa, siendo detallado y exhaustivo, incluyendo toda la información de contexto relevante.\
        Escribe al final de la respuesta el nombre del archivo y título que estás utilizando para responder.\
        Cuando sea necesario, utiliza viñetas para enumerar la información relevante.\
        Si el pasaje no es relevante para la respuesta, puedes ignorarlo.\
        Descompón los conceptos complicados.\
        Si la respuesta no se encuentra explícitamente en los textos, infiere la mejor respuesta posible utilizando tu conocimiento experto.\
        """
    #model = st.selectbox("Selecciona el modelo", ["Pregunta/Respuesta", "Clasificación", "Extracción Entidades"])


elif idioma == "English":
    file = 'data_embeddings_en.xlsx'
    ms = 'You can ask me anything about Biomicrosystems protocols:'
    ms_prompt = """You are a helpful and informative bot that answers questions using text from the reference passages included below. \
    Be sure to respond in a complete sentence, being comprehensive, be exhaustive including all relevant background information. \
    Write the file and title you are using to respond at the end of the answer. \
    When necessary use bullet points to list the relevant information. \
    If the passage is irrelevant to the answer, you may ignore it. \
    Break down complicated concepts. \
    If the answer is not explicitly found in the texts, infer the best possible answer using your expert knowledge.\
        """
    ms_cls = 'Write down the text to classify. The classes are MICROFLUIDICS, SENSORS, MEASUREMENT TECHNIQUES, NANOCOMPOSITES AND NANOSTRUCTURATION | CHEMICAL AND PHYSICAL PROCESSES, ELECTRONICS | PROGRAMMING AND SW DEVELOPMENT'
    #model = st.selectbox("Select the model", ["Question/Answer", "Classification", "Entities Extraction"])

        

#if model == "Pregunta/Respuesta" or model == "Question/Answer":#, "Clasificación", "Extracción Entidades"
df = read_db(file)

# Descripción
st.markdown("""
    <div style="text-align: center; font-size: 18px;">
        """ + ms + """
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
        passage = find_best_passage(user_input, df, top_n=5)[['Title','File','Text']]
        prompt = make_prompt(user_input, passage, ms_prompt)
        answer = model_nlp.generate_content(prompt)

        response = f"**Biomicrobot:** {answer.text}"
        st.markdown(response)        
        
#elif model == "Clasificación" or model == "Classification":#, "Clasificación", "Extracción Entidades"
    #st.markdown('Clasificacion')


if st.button("Clean chat"):
    st.rerun()
