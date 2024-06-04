from openai import OpenAI
import pandas as pd
from datetime import datetime as dt
import qdrant_client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models 
from fastembed import TextEmbedding
import tqdm
import pickle

##
from pydantic import BaseModel
import openai
import instructor
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from langchain_openai import OpenAIEmbeddings

import cohere 

import pdb
# pdb.set_trace()

##########################################################################################################################################

# Function to process embeddings in parallel
def process_embeddings(answer):

    return create_embedding(answer)

# Create a Response Model for the classification task 
class ResponseModel(BaseModel):

    """Response from the AI model based on the given **USER QUESTION**"""
    response: str

def get_embeddings_in_batches(texts, model="text-embedding-ada-002", batch_size=100):

    try:
        with open('dkl_embeddings.pkl', 'rb') as f:
            emb_list = pickle.load(f)
    except Exception as e:
        print(f'emb_list not found')
        emb_list = []

    if len(emb_list) != 0:
        return emb_list
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch_texts,
            model=model
        )
        batch_embeddings = [res.embedding for res in response.data]
        emb_list.extend(batch_embeddings)
        
    with open('dkl_embeddings.pkl', 'wb') as file:
        pickle.dump(emb_list, file)

    return emb_list

def qdrant_search(query_text, qclient, top_k=5):
    
    query_vector = embedding_model.embed_query( query_text )
    search_response = qclient.search(collection_name='rag_contexts', query_vector=query_vector, limit=top_k)
    
    return search_response
    
def create_embedding(text):

    vsize = dkl_vsize

    return openai_client.embeddings.create(input = [text], model="text-embedding-3-small", dimensions = vsize ).data[0].embedding

def setup_qdrant( vsize ):

    # Define collection name and vector configuration
    collection_name = 'rag_contexts'
    vector_params = models.VectorParams(size=vsize, distance=models.Distance.COSINE)

    qdrant_client = QdrantClient(":memory:") # spin up a local instance if you require more advanced features

    # Check if the collection exists
    if qdrant_client.collection_exists(collection_name):
        # Delete the existing collection
        qdrant_client.delete_collection(collection_name)

    # Create the collection with the specified vector configuration
    qdrant_client.create_collection(collection_name, vectors_config=vector_params)
    print(f"Collection '{collection_name}' has been recreated with the specified vector configuration. {vsize}")
    
    return qdrant_client

# this is the function that call co.rerank, the doc will always be taking 1 out of 1000 lines of context
def get_dkl_context( query ):

    dkl_topn = 1
    dkl_context = ''

    chunk_size = 1000  # Number of rows per chunk, adjust based on your memory capacity
    sa = 0
    ea = chunk_size

    idx = 0
    while sa <= len(dkl_doc):

        doc = dkl_doc[ sa : ea ]    
        if idx % 1000 == 0:
            try:
                responses = co.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=doc,
                    top_n=dkl_topn,
                    return_documents=True
                )
            except Exception as e:
#                print(f'corerank: {e}')        
                print(f'query: {query} doc: {len(doc)}')
                dkl_context = ''
                break

            assert len(responses.results) == dkl_topn
            for result in responses.results:
#               print(f'\t{result}')
#               print(f"text: {result.document.text}")
#               print(f"score: {result.relevance_score}")
                dkl_context += result.document.text + ","

        sa += chunk_size
        ea += chunk_size
        if ea > len(dkl_doc):
            ea = len(dkl_doc)
        idx += 1
#        break

    return dkl_context

def get_model_points( df ):

    points = []
    try:
        with open('dkl_points.pkl', 'rb') as f:
            points = pickle.load(f)
    except Exception as e:
        print(f'point_list not found')

    if len(points) != 0:
        return points

    points = []
    for idx, row in (df.iterrows()):
        emb = row['embedding']
    #        emb = emb[ : dkl_vsize ]
        # print(f'{len(emb)}') # 1536
        try:
            point = models.PointStruct(
                id=idx,  # Use the dataframe index as the point ID
                vector=emb, # row['embedding'],  # Convert the embedding to a list
                payload={'id': idx , "text":row['text']}  # Use the label_text as the payload
            )
        except Exception as e:
            print(f'{idx} e = {e}')
            print(f'{point}')
            assert False
        
        points.append(point)
        #    break

    with open('dkl_points.pkl', 'wb') as file:
        pickle.dump(points, file)

    return points

#############################################################################################################

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
@retry(
    stop=stop_after_attempt(2),     # Stop after 2 attempts
    wait=wait_fixed(60),            # Wait 60 second between retries
) #Handle retries for the OpenAI API Rate Limit Calls
def generate_answer(question_id, query_text) -> str:
    # Prepare the OpenAI Request Body
    question = query_text
    contexts = get_dkl_context(question)
#    print(f'con: {contexts}')

    user_message = f"""
        Using the below Contexts: \n\n
        {contexts}\n\n
        **Please Answer the following Question.**\n
        {question}
    """
    # Prepare the OpenAI Request Body
    openai_request_body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        # Response Model is the CategoryModel
        "response_model": ResponseModel,
        "model": "gpt-3.5-turbo",
        "temperature": 0.6, # Adjust the temperature for more creative responses
        "max_tokens": 1000, # Limit the tokens as we are only classifying the text
        "seed": 42,
    }
    try:
        chat_completion = openai_client.chat.completions.create(**openai_request_body)
    except Exception as e:
        raise e
    
    # Assuming the chat_completion returns the category directly
    return chat_completion.response

# Function to generate answers for a single question
def generate_answer_with_id(question_id, question_text):

    try:
        answer = generate_answer(question_id, question_text)
        return question_id, answer
    
    except Exception as e:
        print(f"Error generating answer for question ID {question_id}: {e}")
        return question_id, None
    
#######################################################################################################################################
#
#######################################################################################################################################

if __name__ == "__main__":

    stime = dt.now()

    rm_pkl = [
        "dkl_context.pkl",
        "dkl_embeddings.pkl",
        "dkl_points.pkl"
    ]

    if False:
        for pk in rm_pkl:
            try:
                os.remove( pk )
            except Exception as e:
                print(f'no {pk}')

    dkl_vsize = 768 # 1536
    dkl_vsize = 1536

    api_key = '...'
    client = OpenAI()

    if True:
        embedding_model = OpenAIEmbeddings(api_key=api_key)
#        embedded_query = embedding_model.embed_query("What was the name mentioned in the conversation?")
#        print(f'{len(embedded_query)}')
    else:
        embedding_model = TextEmbedding("BAAI/bge-base-en-v1.5")
#        embedding_model = TextEmbedding("text-3-embedding-small")

    # free trial
    api_key = 'Your key'
    co = cohere.Client(api_key)

    # Load contexts
    contexts_df = pd.read_csv('cohort-2-rag-challenge/contexts.csv')    
    test_data = pd.read_csv('cohort-2-rag-challenge/test.csv')
    train_data = pd.read_csv('cohort-2-rag-challenge/train.csv')

    # Generate embeddings for the text column
    contexts_embeddings = get_embeddings_in_batches(contexts_df['text'].tolist())
    # Add embeddings to the dataframe
    contexts_df['embedding'] = contexts_embeddings
    
    qclient = setup_qdrant( dkl_vsize )
    points = get_model_points( contexts_df )

    qclient.upload_points(collection_name='rag_contexts', points=points)

    # pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 500)
    train_data[['question', 'contexts', 'ground_truth']]

    test_data['id'].nunique()

    # Specify the OpenAI API Key
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Patch the OpenAI client for instructor
    openai_client = instructor.patch(openai.OpenAI(api_key=openai_api_key))

    # save the contect as a list of text 
    dkl_doc = []
    for idx, row in contexts_df.iterrows():
        dkl_doc.append(row['text'])
#    print(f'dkl_doc {len(dkl_doc)}')

    from tenacity import (
        retry,
        stop_after_attempt,
        wait_fixed,
    )
    ## Construct a  PROMPT 
    system_prompt = """
        You are an agent that is specialized in answering medical questions.\n
        Along with the input text, you are provided with the top 10 documents retrieved from a Retrieval-Augmented Generation (RAG) model. 
        Use this information to answer the following **USER QUESTION**
        Note: The documents are included in the user's message for context.
    """    

    # Batch processing of questions
    number_of_batches = 8  
    questions_per_batch = len(test_data) // number_of_batches

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=number_of_batches) as executor:
        # Create a list to hold the futures
        futures = []
        for batch_index in tqdm(range(number_of_batches), desc="Batching questions"):
            start_index = batch_index * questions_per_batch
            end_index = (start_index + questions_per_batch) if batch_index < number_of_batches - 1 else len(test_data)
            # Submit each question in the batch to the executor
            batch_data = test_data.iloc[start_index:end_index]
            for question_id, question_text in zip(batch_data['id'], batch_data['question']):
                futures.append(executor.submit(generate_answer_with_id, question_id, question_text))

        # Collect the results as they complete
        answers = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
            question_id, answer = future.result()
            if answer is not None:
                answers[question_id] = answer

    # Convert the answers to a DataFrame
    answers_df = pd.DataFrame.from_dict(answers, orient='index', columns=['answer'])
    answers_df.reset_index(inplace=True)
    answers_df.rename(columns={'index': 'id'}, inplace=True)
    print(f'{answers_df.head()}')
    answers_df.to_csv( 'answers_df.csv' )
        
    # Number of workers based on the number of available CPUs
    num_workers = os.cpu_count()

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use list comprehension to process embeddings in parallel using executor.map
        embeddings = list(tqdm(executor.map(process_embeddings, answers_df['answer']), total=len(answers_df), desc="Generating embeddings"))

    # Assign the embeddings to the DataFrame
    answers_df['embedding'] = embeddings

    #sort by id and save the dataframe
    answers_df = answers_df.sort_values(by='id')
    answers_df[['id','embedding']].to_csv('submission_visu.6.csv',index=False)

    print(f'done')