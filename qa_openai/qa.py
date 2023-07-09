import os
from typing import Literal
import pathlib

import pandas as pd

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

class qa_openai :
    """
    A class used to represent an question answer system using OpenAI apis

    """

    def __init__(self, source_file:str = "content/source_data/KnowledgeDocument.txt", 
                sample_qa_file:str = "content/source_data/SampleQuestions.csv") -> None:
        """
        Parameters
        ----------
        source_file : str
            The path of the markdown file.
        
        sample_qa_file : str
            The path of the sample question answer file in csv format with headers ["Question", "Ideal Answer"].
        """
        
        self.api_saver_mode = False # set as True while development, to save api calls to GPT

        self.source_file = source_file
        self.sample_qa_file = sample_qa_file
        self.embedding = OpenAIEmbeddings()
        self.vectordb = None    # defered init
        self.vectordb_for_metric = None # defered init
        self.qa_dict = None # defered init

        self.llm_models = {
            'gpt-3.5-turbo' : ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
            'gpt-4' : ChatOpenAI(model_name='gpt-4', temperature=0)
        }

        # This will be initialized after embedings are creared. 
        self.qa_chain = {
            'gpt-3.5-turbo' : None,
            'gpt-4' : None
        }


    
    def load_markdown_docs(self)  -> None:
        """Loads the markdon file.

        """
        
        abs_path = os.path.join(os.getcwd(), self.source_file)
        print(f"Loading markdown from {abs_path}")
        # path = "content/source_data/KnowledgeDocument.txt"
        loader = TextLoader(abs_path, encoding='utf-8')
        docs = loader.load()
        self.markdown_document = ' '.join([d.page_content for d in docs])
        print(f"Loaded markdown file.")

    
    def split_markdown_data(self, basic_cleanup:bool = True) -> None:
        """Splits the markdown file into chunks.

        Parameters
        ----------
        basic_cleanup : boolean, optional (default is True)
            Specify whether the loaded documents to be cleaned, currently only * are removed. 

        """
        
        headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        self.md_header_splits_all = markdown_splitter.split_text(self.markdown_document)
        self.md_header_splits_all = self.md_header_splits_all[0:3] if self.api_saver_mode else self.md_header_splits_all

        if basic_cleanup:
            '''
            Although llm is able to handle dirty chars, lets remove all astrics from original content.
            We may get very minor improvment in answers from llm
            '''
            import re
            regex_search_term = '\**'
            regex_replacement = ''
            for doc in self.md_header_splits_all:
                doc.page_content = re.sub(regex_search_term, regex_replacement, doc.page_content)
                for each in doc.metadata:
                    doc.metadata[each] = re.sub(regex_search_term, regex_replacement, doc.metadata[each])

        print(f"Total document chunks: {len(self.md_header_splits_all)}") 

    
    def load_embedings(self) -> None :
        """Loads the data embedings in vector database.

        We tried using Chroma which works very well for Pan Card service, but installation on windows give errors to many people.
        Lets try FAISS (Facebook AI Similarity Search) datastore. 

        """
        
        '''
        self.vectordb = Chroma.from_documents(
            documents = self.md_header_splits_all,
            embedding = self.embedding,
            # persist_directory = self.persist_directory # feature to persist the embedings. if not specified in-memory is used. 
        )
        print(f"Knowledge doc chunks in vectordb: self.vectordb._collection.count()")
        '''

        self.vectordb = FAISS.from_documents(
            documents = self.md_header_splits_all,
            embedding = self.embedding
        )

    
    def build_qa_chain(self) -> None :
        """Builds the prompt and qa chain.
        Prompt is technique using with we instruct GPT, what to do with inputs it receives.
        qa_chain is build for two models ['gpt-3.5-turbo', 'gpt-4'] to simplify access to LLM

        """

        # Build prompt
        template = """Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                    Use three sentences maximum. Keep the answer as concise as possible. 
                    Always say "thanks for asking!" at the end of the answer. 
                    {context}
                    Question: {question}
                    Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Build chain
        self.qa_chain['gpt-3.5-turbo'] = RetrievalQA.from_chain_type(
                self.llm_models['gpt-3.5-turbo'],
                retriever = self.vectordb.as_retriever(),
                return_source_documents=False,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        self.qa_chain['gpt-4'] = RetrievalQA.from_chain_type(
                self.llm_models['gpt-4'],
                retriever = self.vectordb.as_retriever(),
                return_source_documents=False,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        print("Ready for answring questions.")


    def build_score_system(self) -> None :
        """This function prepares recipy for calculating score for the answer from LLM.

        Here, we use questions and answers curated by human to compare with GPT answers.
        Its feasible to use such technique for small dataset like Pan card service. 
        For bigger dataset we can use LLM based question generated using chunks from source documents, and then manualy reduce irelevent questions.

        sample questions are stored in vecotordb in embedings format.
        the associated sample answer is then compaired with LLM's answer uisng various techniqes like eucledien distance, dot product of embedings, etc.

        In future we can evaluate QA system using other techniques proposed in various research papers given below.
        BERT_Score, Perplexity, QAEval, etc

        """

        abs_path = os.path.join(os.getcwd(), self.sample_qa_file)
        annotations = pd.read_csv(abs_path)
        annotations_dict = annotations.to_dict(orient='records')
        self.qa_dict = {each['Question'] : each['Ideal Answer'] for each in annotations_dict }
        all_qns = [d['Question'] for d in annotations_dict]
        all_qns = all_qns[0:3] if self.api_saver_mode else all_qns
        print(f"Total qns chunks for score: {len(self.md_header_splits_all)}") 


        # NOTE setting chunk_size=1000 to get full question as one sentence. Assuming question will have max 1000 char lenth
        txt_splitter = RecursiveCharacterTextSplitter.from_language(
                            language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=0
                        )
        txt_docs = txt_splitter.create_documents(all_qns)
        
        '''
        # persist_directory = 'content/FAISS_metric/'
        self.vectordb_for_metric = FAISS.from_documents(
            documents=txt_docs,
            embedding=self.embedding,
            # persist_directory=persist_directory # Required only if we want to persist the embedings to reduce API calls
            )
        print(f"metric questions in vectordb: self.vectordb._collection.count()")
        '''

        self.vectordb_for_metric = FAISS.from_documents(
            documents=txt_docs,
            embedding=self.embedding,
            )
        

    def run_qa_chain(self, model: Literal['gpt-3.5-turbo', 'gpt-4'], question: str) -> dict:
        """This function actually queries the LLM.

        """

        result = self.qa_chain[model]({"query": question})
        print(result)
        return result


    def cal_score(self, question:str, llm_answer:str) -> float:
        """Calculates the how meaningful the LLM's answer is w.r.t human curated answer.

        Parameters
        ----------
        question : str, 
            Question asked to LLM

        llm_answer : str, 
            Answer generated by LLM

        """
        
        matching_qn = self.vectordb_for_metric.similarity_search(question, k=3)[0]
        ideal_answer = self.qa_dict[matching_qn.page_content]
        
        llm_answer_embedding = self.embedding.embed_query(llm_answer)
        ideal_answer_embedding = self.embedding.embed_query(ideal_answer)

        import numpy as np
        score = np.dot(llm_answer_embedding, ideal_answer_embedding) # dot product of embedings can be used as simple score 

        # P, R, F1 = score([llm_answer], [ideal_answer], lang="en", verbose=True)
        # print(f"P={P}, R={R}, F1={F1}")
        return score


    def setup(self) :
        self.load_markdown_docs()
        self.split_markdown_data()
        self.load_embedings()
        self.build_qa_chain()
        self.build_score_system()
    



