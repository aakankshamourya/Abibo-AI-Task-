import pandas as pd
import spacy
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key
from typing import Any, Dict, Optional, List
from itertools import islice
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.pydantic_v1 import BaseModel, Field

# Replace 'en_core_web_sm' with the appropriate spaCy model
nlp = spacy.load('en_core_web_sm')

# Replace this with your actual dataframe
dataframe = pd.read_csv('chat_history.csv')

class ConversationEntityMemory(BaseChatMemory):
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    dataframe: pd.DataFrame = 'dataframe'
    k: int = 'k'
#     entity_extraction_prompt: BasePromptTemplate
#     entity_summarization_prompt: BasePromptTemplate

    def __init__(self, dataframe:pd.DataFrame, k:int=k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataframe = dataframe
        self.k = k
        
    def memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement this method to return the memory variables as a dictionary.
        """
        # Return the memory variables as a dictionary
        return {
            "entities": self.entity_cache,
            self.chat_history_key: self.get_chat_history(),
        }

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_email = inputs.get('user_email', None)  # Replace with your user identifier
        if user_email:
            # Filter the dataframe based on user_email and get the last 2 questions and answers
            user_data = self.dataframe[self.dataframe['user_email'] == user_email]
            user_data = user_data[['question', 'answer']].tail(self.k)

            # Extract questions and answers from the filtered dataframe
            questions = user_data['question'].tolist()
            answers = user_data['answer'].tolist()

            # Combine questions and answers into chat history
            chat_history = []
            for i in range(len(questions)):
                chat_history.append(f'Human: {questions[i]}')
                chat_history.append(f'AI: {answers[i]}')

            # Return the chat history
            chat_history_text = "\n".join(chat_history)

            # Extract entities from the chat history
            entities = self.extract_entities(chat_history_text)

            # Return chat history and extracted entities
            return {self.chat_history_key: chat_history_text, 'entities': entities}
        
        return {self.chat_history_key: '', 'entities': {}}

    def extract_entities(self, text: str) -> Dict[str, str]:
        doc = nlp(text)
        entities = {}
        for ent in doc.ents:
            entities[ent.text] = ""  # Initialize entity summaries as empty strings
        return entities

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        user_email = inputs.get('user_email', None)  # Replace with your user identifier
        if user_email:
            # Get the last question and answer
            user_data = self.dataframe[self.dataframe['user_email'] == user_email]
            user_data = user_data[['question', 'answer']].tail(1)
            last_question = user_data['question'].values[0]
            last_answer = user_data['answer'].values[0]

            if last_question and last_answer:
                # Create a chat history entry for the last question and answer
                chat_history_entry = f'Human: {last_question}\nAI: {last_answer}'

                # Update the chat history in the chat memory
                self.buffer.append(BaseMessage(content=chat_history_entry))

                # Extract entities from the new chat history entry
                new_entities = self.extract_entities(chat_history_entry)

                # Merge the new entities with existing entities
                existing_entities = self.memory['entities']
                all_entities = {**existing_entities, **new_entities}

                # Update the entity cache
                self.memory['entities'] = all_entities

        super().save_context(inputs, outputs)
        
        

