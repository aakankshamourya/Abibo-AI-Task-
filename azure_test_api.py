import os
from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

from langchain.chat_models import AzureChatOpenAI
llm = AzureChatOpenAI(
        model_name="gpt-4o-mini",
        deployment_name="gpt-4o-mini",
        n=1,
        temperature=0,
        openai_api_type = "azure",
        openai_api_base = "https://gpt4o-mini-endpoint.openai.azure.com/",
        openai_api_version = "2023-03-15-preview",
        openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key"))

conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm, k=2)
)

print(conversation.predict(input="What is india?"))