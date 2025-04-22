import asyncio
import signal
import threading
from typing import Optional
import logging
import queue
import os

from pydantic_settings import BaseSettings, SettingsConfigDict
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.models.agent import AgentConfig, LangchainAgentConfig
from vocode.streaming.agent.langchain_agent import LangchainAgent
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

configure_pretty_logging()

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    azure_speech_key: str = os.getenv("AZURE_SPEECH_KEY", "")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    azure_speech_region: str = os.getenv("AZURE_SPEECH_REGION", "eastus")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

class PoemLangchainAgentConfig(LangchainAgentConfig):
    poem_topic: str = "technology"  # Default topic

class CustomLangchainAgent(LangchainAgent):
    def create_chain(self):
        prompt = ChatPromptTemplate.from_template(
            f"You are a poetic assistant. Create a short poem about {self.agent_config.poem_topic} "
            "that also answers the user's question: {input}"
        )
        
        llm = ChatOpenAI(
            model_name=self.agent_config.model_name,
            temperature=self.agent_config.temperature,
            openai_api_key=self.agent_config.openai_api_key
        )
        
        chain = prompt | llm | StrOutputParser()
        return chain

class CustomAgentFactory(AgentFactory):
    def create_agent(
        self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None
    ) -> BaseAgent:
        if isinstance(agent_config, PoemLangchainAgentConfig):
            return CustomLangchainAgent(
                agent_config=agent_config,
                logger=logger
            )
        elif isinstance(agent_config, LangchainAgentConfig):
            return LangchainAgent(
                agent_config=agent_config,
                logger=logger
            )
        else:
            raise Exception("Invalid agent config")

class ConversationManager:
    def __init__(self):
        self.conversation = None
        self.microphone_input = None
        self.speaker_output = None
        self.event_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = False

    async def start_conversation(self, poem_topic: str):
        self.microphone_input, self.speaker_output = create_streaming_microphone_input_and_speaker_output(
            use_default_devices=False,
            use_blocking_speaker_output=True,
        )

        agent_factory = CustomAgentFactory()

        self.conversation = StreamingConversation(
            output_device=self.speaker_output,
            transcriber=DeepgramTranscriber(
                DeepgramTranscriberConfig.from_input_device(
                    self.microphone_input,
                    endpointing_config=PunctuationEndpointingConfig(),
                    api_key=settings.deepgram_api_key,
                ),
            ),
            agent=agent_factory.create_agent(
                PoemLangchainAgentConfig(
                    openai_api_key=settings.openai_api_key,
                    initial_message=BaseMessage(text=f"Hello! I'm your sales assistant. I'll be writing poems about {poem_topic}. What would you like to discuss?"),
                    prompt_preamble=f"The assistant is a creative poet who responds to questions with short poems about {poem_topic}.",
                    model_name="gpt-3.5-turbo",
                    temperature=0.7,
                    poem_topic=poem_topic
                )
            ),
            synthesizer=AzureSynthesizer(
                AzureSynthesizerConfig.from_output_device(self.speaker_output),
                azure_speech_key=settings.azure_speech_key,
                azure_speech_region=settings.azure_speech_region,
            ),
        )

        await self.conversation.start()
        self.is_running = True

        while self.is_running:
            chunk = await self.microphone_input.get_audio()
            self.conversation.receive_audio(chunk)
            
            # Get transcription and response updates
            if hasattr(self.conversation.transcriber, 'transcription_queue'):
                try:
                    transcription = self.conversation.transcriber.transcription_queue.get_nowait()
                    self.transcription_queue.put(transcription)
                except queue.Empty:
                    pass
            
            if hasattr(self.conversation.agent, 'response_queue'):
                try:
                    response = self.conversation.agent.response_queue.get_nowait()
                    self.response_queue.put(response)
                except queue.Empty:
                    pass

    async def stop_conversation(self):
        if self.conversation:
            await self.conversation.terminate()
            self.is_running = False
        if self.microphone_input:
            self.microphone_input.tear_down()
        if self.speaker_output:
            self.speaker_output.tear_down()

def run_conversation(manager: ConversationManager, poem_topic: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(manager.start_conversation(poem_topic))

def main():
    st.title("🎙️ Poetic Voice Assistant")
    st.markdown("""
    This app creates a voice conversation with an AI poet. 
    Speak to the assistant and it will respond with poetic answers!
    """)

    # Initialize session state
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        poem_topic = st.text_input("Poem Topic", "technology and human connection")
        
        if st.button("Update Topic"):
            st.success(f"Topic updated to: {poem_topic}")

        st.markdown("---")
        st.markdown("### API Keys")
        settings.openai_api_key = st.text_input("OpenAI API Key", settings.openai_api_key, type="password")
        settings.azure_speech_key = st.text_input("Azure Speech Key", settings.azure_speech_key, type="password")
        settings.deepgram_api_key = st.text_input("Deepgram API Key", settings.deepgram_api_key, type="password")

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.header("Conversation")
        transcription_placeholder = st.empty()
        transcription_placeholder.markdown("**Your speech will appear here...**")

    with col2:
        st.header("AI Response")
        response_placeholder = st.empty()
        response_placeholder.markdown("**AI responses will appear here...**")

    # Control buttons
    if not st.session_state.conversation_active:
        if st.button("Start Conversation", type="primary"):
            st.session_state.conversation_active = True
            thread = threading.Thread(
                target=run_conversation,
                args=(st.session_state.conversation_manager, poem_topic),
                daemon=True
            )
            add_script_run_ctx(thread)
            thread.start()
            st.rerun()
    else:
        if st.button("Stop Conversation", type="secondary"):
            asyncio.run(st.session_state.conversation_manager.stop_conversation())
            st.session_state.conversation_active = False
            st.rerun()

    # Update UI with conversation data
    while st.session_state.conversation_active:
        try:
            transcription = st.session_state.conversation_manager.transcription_queue.get_nowait()
            transcription_placeholder.markdown(f"**You:** {transcription}")
        except queue.Empty:
            pass

        try:
            response = st.session_state.conversation_manager.response_queue.get_nowait()
            response_placeholder.markdown(f"**AI:** {response}")
        except queue.Empty:
            pass

if __name__ == "__main__":
    main()