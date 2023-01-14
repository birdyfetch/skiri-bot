import asyncio
import functools
import math
import os
import tempfile
import traceback
import uuid
from typing import Tuple, List, Any

import aiohttp
import backoff
import discord

# An enum of two modes, TOP_P or TEMPERATURE
import requests
from PIL import Image
from aiohttp import RequestInfo
from discord import File


class Mode:
    TOP_P = "top_p"
    TEMPERATURE = "temperature"


class Models:
    DAVINCI = "text-davinci-003"
    CURIE = "text-curie-001"
    EMBEDDINGS = "text-embedding-ada-002"
    EDIT = "text-davinci-edit-001"
    CODE_EDIT = "code-davinci-edit-001"



class Model:
    def __init__(self, usage_service):
        self._mode = Mode.TEMPERATURE
        self._temp = 0.1  # Higher value means more random, lower value means more likely to be a coherent sentence
        self._top_p = 0.9  # 1 is equivalent to greedy sampling, 0.1 means that the model will only consider the top 10% of the probability distribution
        self._max_tokens = 4000  # The maximum number of tokens the model can generate
        self._presence_penalty = (
            0  # Penalize new tokens based on whether they appear in the text so far
        )
        self._frequency_penalty = 0  # Penalize new tokens based on their existing frequency in the text so far. (Higher frequency = lower probability of being chosen.)
        self._best_of = 1  # Number of responses to compare the loglikelihoods of
        self._prompt_min_length = 8
        self._max_conversation_length = 100
        self._model = Models.DAVINCI
        self._low_usage_mode = False
        self.usage_service = usage_service
        self.DAVINCI_ROLES = ["admin", "Admin", "GPT", "gpt"]
        self._summarize_conversations = True
        self._summarize_threshold = 2500
        self.model_max_tokens = 4024
        self._welcome_message_enabled = True
        self._num_static_conversation_items = 8
        self._num_conversation_lookback = 6

        self._hidden_attributes = [
            "usage_service",
            "DAVINCI_ROLES",
            "custom_web_root",
            "_hidden_attributes",
            "model_max_tokens",
            "openai_key",
        ]

        self.openai_key = os.getenv("OPENAI_TOKEN")

    # Use the @property and @setter decorators for all the self fields to provide value checking

    @property
    def num_static_conversation_items(self):
        return self._num_static_conversation_items

    @num_static_conversation_items.setter
    def num_static_conversation_items(self, value):
        value = int(value)
        if value < 3:
            raise ValueError("num_static_conversation_items must be >= 3")
        if value > 20:
            raise ValueError(
                "num_static_conversation_items must be <= 20, this is to ensure reliability and reduce token wastage!"
            )
        self._num_static_conversation_items = value

    @property
    def num_conversation_lookback(self):
        return self._num_conversation_lookback

    @num_conversation_lookback.setter
    def num_conversation_lookback(self, value):
        value = int(value)
        if value < 3:
            raise ValueError("num_conversation_lookback must be >= 3")
        if value > 15:
            raise ValueError(
                "num_conversation_lookback must be <= 15, this is to ensure reliability and reduce token wastage!"
            )
        self._num_conversation_lookback = value

    @property
    def welcome_message_enabled(self):
        return self._welcome_message_enabled

    @welcome_message_enabled.setter
    def welcome_message_enabled(self, value):
        if value.lower() == "true":
            self._welcome_message_enabled = True
        elif value.lower() == "false":
            self._welcome_message_enabled = False
        else:
            raise ValueError("Value must be either true or false!")

    @property
    def summarize_threshold(self):
        return self._summarize_threshold

    @summarize_threshold.setter
    def summarize_threshold(self, value):
        value = int(value)
        if value < 800 or value > 4000:
            raise ValueError(
                "Summarize threshold cannot be greater than 4000 or less than 800!"
            )
        self._summarize_threshold = value

    @property
    def summarize_conversations(self):
        return self._summarize_conversations

    @summarize_conversations.setter
    def summarize_conversations(self, value):
        # convert value string into boolean
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            raise ValueError("Value must be either true or false!")
        self._summarize_conversations = value

    @property
    def low_usage_mode(self):
        return self._low_usage_mode

    @low_usage_mode.setter
    def low_usage_mode(self, value):
        # convert value string into boolean
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            raise ValueError("Value must be either true or false!")

        if value:
            self._model = Models.CURIE
            self.max_tokens = 1900
            self.model_max_tokens = 1000
        else:
            self._model = Models.DAVINCI
            self.max_tokens = 4000
            self.model_max_tokens = 4024

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model not in [Models.DAVINCI, Models.CURIE]:
            raise ValueError(
                "Invalid model, must be text-davinci-003 or text-curie-001"
            )
        self._model = model

    @property
    def max_conversation_length(self):
        return self._max_conversation_length

    @max_conversation_length.setter
    def max_conversation_length(self, value):
        value = int(value)
        if value < 1:
            raise ValueError("Max conversation length must be greater than 1")
        if value > 500:
            raise ValueError(
                "Max conversation length must be less than 500, this will start using credits quick."
            )
        self._max_conversation_length = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in [Mode.TOP_P, Mode.TEMPERATURE]:
            raise ValueError("mode must be either 'top_p' or 'temperature'")
        if value == Mode.TOP_P:
            self._top_p = 0.1
            self._temp = 0.7
        elif value == Mode.TEMPERATURE:
            self._top_p = 0.9
            self._temp = 0.6

        self._mode = value

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, value):
        value = float(value)
        if value < 0 or value > 1:
            raise ValueError(
                "temperature must be greater than 0 and less than 1, it is currently "
                + str(value)
            )

        self._temp = value

    @property
    def top_p(self):
        return self._top_p

    @top_p.setter
    def top_p(self, value):
        value = float(value)
        if value < 0 or value > 1:
            raise ValueError(
                "top_p must be greater than 0 and less than 1, it is currently "
                + str(value)
            )
        self._top_p = value

    @property
    def max_tokens(self):
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value):
        value = int(value)
        if value < 15 or value > 4096:
            raise ValueError(
                "max_tokens must be greater than 15 and less than 4096, it is currently "
                + str(value)
            )
        self._max_tokens = value

    @property
    def presence_penalty(self):
        return self._presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value):
        if int(value) < 0:
            raise ValueError(
                "presence_penalty must be greater than 0, it is currently " + str(value)
            )
        self._presence_penalty = value

    @property
    def frequency_penalty(self):
        return self._frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value):
        if int(value) < 0:
            raise ValueError(
                "frequency_penalty must be greater than 0, it is currently "
                + str(value)
            )
        self._frequency_penalty = value

    @property
    def best_of(self):
        return self._best_of

    @best_of.setter
    def best_of(self, value):
        value = int(value)
        if value < 1 or value > 3:
            raise ValueError(
                "best_of must be greater than 0 and ideally less than 3 to save tokens, it is currently "
                + str(value)
            )
        self._best_of = value

    @property
    def prompt_min_length(self):
        return self._prompt_min_length

    @prompt_min_length.setter
    def prompt_min_length(self, value):
        value = int(value)
        if value < 10 or value > 4096:
            raise ValueError(
                "prompt_min_length must be greater than 10 and less than 4096, it is currently "
                + str(value)
            )
        self._prompt_min_length = value

    def backoff_handler(details):
        print(
            f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries calling function {details['target']} | "
            f"{details['exception'].status}: {details['exception'].message}"
        )

    async def valid_text_request(self, response):
        try:
            tokens_used = int(response["usage"]["total_tokens"])
            await self.usage_service.update_usage(tokens_used)
        except:
            raise ValueError(
                "The API returned an invalid response: "
                + str(response["error"]["message"])
            )

    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        factor=3,
        base=5,
        max_tries=4,
        on_backoff=backoff_handler,
    )
    async def send_embedding_request(self, text, custom_api_key=None):
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            payload = {
                "model": Models.EMBEDDINGS,
                "input": text,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key if not custom_api_key else custom_api_key}",
            }
            async with session.post(
                "https://api.openai.com/v1/embeddings", json=payload, headers=headers
            ) as resp:
                response = await resp.json()

                try:
                    return response["data"][0]["embedding"]
                except Exception:
                    print(response)
                    traceback.print_exc()
                    return

    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        factor=3,
        base=5,
        max_tries=6,
        on_backoff=backoff_handler,
    )
    async def send_edit_request(
        self,
        instruction,
        input=None,
        temp_override=None,
        top_p_override=None,
        codex=False,
        custom_api_key=None,
    ):

        # Validate that  all the parameters are in a good state before we send the request
        if len(instruction) < self.prompt_min_length:
            raise ValueError(
                "Instruction must be greater than 8 characters, it is currently "
                + str(len(instruction))
            )

        print(
            f"The text about to be edited is [{input}] with instructions [{instruction}] codex [{codex}]"
        )
        print(f"Overrides -> temp:{temp_override}, top_p:{top_p_override}")

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            payload = {
                "model": Models.EDIT if codex is False else Models.CODE_EDIT,
                "input": "" if input is None else input,
                "instruction": instruction,
                "temperature": self.temp if temp_override is None else temp_override,
                "top_p": self.top_p if top_p_override is None else top_p_override,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key if not custom_api_key else custom_api_key}",
            }
            async with session.post(
                "https://api.openai.com/v1/edits", json=payload, headers=headers
            ) as resp:
                response = await resp.json()
                await self.valid_text_request(response)
                return response

    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        factor=3,
        base=5,
        max_tries=6,
        on_backoff=backoff_handler,
    )
    async def send_moderations_request(self, text):
        # Use aiohttp to send the above request:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key}",
            }
            payload = {"input": text}
            async with session.post(
                "https://api.openai.com/v1/moderations",
                headers=headers,
                json=payload,
            ) as response:
                return await response.json()

    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        factor=3,
        base=5,
        max_tries=4,
        on_backoff=backoff_handler,
    )
    async def send_summary_request(self, prompt, custom_api_key=None):
        """
        Sends a summary request to the OpenAI API
        """
        summary_request_text = []
        summary_request_text.append(
            "The following is a conversation instruction set and a conversation"
            " between two people, a <username>, and GPTie. Firstly, determine the <username>'s name from the conversation history, then summarize the conversation. Do not summarize the instructions for GPTie, only the conversation. Summarize the conversation in a detailed fashion. If <username> mentioned their name, be sure to mention it in the summary. Pay close attention to things the <username> has told you, such as personal details."
        )
        summary_request_text.append(prompt + "\nDetailed summary of conversation: \n")

        summary_request_text = "".join(summary_request_text)

        tokens = self.usage_service.count_tokens(summary_request_text)

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            payload = {
                "model": Models.DAVINCI,
                "prompt": summary_request_text,
                "temperature": 0.5,
                "top_p": 1,
                "max_tokens": self.max_tokens - tokens,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "best_of": self.best_of,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_key if not custom_api_key else custom_api_key}",
            }
            async with session.post(
                "https://api.openai.com/v1/completions", json=payload, headers=headers
            ) as resp:
                response = await resp.json()

                await self.valid_text_request(response)

                # print(response["choices"][0]["text"])

                return response

    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        factor=3,
        base=5,
        max_tries=4,
        on_backoff=backoff_handler,
    )
    async def send_request(
        self,
        prompt,
        tokens,
        temp_override=None,
        top_p_override=None,
        best_of_override=None,
        frequency_penalty_override=None,
        presence_penalty_override=None,
        max_tokens_override=None,
        model=None,
        custom_api_key=None,
    ) -> (
        dict,
        bool,
    ):  # The response, and a boolean indicating whether or not the context limit was reached.

        # Validate that  all the parameters are in a good state before we send the request
        if len(prompt) < self.prompt_min_length:
            raise ValueError(
                "Prompt must be greater than 8 characters, it is currently "
                + str(len(prompt))
            )

        print("The prompt about to be sent is " + prompt)
        print(
            f"Overrides -> temp:{temp_override}, top_p:{top_p_override} frequency:{frequency_penalty_override}, presence:{presence_penalty_override}"
        )

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            payload = {
                "model": self.model if model is None else model,
                "prompt": prompt,
                "temperature": self.temp if temp_override is None else temp_override,
                "top_p": self.top_p if top_p_override is None else top_p_override,
                "max_tokens": self.max_tokens - tokens
                if not max_tokens_override
                else max_tokens_override,
                "presence_penalty": self.presence_penalty
                if presence_penalty_override is None
                else presence_penalty_override,
                "frequency_penalty": self.frequency_penalty
                if frequency_penalty_override is None
                else frequency_penalty_override,
                "best_of": self.best_of if not best_of_override else best_of_override,
            }
            headers = {
                "Authorization": f"Bearer {self.openai_key if not custom_api_key else custom_api_key}"
            }
            async with session.post(
                "https://api.openai.com/v1/completions", json=payload, headers=headers
            ) as resp:
                response = await resp.json()
                # print(f"Payload -> {payload}")
                # Parse the total tokens used for this request and response pair from the response
                await self.valid_text_request(response)
                print(f"Response -> {response}")

                return response

    @staticmethod
    async def send_test_request(api_key):

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": Models.CURIE,
                "prompt": "test.",
                "temperature": 1,
                "top_p": 1,
                "max_tokens": 10,
            }
            headers = {"Authorization": f"Bearer {api_key}"}
            async with session.post(
                "https://api.openai.com/v1/completions", json=payload, headers=headers
            ) as resp:
                response = await resp.json()
                try:
                    int(response["usage"]["total_tokens"])
                except:
                    raise ValueError(str(response["error"]["message"]))

                return response
