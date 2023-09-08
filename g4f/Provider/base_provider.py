from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import browser_cookie3

from ..typing import Any, AsyncGenerator, CreateResult, Union


class BaseProvider(ABC):
    url: str
    working               = False
    needs_auth            = False
    supports_stream       = False
    supports_gpt_35_turbo = False
    supports_gpt_4        = False

    @staticmethod
    @abstractmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True,
        proxy: str = None,
        **kwargs: Any) -> CreateResult:
        
        raise NotImplementedError()

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("proxy", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
    

_cookies = {}

def get_cookies(cookie_domain: str) -> dict:
    if cookie_domain not in _cookies:
        _cookies[cookie_domain] = {}
        try:
            for cookie in browser_cookie3.load(cookie_domain):
                _cookies[cookie_domain][cookie.name] = cookie.value
        except:
            pass
    return _cookies[cookie_domain]


def format_prompt(messages: list[dict[str, str]], add_special_tokens=False):
    if add_special_tokens or len(messages) > 1:
        formatted = "\n".join(
            ["%s: %s" % ((message["role"]).capitalize(), message["content"]) for message in messages]
        )
        return f"{formatted}\nAssistant:"
    else:
        return messages.pop()["content"]



class AsyncProvider(BaseProvider):
    @classmethod
    async def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        stream: bool = False, **kwargs: Any) -> CreateResult:
        
        # yield asyncio.run(cls.create_async(model, messages, proxy = proxy, **kwargs))
        yield await cls.create_async(model, messages, proxy = proxy, **kwargs)

    @staticmethod
    @abstractmethod
    async def create_async(
        model: str,
        proxy: str,
        messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError()


class AsyncGeneratorProvider(AsyncProvider):
    supports_stream = True

    @classmethod
    async def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True,
        proxy: str = None,
        **kwargs
    ) -> CreateResult:
        yield await cls.create_async(model, messages, proxy=proxy, **kwargs)

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> str:
        chunks = [chunk async for chunk in cls.create_async_generator(model, messages, stream=False, proxy=proxy, **kwargs)]
        if chunks:
            return "".join(chunks)
        
    @staticmethod
    @abstractmethod
    def create_async_generator(
            model: str,
            messages: list[dict[str, str]],
            proxy: str = None,
            **kwargs
        ) -> AsyncGenerator:
        raise NotImplementedError()


async def run_generator(generator: AsyncGenerator[Union[Any, str], Any]):
    loop = asyncio.new_event_loop()
    gen  = generator.__aiter__()

    while True:
        try:
            # yield loop.run_until_complete(gen.__anext__())
            yield await gen.__anext__()

        except StopAsyncIteration:
            break
