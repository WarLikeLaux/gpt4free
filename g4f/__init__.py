from __future__ import annotations
from .          import models
from .Provider  import BaseProvider
from .typing    import Any, CreateResult, Union

logging = False

class ChatCompletion:
    @staticmethod
    async def create(
        model    : Union[models.Model, str],
        messages : list[dict[str, str]],
        proxy    : str                             = None,
        provider : Union[type[BaseProvider], None] = None,
        stream   : bool                            = False,
        auth     : Union[str, None]                = None, **kwargs: Any) -> Union[CreateResult, str]:
        
        if isinstance(model, str):
            try:
                model = models.ModelUtils.convert[model]
            except KeyError:
                raise Exception(f'The model: {model} does not exist')

        provider = model.best_provider if provider == None else provider

        if not provider.working:
            raise Exception(f'{provider.__name__} is not working')

        if provider.needs_auth and not auth:
            raise Exception(
                f'ValueError: {provider.__name__} requires authentication (use auth=\'cookie or token or jwt ...\' param)')
            
        if provider.needs_auth:
            kwargs['auth'] = auth

        if not provider.supports_stream and stream:
            raise Exception(
                f'ValueError: {provider.__name__} does not support "stream" argument')

        if logging:
            print(f'Using {provider.__name__} provider')

        result = provider.create_completion(model.name, messages, stream, proxy, **kwargs)
        result = await gather_results_from_generator(result)
        return result
        return result
        return result if stream else ''.join(result)

async def gather_results_from_generator(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return ''.join(results)
