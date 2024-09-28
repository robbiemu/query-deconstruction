from typing import Tuple, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, \
    HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.language_models import BaseChatModel

from models import Messages, get_type


class MyTypedDict(TypedDict):
    __module__ = 'typing'
    __origin__ = TypedDict

def typeddict_from_dict(d: dict) -> type:
    new_type = type('MyTypedDict', (MyTypedDict,), d)
    return new_type

class PolyaNode():
    def __init__(self, prompt_key: str, llm: BaseChatModel):
        self.prompt_key = prompt_key
        self.llm = llm

    def _prepare_messages_for_tool_call(
        self, 
        system: str, 
        prompt: str, 
        conversation: Messages, 
        context: AIMessage = None
    ) -> Messages:
        messages = []
        if context:
            messages.append(context)

        messages += [msg for msg in conversation
                        if not isinstance(msg, SystemMessage)]
                
        chat_template = ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate.from_template("{system}")] 
                    + messages 
                    + [HumanMessagePromptTemplate.from_template("{prompt}")],
        )
        msgs = chat_template.format_messages(system=system, prompt=prompt)

        return msgs
    
    def _default_step(
            self, 
            template: str,
            messages: Messages, 
            context: AnyMessage, 
            system: str,
            prompts: Tuple[str, str],
            review: bool, 
    ):
        __type__ = get_type(template)
        if not review:
            if context:
                messages = self._prepare_messages_for_tool_call(
                    system = system,
                    prompt = prompts[0], 
                    conversation = messages,
                    context = context)
            else:
                messages = self._prepare_messages_for_tool_call(
                    system = system,
                    prompt = prompts[0], 
                    conversation = messages)                
        else:
            messages = self._prepare_messages_for_tool_call(
                system = system,
                prompt = prompts[1], 
                conversation = messages,
                context = context)

        for m in messages:
            m.pretty_print()

        response = self.llm.with_structured_output(__type__).invoke(messages)
        return response
