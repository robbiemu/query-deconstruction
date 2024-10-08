from typing import Optional, Tuple, TypedDict
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
        messages += [msg for msg in conversation
                        if not isinstance(msg, SystemMessage)]
        if context:
            messages.insert(-1, context)
        chat_template = ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate.from_template("{system}")] 
                    + messages 
                    + [HumanMessagePromptTemplate.from_template("{prompt}")],
        )
        msgs = chat_template.format_messages(system=system, prompt=prompt)

        return msgs
    
    def _default_step(
            self, 
            messages: Messages, 
            system: str,
            prompts: Tuple[str, str],
            template: Optional[str] = None,
            context: AnyMessage = None, 
            review: bool = False, 
    ):
        prompt = prompts[0] if not review else prompts[1]
        if context:
            conversation = self._prepare_messages_for_tool_call(
                system = system,
                prompt = prompt, 
                conversation = messages,
                context = context)
        else:
            conversation = self._prepare_messages_for_tool_call(
                system = system,
                prompt = prompt, 
                conversation = messages)                

        for m in conversation:
            m.pretty_print()

        if template:
            __type__ = get_type(template)
            response = self.llm.with_structured_output(__type__, include_raw=True)\
                .invoke(conversation)
            print(response)
            response = response["parsed"]
        else:
            response = self.llm.invoke(conversation)
        return response
