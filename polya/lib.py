from typing import Tuple, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, \
    HumanMessage

from models import Messages, get_type


class MyTypedDict(TypedDict):
    __module__ = 'typing'
    __origin__ = TypedDict

def typeddict_from_dict(d: dict) -> type:
    new_type = type('MyTypedDict', (MyTypedDict,), d)
    return new_type

class PolyaNode():
    def _prepare_messages_for_tool_call(
        self, 
        content: str, 
        conversation: Messages, 
        context: AIMessage = None
    ) -> Messages:
        sys = SystemMessage(content)
        messages = []
        messages.append(sys)
        if context:
            messages.append(context)

        etc = [msg for msg in conversation
                        if not isinstance(msg, SystemMessage)]
        if isinstance(etc[-1], AIMessage):
            etc[-1] = HumanMessage(content=messages[-1], source="llm")

        messages += etc

        return messages
    
    def _default_step(
            self, 
            template: str,
            messages: Messages, 
            context: AnyMessage, 
            prompts: Tuple[str, str],
            review: bool, 
    ):
        __type__ = get_type(template)
        if not review:
            if context:
                messages = self._prepare_messages_for_tool_call(
                    content = prompts[0], 
                    conversation = messages,
                    context = context)
            else:
                messages = self._prepare_messages_for_tool_call(
                    content = prompts[0], 
                    conversation = messages)                
        else:
            messages = self._prepare_messages_for_tool_call(
                content = prompts[1], 
                conversation = messages,
                context = context)

        for m in messages:
            m.pretty_print()

        response = self.llm.with_structured_output(__type__).invoke(messages)
        return response
