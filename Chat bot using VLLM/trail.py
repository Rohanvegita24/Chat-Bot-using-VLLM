from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
import gradio as gr


lama_llm =Llama(
    model_path="/home/team2/Rohan/llama-2-7b-chat.Q6_K.gguf",n_gpu_layers=-1,n_threads=9,n_ctx=3584, n_batch=521,verbose=True,chat_format="llama-2" )

llama_cpp_agent = LlamaCppAgent(lama_llm, debug_output=False,
                              system_prompt="You are llama-2, an AI assistant , answer the question simple and short .", predefined_messages_formatter_type=MessagesFormatterType.CHATML)

def  chatwithbot(prompt):
    
    outputs =llama_cpp_agent.get_chat_response(prompt,temperature=0.9,print_output=False,add_message_to_chat_history=False,add_response_to_chat_history=False)
    
   
    return outputs
 
iface = gr.Interface(


    fn=chatwithbot,
    inputs="text",
    outputs="text",
    title="llama-2 chatbot",
)

iface.launch()
