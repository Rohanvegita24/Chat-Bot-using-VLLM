from vllm import LLM, SamplingParams
import gradio as gr
import torch



llm = LLM(
    model_path="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    quantization="awq",
    dtype="half",
    max_model_len=1000,
    enforce_eager=True
)
sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=150)


def generate_text(prompt , num_words=100):

    system_message = "Answer this question correctly: "
    prompt_template = f"{system_message}{prompt}\n"

    prompt_with_template = prompt_template.format(prompt=prompt)
    sampling_params.max_tokens = num_words
    outputs = llm.generate([prompt_with_template], sampling_params)
    generated_text = outputs[0].outputs[0].text if outputs else ""
    return generated_text

iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="LLM Text Generation",
    description="Generate text based on the provided prompt.",
).launch()