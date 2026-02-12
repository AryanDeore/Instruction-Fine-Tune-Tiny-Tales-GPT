"""Gradio web UI for story generation using fine-tuned GPT-2."""

import gradio as gr
import torch

from checkpoint import load_model
from generate import generate_story

# Default checkpoint path (30M SFT model)
CHECKPOINT_PATH = "checkpoints/sft_30M_model/finetune_epoch_5.pt"

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Load model once at startup
print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...")
model = load_model(CHECKPOINT_PATH, DEVICE)
print("Model loaded! Ready to generate stories.\n")


def generate(topic, ending, temperature):
    """Generate a story based on user inputs."""
    if not topic.strip():
        return "Please enter a topic for the story."

    story = generate_story(
        model=model,
        topic=topic.strip(),
        ending=ending.lower(),
        max_new_tokens=192,
        temperature=temperature,
        top_k=50,
        device=DEVICE,
    )

    return story


with gr.Blocks(title="Monday Morning Moral") as demo:
    gr.Markdown(
        "# Monday Morning Moral\nGenerate short stories with happy or sad endings using a fine-tuned GPT-2 model."
    )

    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(
                label="Generate a short story about:",
                placeholder="a brave knight on an adventure",
                lines=2,
            )
            with gr.Row():
                ending = gr.Radio(
                    choices=["Happy", "Sad"],
                    label="With ending:",
                    value="Happy",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.4,
                    value=0.7,
                    step=0.1,
                    label="Temperature (creativity)",
                    info="Low = predictable, High = creative",
                )
            submit_btn = gr.Button("Generate Story", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Generated Story", lines=12)

    submit_btn.click(fn=generate, inputs=[topic, ending, temperature], outputs=output)
    topic.submit(fn=generate, inputs=[topic, ending, temperature], outputs=output)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
