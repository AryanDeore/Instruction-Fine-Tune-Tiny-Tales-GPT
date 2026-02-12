# Instruction Fine-Tuning Plan

## Goal
Fine-tune the pretrained 30M and 125M param TinyStories model to follow instructions, turning it from a text-completion model into a prompt-following story generator.

## Demo UI
Two input fields + generate button:
```
Write a story about: [_______________]
With: [_______] ending
[Generate Story]
```

## Instruction Template
```
Write a story about: {summary}
With: {ending} ending

### Story:
{story}
```

At inference, the model receives everything above `### Story:` and generates the story.

---

## Steps

### [x] 1. Parse TinyStories-Instruct Dataset
- Source: `roneneldan/TinyStoriesInstruct` on HuggingFace
- Single `text` column, one line per row, examples delimited by `<|endoftext|>`
- Each example has fields in varying order: `Features:`, `Words:`, `Summary:`, `Story:`
- Parser needs to:
  - Accumulate rows until `<|endoftext|>`
  - Extract `Summary` field → becomes the topic
  - Extract `Features` field → parse ending type (`BadEnding` → "sad", else → "happy")
  - Extract `Story` field (multi-line, everything after `Story:\n`) → becomes the response
  - Skip examples missing `Summary` or `Story`

### [x] 2. Curate & Upload Dataset to HuggingFace Hub

#### Dataset Schema
Final dataset columns with explicit features:
```python
from datasets import Features, Value, ClassLabel

features = Features({
    'instruction': Value('string'),   # formatted prompt (template without story)
    'response': Value('string'),      # the story text
    'ending': ClassLabel(names=['happy', 'sad']),  # ending type tag
})
```

#### Dataset Size
- Raw source: ~21.8M train rows, ~200k test rows (rows, not examples — each example spans multiple rows)
- After parsing and grouping by `<|endoftext|>`, expect significantly fewer complete examples
- This is a **large dataset** — use `Dataset.from_generator()` for memory efficiency

#### Upload Steps
```python
from datasets import Dataset, DatasetDict

# 1. Use generator for memory efficiency (21M+ rows)
def parsed_examples_generator(split="train"):
    # streams raw dataset, accumulates rows into examples, yields parsed dicts
    ...

train_ds = Dataset.from_generator(
    parsed_examples_generator, gen_kwargs={"split": "train"}, features=features
)
test_ds = Dataset.from_generator(
    parsed_examples_generator, gen_kwargs={"split": "test"}, features=features
)

# 2. Test with subset first
small = train_ds.select(range(100))
print(small[0])

# 3. Push to hub with sharding (large dataset)
ds = DatasetDict({"train": train_ds, "test": test_ds})
ds.push_to_hub("aryandeore/tinystories-instruct-curated", max_shard_size="500MB")

# 4. Verify: load_dataset("aryandeore/tinystories-instruct-curated")
```

#### Dataset Card
Include a README.md with:
- Description: curated instruction-tuning dataset derived from TinyStoriesInstruct
- Template format used
- License info (same as original TinyStories)
- Example rows

### [x] 3. Set Up New Repo
```
monday-morning-moral-sft/
├── models/                      # copy from pretraining repo (unchanged)
│   ├── gpt2.py
│   ├── transformer.py
│   ├── multi_head_attention.py
│   ├── feed_forward.py
│   └── embeddings.py
├── data/
│   ├── prepare_dataset.py       # parse raw TinyStoriesInstruct → curated format
│   ├── instruction_dataset.py   # InstructionDataset (tokenizes at init)
│   └── collate.py               # custom_collate_fn (pad + mask)
├── utils/
│   ├── config.py                # copy from pretraining repo (model config must match)
│   └── formatting.py            # format_input() template function
├── finetune.py                  # main training script
├── generate.py                  # updated for instruction-style prompts
└── checkpoint.py                # copy from pretraining repo
```

### [x] 4. Build InstructionDataset + Custom Collate
- **InstructionDataset**: takes curated dataset, formats with template, tokenizes each example at init
- **custom_collate_fn**:
  - Pad sequences to batch max length (using EOT token 50256)
  - Create input/target pairs (shift by 1)
  - Mask padding tokens in targets with `ignore_index=-100`
  - Optionally truncate to model's context_length (512 for 30M model)

### 5. Write `finetune.py`
- Load pretrained 30M checkpoint from `checkpoints/`
- Hyperparameters:
  - Optimizer: AdamW, lr=5e-5, weight_decay=0.1
  - Batch size: ~8
  - Epochs: ~2
  - Eval every N steps
- Train/val/test split: 85/5/10
- Log train loss, val loss, perplexity
- Save fine-tuned weights

### 6. Update `generate.py` for Instruction Format
- At inference, format user input with the template (without the story part)
- Model generates tokens after `### Story:\n`
- Reuse existing temperature + top-k sampling logic

### 7. Evaluate
- Generate stories on held-out test set
- Check that generated stories match the requested ending type
- Check coherence and story quality

### 8. Build Demo UI (Future)
- Gradio interface with 2 input fields (topic, ending)
- Calls generate.py under the hood
- Deploy to HF Spaces

---

## Key Decisions
- **Template**: natural language format matching the UI, not the dataset's native field format
- **Ending mapping**: `BadEnding` → "sad", everything else → "happy"
- **Moral field**: dropped for simplicity (dataset only flags existence, not theme)
- **Model**: 30M pretrained checkpoint (context_length=512)
- **Dataset upload**: curated version goes to HuggingFace Hub for reuse

## Reference
- Raschka Chapter 7: `reference/raschka/LLMs from Scratch Chapter 7/01_main-chapter-code/`
- Key file: `gpt_instruction_finetuning.py` (standalone SFT script)
