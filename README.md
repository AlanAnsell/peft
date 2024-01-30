<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->



<h1 align="center"> <p>🤗 PEFT-SFT </p></h1>
<h3 align="center">
    <p>Sparse Fine-Tuning for Large Language Models</p>
</h3>

This is a fork of 🤗 PEFT implementing efficient sparse fine-tuning (SFT) as described in the paper [Scaling Sparse Fine-Tuning to Large Language Models](https://arxiv.org/abs/2401.16405). The scripts for the instruction-tuning experiments from the paper can be found at [https://github.com/ducdauge/sft-llm](https://github.com/ducdauge/sft-llm). You can also find a simple QA example with 🤗 Trainer [here](examples/question_answering).


You can install this package as follows:
```bash
git clone https://github.com/AlanAnsell/peft.git
cd peft
python setup.py develop # or "pip install .", but this way is recommended
```

or use 
```
pip install git+https://github.com/AlanAnsell/peft.git
```

You can prepare a model for SFT as follows:

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, SftConfig, TaskType
model_name_or_path = "meta-llama/Llama-2-7b-hf"

peft_config = SftConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    density=0.01,
    selection_algorithm="rigl", # or "sm3" for moment approximation SFT
    target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
```

Because SFT updates the set of trainable parameters during training, some code needs to be added to the training loop. If you are using 🤗 Trainer, create an `SftTrainer` subclass and then construct it normally with your `peft_config` as argument like so:
```python
from peft import SftTrainer

...

trainer_cls = SftTrainer(MyTrainer) # MyTrainer = Trainer or any subclass thereof
trainer = trainer_cls(
    model=model,
    args=training_args,
    ...
    sft_config=peft_config,
)

```
You should then be able to use `trainer` as you would normally.

If you are using a custom training loop, you should use the SftAdamW/SftSM3 optimizer depending on whether you are using accumulated gradient or moment approximation SFT, and construct an `SftSelector` object:
```python
from peft import SftAdamW, SftSM3, SftSelector

...

optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in model.named_parameters()
            if p.requires_grad
        ],
        "weight_decay": weight_decay,
    },
]

if peft_config.selection_algorithm == "sm3":
    deltas = {
        delta.values: delta
        for _1, _2, delta in model.active_deltas()
    }
    optimizer = SftSM3(
        optimizer_grouped_parameters,
        deltas,
        lr=learning_rate,
    )
else:
    optimizer = SftAdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        momentum_dtype=torch.float32,
    )

...

selector = SftSelector(
    model,
    optimizer,
    peft_config,
    num_train_steps, # total expected duration of training in update steps
    gradient_accumulation_steps, # grad accumulation steps per update step
)
```
Then call the `selector`'s `.step()` method at the end of each update step, e.g.
```python
for i, batch in enumerate(train_dataloader):
    ...
    loss = model(**batch)
    loss.backward()
    ...

    if (i + 1) % grad_accumulation_steps == 0:
        ...
        optimizer.step()
        optimizer.zero_grad()
        selector.step()
```

The following hyperparameters can be modified through the `SftConfig`:
* `density`/`num_tunable_weights` set the number of tunable parameters as a proportion of total model params / as an absolute number respectively. Defaults to `density=0.01`.
* `selection_algorithm`: sets the SFT selection algorithm. Supply `"rigl"` for gradient accumulation/RigL-style SFT or `"sm3"` for moment approximation SFT with the SM3 optimizer. Defaults to `"rigl"`.
* `reselection_steps`: sets the number of steps between parameter reselections. Defaults to 20. You may want to use a larger value for small batch sizes.
* `selection_accumulation_steps`: for gradient accumulation SFT, controls the number of steps over which gradients are accumulated.
* `initial_reselection_rate`: the proportion of parameters that will be reselected initially. This is reduced linearly to zero over the course of training. Defaults to 0.2.
* `target_modules`: must be supplied, controls which linear modules SFT is applied to. It is recommended to apply SFT to all linear modules, e.g. `target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]` for LLaMA.

### PEFT

For details on using PEFT please refer to the [HuggingFace documentation](https://huggingface.co/docs/peft/quicktour) or the 🤗 [PEFT repository](https://github.com/huggingface/peft/).

## Citing
If you are using our SFT implemantation, please use the following snippet to cite our work:
```bibtex
@misc{ansell2024scaling,
      title={Scaling Sparse Fine-Tuning to Large Language Models}, 
      author={Alan Ansell and Ivan Vulić and Hannah Sterz and Anna Korhonen and Edoardo M. Ponti},
      year={2024},
      eprint={2401.16405},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you want to cite 🤗 PEFT in your publication, use the following snippet:

```bibtex
@Misc{peft,
  title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
```
