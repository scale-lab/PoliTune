# PoliTune: Analyzing the Impact of Data Selection and Fine-Tuning on Economic and Political Biases in Large Language Models

This repository provides training scripts for fine-tuning LLMs using our preference datasets as described in the [paper](https://arxiv.org/abs/2404.08699).

## Dataset

The datasets are hosted on Hugging Face Hub. There are two preference datasets:
- [Left-leaning preference dataset](https://huggingface.co/datasets/scale-lab/politune-left)
- [Right-leaning preference dataset](https://huggingface.co/datasets/scale-lab/politune-right)

## Repository Structure

- `configs/` - Contains the training recipes for LLMs.
- `data/` - Contains the dataset wrappers.
- `finetune/` - Contains the fine-tuning script.

## Dependencies

The codebase depends on [torchtune](https://pytorch.org/torchtune) and [huggingface](https://huggingface.co).

## Fine-Tuning the Model

To fine-tune the model, follow these steps:

1. Download the model weights using [torchtune](https://github.com/pytorch/torchtune)'s `tune download`.
2. Ensure the configuration file under `configs/` is correctly pointing to the downloaded model.
3. Run the fine-tuning process using `torchtune`:
   ```bash
   tune run finetune/dpo_finetune.py --config configs/<config file> checkpointer.output_dir=<path to save the fine-tuned model> output_dir=<path to save the outputs and logs> dataset._component_=<data.datasets.politune_left|data.datasets.politune_right>
   ```
   For example:
    ```bash
   tune run finetune/dpo_finetune.py --config configs/llama8b_lora_dpo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=output/ dataset._component_=data.datasets.politune_left
    ```
## Citation

If you use this codebase or the datasets in your work, please cite our paper:

```
@inproceedings{agiza2024politune,
  title={PoliTune: Analyzing the Impact of Data Selection and Fine-Tuning on Economic and Political Biases in Large Language Models},
  author={Agiza, Ahmed and Mostagir, Mohamed and Reda, Sherief},
  booktitle={Proceedings of the 2024 AAAI/ACM Conference on AI, Ethics, and Society},
  pages={},
  year={2024}
}
```

## License
MIT License. See [LICENSE](LICENSE) file
