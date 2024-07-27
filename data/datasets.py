from torchtune.data import AlpacaInstructTemplate
from torchtune.datasets._preference import PreferenceDataset
from torchtune.datasets._instruct import InstructDataset
from torchtune.modules.tokenizers import Tokenizer

def politune_right(
    tokenizer: Tokenizer,
    source: str = "scale-lab/politune-right",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
) -> InstructDataset:
    return PreferenceDataset(
        tokenizer=tokenizer,
        source=source,
        column_map={
            "instruction": "prompt",
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected"
        },
        template=AlpacaInstructTemplate,
        max_seq_len=max_seq_len,
        split="train",
    )


def politune_left(
    tokenizer: Tokenizer,
    source: str = "scale-lab/politune-left",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
) -> InstructDataset:
    return PreferenceDataset(
        tokenizer=tokenizer,
        source=source,
        column_map={
            "instruction": "prompt",
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected"
        },
        template=AlpacaInstructTemplate,
        max_seq_len=max_seq_len,
        split="train",
    )


