"""
src.training -- optimization loops for MiniGPT (Step 12) and MiniBERT
(Step 13).

Design note on why this __init__.py does NOT flatten every function into
the package namespace (unlike src/layers and src/model):

train_gpt.py and train_bert.py both define functions with the *same
names* -- build_dataset(...) and train(...) -- but with different
signatures and different return types (GPT: token-window batches / a
single scalar loss; BERT: MLM+NSP pairs / an (mlm_loss, nsp_loss) tuple).
`from src.training import train` would silently shadow one of them, which
is the kind of bug that's invisible until the wrong model trains on the
wrong loop. Both scripts are also meant to be run directly:

    python -m src.training.train_gpt  --corpus path/to/text.txt
    python -m src.training.train_bert

So this package exposes the *submodules*, not their contents. Call
train_gpt.train(...) / train_bert.train(...) explicitly, or import the
submodule you need:

    from src.training import train_gpt, train_bert
    train_gpt.set_seed(42)
    losses = train_gpt.train(model, tokenizer, config)

set_seed and make_cosine_warmup_scheduler are unique to train_gpt.py
today (train_bert.py does not yet define its own seeding/schedule
helpers), so they are re-exported directly for convenience -- but note
this means BERT training currently relies on whatever seeding the caller
does manually. Worth unifying into a shared src/training/utils.py if the
two scripts' seeding/scheduling logic is meant to stay identical.
"""

from __future__ import annotations

from src.training import train_gpt
from src.training import train_bert
from src.training.train_gpt import set_seed, make_cosine_warmup_scheduler

__all__ = [
    "train_gpt",
    "train_bert",
    "set_seed",
    "make_cosine_warmup_scheduler",
]