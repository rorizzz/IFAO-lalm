"""Prepare cut.conversation for QA evaluation.

Reads a Lhotse CutSet manifest where each cut has
``supervisions[0].custom.question`` and ``supervisions[0].custom.answer``,
builds a conversation with a system prompt + audio + question, and writes
the result to ``cut.custom["conversation"]``.

Usage::

    python prepare_conversation_qa.py \
        --input_manifest data/test/mmau_test_mini.jsonl.gz \
        --output_manifest data/test/mmau_test_mini_conversation.jsonl.gz \
        --tokenizer models/qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm
from transformers import AutoTokenizer

_DEFAULT_SYSTEM = (
    "You are an audio understanding assistant. "
    "Listen to the audio and answer the question concisely."
)

# single-word answer version:
# _DEFAULT_SYSTEM = (
#     "You are an audio understanding assistant. "
#     "Listen to the audio and answer the question directly. Give only the answer, no explanation."
# )


def _build_conversation(cut, system: str):
    question = cut.supervisions[0].custom["question"]
    answer = cut.supervisions[0].custom["answer"].strip()
    audio_source = cut.recording.sources[0].source

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_source},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": answer},
    ]
    return messages


def _render_conversation(
    conversation: list[dict], audio_token: str = "<|audio|>"
) -> str:
    chunks = []
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if isinstance(content, list):
            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type == "audio":
                    parts.append(audio_token)
                elif item_type == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    raise ValueError(f"Unknown content type: {item_type!r}")
            rendered_content = "".join(parts)
        else:
            rendered_content = str(content)
        chunks.append(f"<|im_start|>{role}\n{rendered_content}<|im_end|>\n")
    return "".join(chunks)


def _estimate_text_tokens(text: str, tokenizer) -> int:
    text = text.strip()
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def prepare_cut(cut, tokenizer, system: str):
    if cut.custom is None:
        cut.custom = {}
    conversation = _build_conversation(cut, system)
    rendered_conversation = _render_conversation(conversation)
    cut.custom["conversation"] = conversation
    cut.custom["rendered_conversation"] = rendered_conversation
    cut.custom["num_text_tokens"] = _estimate_text_tokens(
        rendered_conversation, tokenizer
    )
    return cut


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cut.conversation for QA evaluation."
    )
    parser.add_argument(
        "--input_manifest",
        required=True,
        help="Input CutSet manifest path (e.g. *.jsonl.gz).",
    )
    parser.add_argument(
        "--output_manifest",
        required=True,
        help="Output CutSet manifest path.",
    )
    parser.add_argument(
        "--system",
        default=_DEFAULT_SYSTEM,
        help="System prompt (default: audio understanding assistant).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path used to estimate num_text_tokens.",
    )
    args = parser.parse_args()

    out_path = Path(args.output_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    cuts = CutSet.from_file(args.input_manifest)
    prepared = []
    for cut in tqdm(cuts, desc="Preparing QA conversations"):
        prepared.append(prepare_cut(cut, tokenizer=tokenizer, system=args.system))
    prepared = CutSet.from_cuts(prepared)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_file(str(out_path))
    print(f"Saved prepared QA CutSet to: {out_path}")


if __name__ == "__main__":
    main()
