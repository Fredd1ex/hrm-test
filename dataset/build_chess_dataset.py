"""Convert Gigafish FEN evaluations into the repository's puzzle dataset format.

Each FEN is encoded as a fixed, semantic token sequence.  Only the final
<EVAL> position is labelled, so the unchanged token-classification loss and
adaptive halting objective treat an exact evaluation-bin prediction as solved.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import PuzzleDatasetMetadata


PAD_ID = 0
EMPTY_ID = 1
PIECE_IDS = {piece: index for index, piece in enumerate("PNBRQKpnbrqk", start=2)}

SIDE_TO_MOVE_IDS = {"w": 14, "b": 15}
CASTLING_IDS = ((16, 17), (18, 19), (20, 21), (22, 23))  # Present, absent for KQkq.
EN_PASSANT_NONE_ID = 24
EN_PASSANT_FILE_IDS = {file_name: 25 + index for index, file_name in enumerate("abcdefgh")}
HALFMOVE_START_ID = 33
EVAL_TOKEN_ID = 134
EVAL_CLASS_START_ID = 135

EVAL_RANGE_CP = 1000
EVAL_BIN_WIDTH_CP = 25
NUM_EVAL_CLASSES = EVAL_RANGE_CP * 2 // EVAL_BIN_WIDTH_CP + 1
VOCAB_SIZE = EVAL_CLASS_START_ID + NUM_EVAL_CLASSES
SEQ_LEN = 72


def parse_fen(fen: str) -> list[int]:
    board, side_to_move, castling, en_passant, halfmove_clock, _fullmove = fen.split()

    board_tokens = []
    for rank in board.split("/"):
        rank_tokens = []
        for char in rank:
            if char.isdigit():
                rank_tokens.extend([EMPTY_ID] * int(char))
            else:
                rank_tokens.append(PIECE_IDS[char])
        if len(rank_tokens) != 8:
            raise ValueError(f"Invalid rank in FEN: {fen}")
        board_tokens.extend(rank_tokens)

    if len(board_tokens) != 64 or side_to_move not in SIDE_TO_MOVE_IDS:
        raise ValueError(f"Invalid FEN: {fen}")
    if en_passant != "-" and (len(en_passant) != 2 or en_passant[0] not in EN_PASSANT_FILE_IDS):
        raise ValueError(f"Invalid en-passant square in FEN: {fen}")

    castling_tokens = [present_id if right in castling else absent_id for right, (present_id, absent_id) in zip("KQkq", CASTLING_IDS)]
    en_passant_token = EN_PASSANT_NONE_ID if en_passant == "-" else EN_PASSANT_FILE_IDS[en_passant[0]]
    halfmove_token = HALFMOVE_START_ID + min(max(int(halfmove_clock), 0), 100)

    tokens = board_tokens + [SIDE_TO_MOVE_IDS[side_to_move]] + castling_tokens + [en_passant_token, halfmove_token, EVAL_TOKEN_ID]
    assert len(tokens) == SEQ_LEN
    return tokens


def evaluation_label(value: str, side_to_move: str, perspective: str) -> int:
    # Gigafish uses #N / #-N for forced mates.  Preserve the winning side and
    # map all mate distances to the already-reserved clipped extreme bin.
    if value.startswith("#"):
        mate_distance = int(value[1:])
        eval_pawns = math.copysign(EVAL_RANGE_CP / 100, mate_distance)
    else:
        eval_pawns = float(value)

    if not math.isfinite(eval_pawns):
        raise ValueError(f"Evaluation is not finite: {eval_pawns}")

    # Gigafish labels are assumed to be White-relative by default.  The option
    # allows a side-to-move target if that convention produces better results.
    if perspective == "side-to-move" and side_to_move == "b":
        eval_pawns = -eval_pawns

    eval_cp = min(max(round(eval_pawns * 100), -EVAL_RANGE_CP), EVAL_RANGE_CP)
    bin_index = round((eval_cp + EVAL_RANGE_CP) / EVAL_BIN_WIDTH_CP)
    return EVAL_CLASS_START_ID + int(bin_index)


def count_rows(csv_path: Path, max_samples: int | None) -> int:
    with csv_path.open(newline="") as csv_file:
        total = sum(1 for _ in csv.DictReader(csv_file))
    return min(total, max_samples) if max_samples is not None else total


def convert_split(csv_path: Path, output_dir: Path, max_samples: int | None, perspective: str) -> None:
    sample_count = count_rows(csv_path, max_samples)
    inputs = np.empty((sample_count, SEQ_LEN), dtype=np.int16)
    labels = np.full((sample_count, SEQ_LEN), PAD_ID, dtype=np.int16)

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(tqdm(reader, total=sample_count, desc=csv_path.stem)):
            if index == sample_count:
                break
            fen = row["fen"]
            inputs[index] = parse_fen(fen)
            labels[index, -1] = evaluation_label(row["eval"], fen.split()[1], perspective)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = PuzzleDatasetMetadata(
        pad_id=PAD_ID,
        ignore_label_id=PAD_ID,
        blank_identifier_id=PAD_ID,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        num_puzzle_identifiers=1,
        total_groups=sample_count,
        mean_puzzle_examples=1,
        sets=["all"],
    )
    with (output_dir / "dataset.json").open("w") as metadata_file:
        json.dump(metadata.model_dump(), metadata_file)

    np.save(output_dir / "all__inputs.npy", inputs)
    np.save(output_dir / "all__labels.npy", labels)
    np.save(output_dir / "all__puzzle_identifiers.npy", np.zeros(sample_count, dtype=np.int32))
    indices = np.arange(sample_count + 1, dtype=np.int32)
    np.save(output_dir / "all__puzzle_indices.npy", indices)
    np.save(output_dir / "all__group_indices.npy", indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("chess-dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/chess-eval"))
    parser.add_argument("--max-train-samples", type=int, help="Limit training rows for a smoke test.")
    parser.add_argument("--max-test-samples", type=int, help="Limit test rows for a smoke test.")
    parser.add_argument("--perspective", choices=("white", "side-to-move"), default="white")
    args = parser.parse_args()

    convert_split(args.input_dir / "gigafish_train_1m.csv", args.output_dir / "train", args.max_train_samples, args.perspective)
    convert_split(args.input_dir / "gigafish_test_200k.csv", args.output_dir / "test", args.max_test_samples, args.perspective)


if __name__ == "__main__":
    main()
