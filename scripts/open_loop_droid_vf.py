#!/usr/bin/env python3
"""Offline open-loop evaluation for DreamZero on DROID value-function data.

This is intentionally close to `scripts/open_loop_yam.py`, but it reads the
value-augmented DROID dataset and compares the predicted action/value channels
against the per-frame ground truth.
"""

import torch._dynamo

torch._dynamo.config.disable = True

from open_loop_droid_vf_common import evaluate, make_arg_parser


def main():
    parser = make_arg_parser(default_output_dir="results_droid_vf", include_context_length=False)
    args = parser.parse_args()
    # Single-frame eval: each observation is (1, H, W, 3).
    evaluate(args, context_length=1)


if __name__ == "__main__":
    main()
