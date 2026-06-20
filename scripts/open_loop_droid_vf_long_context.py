#!/usr/bin/env python3
"""Offline open-loop evaluation for DreamZero on DROID value-function data.

This variant uses a longer video context window so the observation layout is
closer to the training-time conditioning regime than the single-frame eval.
"""

import torch._dynamo

torch._dynamo.config.disable = True

from open_loop_droid_vf_common import evaluate, make_arg_parser


def main():
    parser = make_arg_parser(default_output_dir="results_droid_vf_long_context", include_context_length=True)
    args = parser.parse_args()
    # Long-context eval: each observation is (T, H, W, 3), with T defaulting to 33.
    evaluate(args, context_length=args.context_length)


if __name__ == "__main__":
    main()
