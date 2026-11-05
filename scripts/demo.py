"""Interactive demo: label comments as a human reviewer and watch the system judge you.

Usage:  python scripts/demo.py [--rounds 3] [--batch 5] [--cap 1.2]
        python scripts/demo.py --auto honest          # scripted reviewer, no typing
        python scripts/demo.py --auto gender-blind --rounds 12   # watch it get flagged
        python scripts/demo.py --show-gold                 # presentation mode: reveal hidden gold items
"""

import argparse

from trai_al.demo import run_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--cap", type=float, default=None, help="exposure cap (e.g. 1.2)")
    parser.add_argument("--auto", choices=["honest", "gender-blind"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-gold", action="store_true",
                        help="presentation mode: mark hidden gold items and show their reference answer")
    args = parser.parse_args()
    run_demo(rounds=args.rounds, batch_size=args.batch, auto=args.auto, seed=args.seed, exposure_cap=args.cap,
             show_gold=args.show_gold)


if __name__ == "__main__":
    main()
