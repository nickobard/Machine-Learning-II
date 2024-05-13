#!/usr/bin/env python3
import cairosvg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_svg", type=str, help="Input SVG image.")
parser.add_argument("--output_png", type=str, help="Output PNG image.")
parser.add_argument("--dpi", type=int, help="DPI (dots per inch) of output image.", default=300)


def main(args: argparse.Namespace):
    cairosvg.svg2png(url=args.input_svg, write_to=args.output_png, dpi=args.dpi)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
