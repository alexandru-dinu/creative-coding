import random
from argparse import Namespace

import numpy as np
from PIL import Image, ImageDraw

SEED = 4242
RNG = np.random.default_rng(SEED)
random.seed(SEED)


def generate_single(args):
    l = max(0, RNG.normal(loc=args.lower_loc, scale=args.lower_scale))
    u = min(255, RNG.normal(loc=args.upper_loc, scale=args.upper_scale))

    r = lambda: RNG.integers(low=l, high=u)
    rc = lambda: (r(), r(), r())
    colors = [rc() for _ in range(args.num_colored)] + [args.bg_color] * args.num_empty

    img = Image.new("RGB", 2 * (args.sprite_canvas_size,))
    draw = ImageDraw.Draw(img)

    for y in range(args.sprite_unit_size):
        for xl in range(args.sprite_unit_size // 2):
            xr = args.sprite_unit_size - xl - 1

            c = random.choice(colors)

            pos = [
                (xl * args.sprite_unit, y * args.sprite_unit),
                ((xl + 1) * args.sprite_unit, (y + 1) * args.sprite_unit),
            ]
            draw.rectangle(pos, c)

            pos = [
                (xr * args.sprite_unit, y * args.sprite_unit),
                ((xr + 1) * args.sprite_unit, (y + 1) * args.sprite_unit),
            ]
            draw.rectangle(pos, c)

    return img


def generate(args):
    sz = args.sprite_canvas_size + 4 * args.margin
    w = sz * args.sprites_per_line
    h = sz * args.num_lines

    wall = Image.new("RGB", (w, h))

    for i in range(args.num_lines):
        for j in range(args.sprites_per_line):

            sprite = generate_single(args)

            img = Image.new("RGB", 2 * (sz,), color=args.bg_color)
            img.paste(sprite, (args.margin, args.margin))

            wall.paste(img, (j * sz, i * sz))

    return wall


def main():
    args = Namespace()
    # args.bg_color = (0xff,) * 3
    args.bg_color = (0x1E,) * 3

    args.sprite_canvas_size = 2**7
    args.sprite_unit_size = 2**3
    args.sprite_unit = args.sprite_canvas_size // args.sprite_unit_size

    args.num_colored = 6
    args.num_empty = 4

    args.lower_loc = 50
    args.lower_scale = 5
    args.upper_loc = 200
    args.upper_scale = 10

    # control final canvas
    args.sprites_per_line = 8
    args.num_lines = 16
    args.margin = 20

    # s = generate_single(args)
    # s.show()

    wall = generate(args)
    wall.show()
    # wall.save(
    #     f"./out/space_invaders_light_canvas_{args.sprite_canvas_size}_unit_{args.sprite_unit_size}.png",
    #     "png",
    # )


if __name__ == "__main__":
    main()
