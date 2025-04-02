import json
import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import squarify
from argparse import ArgumentParser
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


EXISTING_SUPER_TYPES = ["legendary", "basic", "snow", "world"]
EXISTING_CARD_TYPES = [
    "instant",
    "sorcery",
    "land",
    "artifact",
    "enchantment",
    "creature",
    "planeswalker",
    "battle",
    "kindred",
]
MTG_COLORS = ["W", "U", "B", "R", "G", "C"]
MTG_COLORS_NAMES = ["White", "Blue", "Black", "Red", "Green", "Colorless"]


def process_card_info(card: dict):
    type_line = list(map(str.lower, re.findall(r"[A-Za-z]+", card["type_line"])))
    card_type = list({t for t in type_line if t in EXISTING_CARD_TYPES})
    super_types = list({t for t in type_line if t in EXISTING_SUPER_TYPES})
    sub_types = list(set(type_line) - set(card_type) - set(super_types))

    card_info = {
        "name": card.get("name"),
        "mana_cost": card.get("mana_cost"),
        "cmc": card.get("cmc"),
        "super_types": super_types,
        "card_types": card_type,
        "sub_types": sub_types,
        "type_line": card.get("type_line"),
        "oracle_text": card.get("oracle_text"),
        "power": card.get("power"),
        "toughness": card.get("toughness"),
        "loyalty": card.get("loyalty"),
        "colors": card.get("colors"),
        "color_identity": card.get("color_identity"),
        "set": card.get("set"),
        "rarity": card.get("rarity"),
        "prices": card.get("prices"),
        "legalities": card.get("legalities"),
    }
    return card_info


def process_decklist_file(decklist_json: str):
    with open(decklist_json, "r") as f:
        decklist_json = json.load(f)
    return list(map(process_card_info, decklist_json))


def radar_factory(num_vars, frame="circle", r_size: int = 0.5):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    source: https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), r_size)
            elif frame == "polygon":
                return RegularPolygon(
                    (0.5, 0.5), num_vars, radius=r_size, edgecolor="k"
                )
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def spider(
    store_path: str, data: list, labels: list, colors: list, max_value: int = 100
):
    theta = radar_factory(9, frame="polygon")

    fig, axs = plt.subplots(
        figsize=(18, 9), nrows=1, ncols=2, subplot_kw=dict(projection="radar")
    )
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    # Plot the four cases from the example data on separate Axes
    for ax, (spoke_labels, title, case_data) in zip(axs.flat, data):
        ax.set_rgrids(list(range(0, max_value, 5)))
        ax.set_title(
            title,
            weight="bold",
            size="medium",
            position=(0.5, 1.1),
            horizontalalignment="center",
            verticalalignment="center",
        )
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")
        ax.set_varlabels(spoke_labels)

    axs[0].legend(labels, loc=(0.9, 0.95), labelspacing=0.1, fontsize="medium")

    plt.savefig(store_path)


def spider_data(decklist: list):
    labels = MTG_COLORS_NAMES.copy()
    colors = ["y", "b", "m", "r", "g", "0.7"]

    color_data = [
        [0 for _ in range(len(EXISTING_CARD_TYPES))],  # White
        [0 for _ in range(len(EXISTING_CARD_TYPES))],  # Blue
        [0 for _ in range(len(EXISTING_CARD_TYPES))],  # Black
        [0 for _ in range(len(EXISTING_CARD_TYPES))],  # Red
        [0 for _ in range(len(EXISTING_CARD_TYPES))],  # Green
        [0 for _ in range(len(EXISTING_CARD_TYPES))],  # Colorless
    ]
    cmc_data = [
        [0 for _ in range(9)],  # White
        [0 for _ in range(9)],  # Blue
        [0 for _ in range(9)],  # Black
        [0 for _ in range(9)],  # Red
        [0 for _ in range(9)],  # Green
        [0 for _ in range(9)],  # Colorless
    ]
    for card_info in decklist:
        cmc = int(card_info["cmc"]) if card_info["cmc"] < 8 else 8
        for t in card_info["card_types"]:
            type_index = EXISTING_CARD_TYPES.index(t)
            if not card_info["colors"]:
                color_data[-1][type_index] += 1
                if t != "land":
                    cmc_data[-1][cmc] += 1
                continue
            for color in card_info["colors"]:
                color_data[MTG_COLORS.index(color)][type_index] += 1
                if t != "land":
                    cmc_data[MTG_COLORS.index(color)][cmc] += 1
    data = [
        (EXISTING_CARD_TYPES.copy(), "Types by color", color_data),
        (
            ["0", "1", "2", "3", "4", "5", "6", "7", "8+"],
            "Mana Value by color",
            cmc_data,
        ),
    ]
    return data, labels, colors


def beeswarm_data(decklist: list):
    cmc_color_counts = defaultdict(list)
    color_assignation = {
        "aggregated": "black",
        "C": "0.7",
        "W": "y",
        "U": "b",
        "B": "m",
        "R": "r",
        "G": "g",
    }
    for card_info in decklist:
        # we add a bit of artificial jitter, so it gets better represented in the beeswarm plot
        cmc = card_info["cmc"] + random.uniform(-0.35, 0.35)
        if cmc is None:
            cmc = 0
        cmc_color_counts["aggregated"].append(cmc)
        if not card_info["colors"]:
            cmc_color_counts["C"].append(cmc)
            continue

        for color in card_info["colors"]:
            cmc_color_counts[color].append(cmc)
    for k, v in cmc_color_counts.items():
        cmc_color_counts[k] = sorted(v)
    cmc_color_counts["colors"] = [color_assignation[k] for k in cmc_color_counts.keys()]
    return cmc_color_counts


def seaborn_beeswarm(labeled_data: dict, store_path: str):
    import seaborn as sns

    colors = labeled_data.pop("colors")
    sns.swarmplot(data=labeled_data, size=3, palette=colors)
    plt.savefig(store_path)


def treemap_data(decklist: list):
    card_types = defaultdict(list)
    for card_info in decklist:
        for t in card_info["card_types"]:
            card_types[t].append(card_info["name"])
    return card_types


def treemap(labeled_data: dict, store_path: str):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = [
        "#855C75FF",
        "#D9AF6BFF",
        "#AF6458FF",
        "#736F4CFF",
        "#526A83FF",
        "#625377FF",
        "#68855CFF",
        "#9C9C5EFF",
        "#A06177FF",
        "#8C785DFF",
        "#467378FF",
        "#7C7C7CFF",
    ]
    ax.set_axis_off()
    squarify.plot(
        sizes=[len(v) for v in labeled_data.values()],
        label=[f"{k}: {len(v)}".title() for k, v in labeled_data.items()],
        color=[colors[i] for i in range(len(labeled_data))],
        text_kwargs={
            "color": "black",
            "backgroundcolor": "0.95",
            "fontsize": 9,
            "fontweight": "bold",
            "wrap": True,
        },
        pad=True,
        ax=ax,
    )

    plt.savefig(store_path)


if __name__ == "__main__":
    agp = ArgumentParser()
    agp.add_argument(
        "--graph",
        "-g",
        type=str,
        help="Graph type to generate",
        choices=["spider", "beeswarm", "treemap"],
        default="spider",
    )
    agp.add_argument("--deck", "-d", type=str, help="Decklist JSON file", required=True)

    args = agp.parse_args()
    graph = args.graph
    decklist_json = args.deck

    dk = process_decklist_file(decklist_json)
    output_path = (
        "/".join(decklist_json.split("/")[0:-1])
        + "/outputs/"
        + decklist_json.split("/")[-1].replace(".json", "")
    )
    if graph == "spider":
        spider(output_path + "_spider_plot.png", *spider_data(dk))
    elif graph == "beeswarm":
        seaborn_beeswarm(beeswarm_data(dk), output_path + "_beeswarm_seaborn.png")
    elif graph == "treemap":
        treemap(treemap_data(dk), output_path + "_treemap.png")
