import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from FlowAtlas.populate_graph.rule_management import DistRule
from FlowAtlas.populate_graph.terrain_graph import Element, TerrainGraph


SCENARIOS = [
    {
        'name': 'small_balanced_two_slot',
        'description': 'Two provinces with a 50/50 target. Shows the immediate switch after the first assignment.',
        'total_provinces': 2,
        'target_dist': {'forest': 0.5, 'water': 0.5},
        'sequence': ['forest', 'water'],
        'adjusting_factor': 1.0,
    },
    {
        'name': 'small_overshoot_four_slot',
        'description': 'Four provinces with repeated forest picks to show early overshoot pressure.',
        'total_provinces': 4,
        'target_dist': {'forest': 0.5, 'water': 0.5},
        'sequence': ['forest', 'forest', 'forest'],
        'adjusting_factor': 1.0,
    },
    {
        'name': 'large_on_target_then_overshoot',
        'description': 'Twenty provinces. The trace shows how being near target differs from overshooting later.',
        'total_provinces': 20,
        'target_dist': {'forest': 0.5, 'water': 0.5},
        'sequence': ['forest'] * 12,
        'adjusting_factor': 1.0,
    },
    {
        'name': 'blended_factor_partial_pull',
        'description': 'A blended adjustment factor keeps a baseline target contribution even when counts drift.',
        'total_provinces': 20,
        'target_dist': {'forest': 0.7, 'water': 0.3},
        'sequence': ['forest'] * 8 + ['water'] * 2 + ['forest'] * 4,
        'adjusting_factor': 0.5,
    },
]


def build_province_graph(total_provinces: int) -> TerrainGraph:
    graph = TerrainGraph(settings={})
    for index in range(total_provinces):
        graph.add_node(f'P{index}')
    return graph


def run_scenario(scenario: dict, live_trace: bool = False) -> tuple[DistRule, list[dict]]:
    graph = build_province_graph(scenario['total_provinces'])
    rule = DistRule(
        adjusting_province_dist=dict(scenario['target_dist']),
        adjusting_border_dist={'normal': 1.0},
        adjusting_factor=scenario['adjusting_factor'],
        flag='all',
        trace_adjustments=True,
        trace_print=live_trace,
        name=scenario['name'],
    )
    rule.setup(graph)

    for index, terrain in enumerate(scenario['sequence']):
        origin = Element.from_node(f'P{index}', graph)
        origin['terrain'] = terrain
        rule.update_statistics_for_origin(graph, origin)

    return rule, rule.get_adjusting_dist_history('province')


def format_trace_entry(entry: dict) -> str:
    header = (
        f"step={entry['step']:>2} assigned={entry['assigned_total']:>2}/{int(entry['total_elements'])} "
        f"live_dist={entry['live_dist']}"
    )
    terrain_lines = []
    for terrain, values in entry['terrains'].items():
        terrain_lines.append(
            "    "
            f"{terrain:<12} count={values['current_count']:<3} "
            f"target_count={values['target_count']:<6.2f} gap={values['gap']:<6.2f} "
            f"adjustment_part={values['adjustment_part']:<6.3f} "
            f"target_part={values['target_part']:<6.3f} "
            f"adjusted_factor={values['adjusted_factor']:<6.3f}"
        )
    return "\n".join([header, *terrain_lines])


def print_scenario_log(scenario: dict, history: list[dict]):
    print()
    print(f"=== {scenario['name']} ===")
    print(scenario['description'])
    print(
        f"target_dist={scenario['target_dist']} adjusting_factor={scenario['adjusting_factor']} "
        f"sequence={scenario['sequence']}"
    )
    for entry in history:
        print(format_trace_entry(entry))


def plot_scenario(scenario: dict, history: list[dict], output_dir: Path | None = None, show_plot: bool = True):
    if not history:
        return

    terrains = list(history[0]['terrains'].keys())
    steps = [entry['step'] for entry in history]

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    figure.suptitle(
        f"Adjusting Dist Trace: {scenario['name']}\n"
        f"target={scenario['target_dist']} factor={scenario['adjusting_factor']}"
    )

    for terrain in terrains:
        counts = [entry['terrains'][terrain]['current_count'] for entry in history]
        target_counts = [entry['terrains'][terrain]['target_count'] for entry in history]
        gaps = [entry['terrains'][terrain]['gap'] for entry in history]
        adjustment_parts = [entry['terrains'][terrain]['adjustment_part'] for entry in history]
        adjusted_factors = [entry['terrains'][terrain]['adjusted_factor'] for entry in history]
        target_components = [entry['terrains'][terrain]['target_component'] for entry in history]

        axes[0].plot(steps, counts, marker='o', label=f'{terrain} count')
        axes[0].plot(steps, target_counts, linestyle='--', label=f'{terrain} target count')

        axes[1].plot(steps, gaps, marker='o', label=f'{terrain} gap')

        axes[2].plot(steps, adjustment_parts, marker='o', label=f'{terrain} adjustment part')
        axes[2].plot(steps, adjusted_factors, linestyle='--', label=f'{terrain} adjusted factor')
        axes[2].plot(steps, target_components, linestyle=':', label=f'{terrain} target component')

    axes[0].set_ylabel('Counts')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    axes[1].set_ylabel('Gap')
    axes[1].axhline(0.0, color='black', linewidth=1.0, alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)

    axes[2].set_xlabel('Recompute Step')
    axes[2].set_ylabel('Factor Value')
    axes[2].axhline(0.0, color='black', linewidth=1.0, alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    figure.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / f"{scenario['name']}.png", dpi=150)
        with (output_dir / f"{scenario['name']}.json").open('w', encoding='utf-8') as handle:
            json.dump(history, handle, indent=2)
        with (output_dir / f"{scenario['name']}.log").open('w', encoding='utf-8') as handle:
            handle.write("\n\n".join(format_trace_entry(entry) for entry in history))

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Explore DistRule adjusting distribution behavior.')
    parser.add_argument(
        '--scenario',
        action='append',
        choices=[scenario['name'] for scenario in SCENARIOS],
        help='Run only the named scenario. Can be supplied multiple times.',
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=None,
        help='Optional directory for saved plots and trace logs.',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Generate plots without opening interactive windows.',
    )
    parser.add_argument(
        '--live-trace',
        action='store_true',
        help='Also print trace lines directly from DistRule during recompute.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_names = set(args.scenario) if args.scenario else None

    for scenario in SCENARIOS:
        if selected_names is not None and scenario['name'] not in selected_names:
            continue

        _, history = run_scenario(scenario, live_trace=args.live_trace)
        print_scenario_log(scenario, history)
        plot_scenario(
            scenario,
            history,
            output_dir=args.save_dir,
            show_plot=not args.no_show,
        )


if __name__ == '__main__':
    main()