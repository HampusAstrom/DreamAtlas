from copy import deepcopy
from typing import Optional

from .rule_management import BanRule, RuleManager
from .wave_function_collapse import make_wfc_settings_from_global_dist


DEFAULT_GLOBAL_TARGET_DIST = {
    'province_terrains': {
        'plains': 0.6,
        'highlands': 0.2,
        'swamp': 0.1,
        'waste': 0.1,
        'forest': 0.25,
        'farm': 0.15,
        'sea': 0.15,
        'gorge': 0.02,
        'kelp_forest': 0.05,
        'deep_sea': 0.03,
    },
    'border_terrains': {
        'normal': 0.7,
        'mountain_pass': 0.05,
        'river': 0.05,
        'impassable': 0.0,
        'road': 0.02,
        'river_with_bridge': 0.02,
        'impassable_mountain_pass': 0.05,
    },
}


def make_default_rule_managers(base_global_target_dist: dict) -> list[RuleManager]:
    sea_types = {'sea', 'deep_sea', 'kelp_forest', 'gorge'}
    allowed_borders = {'normal', 'impassable'}
    forbidden_borders = set(base_global_target_dist['border_terrains'].keys()) - allowed_borders

    sea_border_ban = BanRule(
        set1=sea_types,
        set2=forbidden_borders,
        range=1.0,
        range_check='leq',
        evaluation=lambda neighbors: len(neighbors) > 0,
        name='sea_borders_ban',
    )

    return [
        RuleManager(
            name='sea_borders',
            rules=[sea_border_ban],
        )
    ]


def make_default_wfc_settings(global_dist_dynamic_adjustment_schedule: Optional[dict] = None) -> dict:
    settings = {
        'base_global_target_dist': deepcopy(DEFAULT_GLOBAL_TARGET_DIST),
        'rule_managers': make_default_rule_managers(DEFAULT_GLOBAL_TARGET_DIST),
    }
    if global_dist_dynamic_adjustment_schedule is not None:
        settings['global_dist_dynamic_adjustment_schedule'] = deepcopy(global_dist_dynamic_adjustment_schedule)
    return make_wfc_settings_from_global_dist(settings)