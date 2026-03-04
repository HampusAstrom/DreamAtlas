settings = {
    # this test config includes all normal single terrain types on the surface layer
    # but no cave terrain or special stuff we might want to add later
    # and it assume no combied terrains
    'base_global_target_dist': {
        # missing for later, all cave combinations (inc cave wall),
        # montains and freshwater (from border terrains, and maybe lake neighbour?)
        # small/large province
        # no_start, good_start, bad_start, good_throne_location, bad_throne_location
        # warmer, colder
        # rare terrain masks (one per magic path)
        # also check for others seen in dominions_data.py:
        # unknown, invisible, vast, infernal waste, void, has gate, flooded,
        # attackers rout once, Cave wall effect/draw as cave, Draw as UW, and some ???
        'province_terrains': {
            'plains': 0.2,
            'highlands': 0.2,
            'swamp': 0.1,
            'waste': 0.2,
            'forest': 0.2,
            'farm': 0.1,
            'sea': 0.1,
            'gorge': 0.1, # a combination terrain in d6m map files, but we list it separately
            'kelp_forest': 0.1, # a combination terrain in d6m map files, but we list it separately
            'deep_sea': 0.1, # a combination terrain in d6m map files, but we list it separately
        },
        'border_terrains': {
            'normal': 0.7,
            'mountain_pass': 0.05,
            'river': 0.05,
            # 'impassable': 0.0,
            'road': 0.02,
            'river_with_bridge': 0.02,
            'impassable_mountain_pass': 0.05,
        }
    }
}