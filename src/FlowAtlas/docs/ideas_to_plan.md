# Preprocessing before waveform collapse

1. Make a draft graph (possibly per plane)
1. Connect planes if not done before
1. Place capitals for nations and thrones and when doing so, do the following:
    1. Maximize graph distance between capitals, but also keep the average distance to nearest neighbours similar
    1. Place thrones:
        * evenly, such that all nations are "near" a similar amount
        * thrones are relatively far from capitals, and never directly neighbour capital
        * thrones are never next to each other, and somewhat evenly spaced out
        * each throne has a similar distance to 3-4 capitals
        * capital placement may be adjusted slightly to make this work, or both could be place at the same time
    1. Adjust capital neighbours to match target neighbour number setting, moving province anchor points and recomputing neighbours until fixed.
    1. Maybe adjust throne neighbours similarly?
1. Determine movement capacity/preference of each nation
1. Determine closest graph distance between each pair of "neighbour" capitals, and assign a "path" rule for each capital pair that enforces:
    * they can reach each in some way
    * neither can reach the other too easily
    * the "normal" distance between them is near some target value
1. Do the same for thrones, making sure that each throne a nation is "near" is somewhat reachable for them

# path rules
They should:
* Make sure you can reach other places without special movement
* That things that shouldn't be too close movement-wise, aren't
* Keep track of graph steps and movement points steps for each mode of movement on closest such path
* Make sure special movement has benefits, but don't dominate too hard
* Use roads to speed up "normal" path sometimes, for balance and flavour
* In disciple mode, path between teammates shoul be easier and shorter than other cap-cap paths
* Paths should be able to add/remove bridges on rivers (but keep the river) as needed for path compliance.
* Paths should be able to add/remove mountain passes (but keep the mountains) as needed for path compliance, and in extreme cases even a mountain connection completely.

* Pahting should also be used to identify if areas are "hard to reach", and limit amount of such areas, expecially large such areas. Since capitals are enforced to not be isolated from their neighbours, connection to one or especially several capitals via normal movement can be used to deem an area "not hard to reach". Small hard to reach areas can be ok.


# todo "natural" terrain rules
Remember that these rules must work on the inverse too, if A increases probability of B, B should also increases probability of A, and so on.
* Rivers should end in water provinces or maybe swamps
* Rivers should start from mountains, highlands, water provinces, or and sometimes "lush terrain" like forest and swamp
* Rivers shouldn't suddenly end for no reason
* Rivers and wastelands should less often be near each other
* Farms should more often be near rivers
* Farms should less often be near wastelands
* Mountains should often be 3+ segements long, but can have mountain passes. If needed full non-mountain connection can be created for path compliance, and the mountain chain still continnued afterwards.


# To consider
When we have global (and sometimes local) target distributions, we should consider if instead of just combining and re-weighting probabilities, the global distributiun might be calculated last, and done so to enforce it's target on the average current resulting probability across the remaining unassigned elements.