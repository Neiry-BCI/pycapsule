import numpy as np
# cmp 
# before - only pick valid channels that were at the start of the session
# after - only pick valid channels that were at the end of the session
# both - pick only those that were valid at the start and the end
# bothInc - pick any that were valid before and after (inclusive)
def pick_channels(resistances, kOhmsThresh=300, cmp='both'):    
    def pick_channels_at(name):
        picked_channels = set()

        for chName, resistanceValues in resistances[name].items():
            if np.average(resistanceValues) < kOhmsThresh * 1000:
                picked_channels.add(chName)

        return picked_channels

    beforeSession = pick_channels_at("resBeforeSession")
    afterSession = pick_channels_at("resAfterSession")

    if cmp == 'before':
        return beforeSession
    elif cmp == 'after':
        return afterSession
    elif cmp == 'bothInc':
        return beforeSession.union(afterSession)

    return beforeSession.intersection(afterSession)