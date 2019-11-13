#!/usr/bin/python

def group(tribe):
    from eqcorrscan.core.match_filter import Party
    party = Party()
    template_groups = [[]]
    for master in tribe.templates:
        for group in template_groups:
            if master in group:
                break
        else:
            new_group = [master]
            for slave in tribe.templates:
                if master.same_processing(slave) and master != slave:
                    new_group.append(slave)
            template_groups.append(new_group)
    for group in template_groups:
        if len(group) == 0:
            template_groups.remove(group)
    return template_groups

