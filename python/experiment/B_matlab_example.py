#!/usr/bin/python

bin_values = [[], [], []]
bins = [(1, 3), (3, 5), (5, 7)]

things = [[intens1, wav1, time1], [intens2, wav2, time2],
          [intens3, wav3, time3], [intens4, wav4, time4]]

# This is kinda how matlab will do it
# Not sure what the equivalents of range() and len() are in MATLAB
for i in range(len(things)):
    for j in range(len(bins)):
        if bins[j][1] <= things[i][2] < bins[j][2]:
            bin_values[j].append(things[i])

# Then go through bin_values and calculate the averages
for b in range(len(bin_values)):
    # Get all the intensities and then average them...