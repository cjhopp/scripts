#!/usr/bin/python

import sys

from glob import glob

def species_mapping(species_file, data_root):
    """
    Loop Christy's files and replace species number with the correct string

    :param species_file: Path to species mapping file (space delimited)
    :param data_root: Root directory for the data files
    :return:
    """

    species_dict = {}
    with open(species_file, 'r') as in_f:
        for ln in in_f:
            line = ln.strip()
            line = line.split()
            print(line)
            species_dict[line[0]] = line[1]
    data_files = glob('{}/RAxML*.txt'.format(data_root))
    for d_file in data_files:
        with open(d_file, 'r') as in_f:
            for ln in in_f:
                new_str = ln
            for no, spec in species_dict.items():
                new_str = new_str.replace(no, spec)
        with open('{}.NEW'.format(d_file), 'w') as out_file:
            out_file.write(new_str)

if __name__ == '__main__':
    species_mapping(sys.argv[1], sys.argv[2])