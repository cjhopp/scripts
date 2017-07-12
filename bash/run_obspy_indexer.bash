#!/bin/bash

DB=postgresql://cjhopp:*blackmore89@localhost:5432/RT_NM_db
DATA=/home/chet/data/test_mseed
LOG=/path/to/indexer.log

#Run only once and remove duplicates:
obspy-indexer -v -i0.0 --run_once --check_duplicates -n1 -u$DB -d$DATA
