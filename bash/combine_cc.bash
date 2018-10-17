#!/usr/bin/#!/usr/bin/env bash
dirs="$1/*"
no=0
for d in $dirs
do
  no=$((no+1))
  cat_file="$1/cc_cat_prt$no"
  cd $d
  echo $d
  pwd
  echo $cat_file
  ls | xargs -n 10000 -P 8 -I % sh -c 'cat %; echo "";' >> $cat_file
done
