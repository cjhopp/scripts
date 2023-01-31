#!/bin/sh
echo "insert latmin latmax" 
read latmin latmax
echo "insert lonmin lonmax" 
read lonmin lonmax
echo "insert spacing"
read spacing
echo "insert depth"
read depth
echo "read minimum number phases" 
read minphase
echo "read grid name"
read name


#-85.00 120.00  33.0  4.00 180.0 8
cat /dev/null > $name.conf

for lat in  `jot -p 1 - $latmin $latmax $spacing`
do
for lon in `jot  -p 1 - $lonmin $lonmax $spacing`
do
echo "$lat $lon $depth 1 10.0 $minphase" >> $name.conf
done
done
