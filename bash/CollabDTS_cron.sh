#!/usr/bin/env bash

cd /mnt/c/Users/loaner/Google\ Drive/collabDataTransfer/rawDTSplots/Collab4100-stim

# Do each individually
declare -a StringArray=("DMU" "DML" "AMU" "AML" )

# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   echo $val
   scp $(ls -tp . | grep -v / | head -n 4 | grep "$val") chet@gmf4.lbl.gov:/var/www/sigmav/web/plots/DTS/$val.png
done