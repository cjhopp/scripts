#!/bin/bash

#BOOTSTRAPING

mkdir -p LOCATIONS

# for file in *; do
# do the loop for 50 bootstraps
for file in {151..200..1}; do

echo $file
cd $file
#######################################################
### Create script in each folder and execute in background ###########
cat >  sample_$file.sh <<EOF
#!/bin/bash
#Copy hypoDD files
 cp -v ../hypoDD.sta .; cp -v ../hypoDD.inp .; cp -v ../event.sel .; cp -v ../Vp_model_p.dat .; cp -v ../hypoDD .; cp -v ../station.sel .;
#Run hypoDD
./hypoDD  hypoDD.inp 
#create catalog
cp hypoDD.reloc sample_$file.txt

cp -v *$file.txt ../LOCATIONS
# rm -v hypo*
# rm -v dt.ct
# rm -v dt.cc
EOF
chmod +x  sample_$file.sh
sh sample_$file.sh &
cd ../
done 


# for file in $(seq 50 $END); do
# echo $file
# done

# for file in {1..50..1}; do
# echo $file
# done