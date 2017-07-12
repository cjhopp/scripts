#!/bin/bash
#Loop over each file in directory of StationXML files
for name in *staxml.xml
do
    #Create output dataless filename
    output=${name:0:4}.dataless
    #Correct syntax when using IRIS station-XML converter
    java -jar ~/stationXML_converter/stationxml-converter-1.0.9.jar --seed $name \
      -o $output
done
