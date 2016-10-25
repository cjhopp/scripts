#!/bin/bash

for file in /home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotnga*sora*QML.xml
do
    curl -v --data-binary @$file -u admin:admin -X POST http://sgees017.geo.vuw.ac.nz:8180/xml/seismology/event/
done;
