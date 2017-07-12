#!/bin/bash

for file in *.xml
do
    curl -v --data-binary @$file -u admin:admin -X POST http://sgees017.geo.vuw.ac.nz:8180/xml/seismology/event/${file:0:14}
done;
