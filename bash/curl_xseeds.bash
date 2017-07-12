#!/bin/bash
#Must convert to XSEED prior to POST to SeisHub
#Correct usage of POST when adding resources to SeisHub database
for file in *xseed.xml
do
    curl -v --data-binary @$file -u admin:admin\
          -X POST http://localhost:8080/xml/seismology/station/${file:0:4}
done
