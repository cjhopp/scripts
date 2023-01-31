start_date=2021-6-1
end_date=2022-12-31

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$start_date")
enddate=$(date -I -d "$end_date")

echo Process period $(date -d $startdate +%Y-%m-%d)' 00:00:00~'$(date -d $end_date +%Y-%m-%d)' 23:59:59'

scart --files 10000 -dsEv --list nslc_patua.txt /Data2/AmplifyEGS/scarchive | scautopick -v -I - --playback --ep -d localhost --log-file scautopick.log > picks.xml
#scart --files 10000 -dsEv --list nslc_patua.txt /Data2/AmplifyEGS/scarchive | scautopick00 -v -I - --playback --ep -d localhost --log-file scautopick00.log > picks00.xml
#scxmlmerge picks.xml picks00.xml > picks_all.xml
scanloc -v --ep picks.xml -d localhost --log-file scanloc.log --cluster-search-log-file cluster.log > origins.xml
scamp -v --ep origins.xml -d localhost -I sdsarchive:///Data2/AmplifyEGS/scarchive --log-file scamp.log > amps.xml
scmag -v --ep amps.xml -d localhost --log-file scmag.log > mags.xml
scevent -v --ep mags.xml -d localhost --log-file scevent.log > events.xml

# Make myCatalog.csv to specify events to relocate
echo "seiscompId" > myCatalog.csv
grep '<origin publicID=' events.xml | cut -d\" -f2 >> myCatalog.csv
scrtdd --reloc-catalog myCatalog.csv --ep events.xml --profile patua -d localhost --verbosity=3 --console=1 \
       --log-file scrtdd.log --xmlout > events-relocated-dd.xml
# Final call to scevent to create full sc3ml with Event objects
scevent --ep events-relocated-dd.xml > scrtdd-relocations.xml

#scart --files 10000 -dsEv -t $(date -d $startdate +%Y-%m-%d)' 00:00:00~'$(date -d $end_date +%Y-%m-%d)' 23:59:59'  -n '2C,ON,SI' -c '(H|E)H(Z|N|E)' /mnt/SED-miniseed > data.mseed
#scautopick -I data.mseed --playback --ep --inventory-db inventory.xml --config-db config.xml --debug > picks.xml
#scanloc --ep picks.xml --inventory-db inventory.xml --config-db config.xml > origins.xml
#scamp --ep origins.xml --inventory-db inventory.xml --config-db config.xml -I data.mseed > amps.xml
#scmag --ep amps.xml --inventory-db inventory.xml --config-db config.xml > mags.xml
#scevent --ep mags.xml --inventory-db inventory.xml --config-db config.xml > events.xml

## Using reloc

#screloc --ep origins.xml -d localhost/seiscomp > origins-relocated.xml
#scevent --ep origins-relocated.xml -d localhost/seiscomp > events-relocated.xml
#remove NLL header manually in events-relocated.xml

#then push to database with scdb or scdispatch / or use scolv offline for events.xml, and scrttv for picks.xml

#--debug is a shortcut for: --console=1 --verbosity=1-4
