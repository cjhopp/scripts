start_date=2018-12-24
end_date=2018-12-26

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$start_date")
enddate=$(date -I -d "$end_date")

echo Process period $(date -d $startdate +%Y-%m-%d)' 00:00:00~'$(date -d $end_date +%Y-%m-%d)' 23:59:59'

scart --files 10000 -dsEv -t $(date -d $startdate +%Y-%m-%d)' 00:00:00~'$(date -d $end_date +%Y-%m-%d)' 23:59:59'  -n '2C,ON,SI' -c '(H|E)H(Z|N|E)' /mnt/SED-miniseed | scautopick -I - --playback --ep --inventory-db inventory.xml --config-db config.xml > picks.xml
scanloc --ep picks.xml --inventory-db inventory.xml --config-db config.xml > origins.xml
scamp --ep origins.xml --inventory-db inventory.xml --config-db config.xml -I sdsarchive:///mnt/SED-miniseed > amps.xml
scmag --ep amps.xml --inventory-db inventory.xml --config-db config.xml > mags.xml
scevent --ep mags.xml --inventory-db inventory.xml --config-db config.xml > events.xml

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
