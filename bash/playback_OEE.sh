start_date=2024-8-1
end_date=2025-2-1

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$start_date")
enddate=$(date -I -d "$end_date")

echo Process period $(date -d $startdate +%Y-%m-%d)' 00:00:00~'$(date -d $end_date +%Y-%m-%d)' 23:59:59'

scart --files 10000 -dsEv --list nslc_oee.txt /Data2/AmplifyEGS/scarchive | scautopick -v -I - --playback --ep -d localhost --log-file scautopick.log > picks.xml
## Working commands for Patua pulling configuration from database
scanloc -v --ep picks.xml -d localhost --log-file scanloc.log --cluster-search-log-file cluster.log > origins.xml
scamp -v --ep origins.xml -d localhost -I sdsarchive:///Data2/AmplifyEGS/scarchive --log-file scamp.log > amps.xml
scmag -v --ep amps.xml -d localhost --log-file scmag.log > mags.xml
scevent -v --ep mags.xml -d localhost --log-file scevent.log > events.xml
