#!/bin/bash
HOST='ftp.ebi.ac.uk'
USER='anonymous'
PASSWD='anonymous'
FILE=/pub/databases/intact/current/psi25/species/human*.xml
s3destination=$1


mkdir data
LDIR=data
ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
prompt noprompt
lcd $LDIR
cd /pub/databases/intact/current/psi25/species
mget human*.xml
quit
END_SCRIPT

if ["$s3destination"!=""];
then
    aws s3 copy data/* $s3destination
else
    echo "No s3destination argument passed, hence not copying to s3. If you want to copy data to s3 .."
    echo "$0 s3://mybucket/data"
fi
