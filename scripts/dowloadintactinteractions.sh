#!/bin/bash
set -e

HOST='ftp.ebi.ac.uk'
USER='anonymous'
PASSWD='anonymous'
FILE=/pub/databases/intact/current/psi25/species/human*.xml
s3destination=$2
LDIR=$1


ftp -n -v $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
prompt noprompt
lcd $LDIR
passive
cd /pub/databases/IntAct/current/psi25/species/
mget human*.xml

END_SCRIPT

if ["$s3destination" == ""]; then
    echo "No s3destination argument passed, hence not copying to s3. If you want to copy data to s3 .."
    echo "$0 s3://mybucket/data"

else
    aws s3 copy data/* $s3destination
fi
