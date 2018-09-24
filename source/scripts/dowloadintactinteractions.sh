#!/bin/bash
set -e

HOST='ftp.ebi.ac.uk'
USER='anonymous'
PASSWD='anonymous'
FTPDIR=/pub/databases/intact/current/psi25/species
s3destination=$3
filepattern=$2
LDIR=$1
ftpmode=$4

if [ "$filepattern" == "" ]; then
    filepattern="human_01*.xml"
fi


if [ "$ftpmode" == "" ]; then
    ftpmode="on"
fi

echo "Downloading files from $FTPDIR to $LDIR, matching pattern $filepattern from host $HOST"
ftp -n -v $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
lcd $LDIR
passive ${ftpmode}
prompt off
cd $FTPDIR
mget $filepattern
quit
END_SCRIPT

if [ "$s3destination" == "" ]; then
    echo "No s3destination argument passed, hence not copying to s3. If you want to copy data to s3 .."
    echo "$0 <localdir>  <filepatterntodownload> <s3://mydestinationbucket/data> <passive mode on|off>"

else
    aws s3 cp ${LDIR}/ $s3destination --recursive --include *.xml
    wdir=`pwd`
    cd ${LDIR} && find . -name "*.xml" > ${LDIR}/manifest.txt && cd $wdir
    aws s3 cp ${LDIR}/manifest.txt ${s3destination}/manifest.txt
fi
