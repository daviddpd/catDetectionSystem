#!/bin/bash

set -x 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case $1 in
    cat)
        FILE="${SCRIPT_DIR}/kodi-cat.json"
        MSG="cat Detected ! ($CDS_CONFIDENCE)"
    ;;
    raccoon)
    	FILE="${SCRIPT_DIR}/kodi-raccoon.json"
        MSG="!! Raccoon Detected !! ($CDS_CONFIDENCE)"
    ;;
    *)
        exit;
    ;;
esac

curl -X POST -H "Content-Type: application/json" \
    -d @${FILE} http://kodi:kodi@kodi-main:8080/jsonrpc

if [ -x "${SCRIPT_DIR}/.env/bin/matrix-commander" ]; then
    matrix-commander -m "$MSG" --user @daviddpd:matrix.org
fi

