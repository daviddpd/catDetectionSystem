#!/bin/bash

set -x 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case $1 in
    cat)
        FILE="${SCRIPT_DIR}/kodi-cat.json"
    ;;
    raccoon)
    	FILE="${SCRIPT_DIR}/kodi-raccoon.json"
    ;;
esac

curl -X POST -H "Content-Type: application/json" \
    -d @${FILE} http://kodi:kodi@kodi-main:8080/jsonrpc
