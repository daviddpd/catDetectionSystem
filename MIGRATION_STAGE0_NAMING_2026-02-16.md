# Stage 0 Migration Notes (February 16, 2026)

This document tracks naming and spelling normalization changes introduced in Stage 0.

## Project Name Normalization
- Canonical project name: `catDetectionSystem`
- Legacy variant: `catDectionSystem`
- Policy: new docs and user-facing text use `catDetectionSystem`.

Note:
- Existing local directory names may still use legacy spelling for compatibility.
- Runtime scripts now avoid hard-coded repository paths and derive repo path dynamically.

## Command and Script Migration
Canonical entrypoint added:
- `./cds detect --uri <uri-or-path> [options]`
- `./cds detect-c4`

Legacy wrappers retained:
- `./run.sh`
- `./run-c4.sh`

Legacy wrapper behavior:
- Both wrappers still run.
- Both wrappers print deprecation warnings and point to `./cds`.

## CLI Compatibility Aliases
`tools/xml2txt.py`:
- New preferred flag: `--writeText`
- Legacy alias retained: `--writetext`

XML format note:
- XML annotations in this repository follow Pascal VOC-style structure.
- Reference: https://host.robots.ox.ac.uk/pascal/VOC/ and https://roboflow.com/formats/pascal-voc-xml
- Typical creators:
  - legacy runtime XML export in `rtsp-object-ident.py`
  - external annotation tools such as LabelImg

## Asset Path Compatibility
Audio file lookup now checks both:
- `assets/Meow-cat-sound-effect.mp3` (canonical spelling)
- `assests/Meow-cat-sound-effect.mp3` (legacy fallback)

## User-Facing Spelling and Wording Cleanup
Updated:
- `README.md` wording and spelling
- CLI help text in:
  - `rtsp-object-ident.py`
  - `tools/xml2txt.py`
  - `tools/xml-change-tool.py`
