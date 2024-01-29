# tag format

The sensors involved are produced by a range of manufacturers, and thus dumps are usually not in a unified format. To make data ingestion easier, this document defines a standardized format that any such data should be translated to.

## overall structure

The data repository contains one CSV file detailing all the acoustic receiver deployments involved (`stations.csv`), and one directory per tag deployment:

```
tag
├── A19124
│   ├── acoustic.csv
│   ├── dst.csv
│   ├── dst_deployment.csv
│   └── metadata.json
├── ...
└── stations.csv
```

Each tag deployment directory must contain the DST log (`dst.csv`) and the DST deployment data (`dst_deployment.csv`). It may also contain a file with all the acoustic detections (`acoustic.csv`) and additional metadata (`metadata.json`).

## csv formatting

All CSV files must separate columns using `,` and new lines using `\n` characters.

All floating point (or fixed-point) values must use `.` as a decimal separator.

All time values must be given in ISO8601 format (`YYYY-MM-DD HH:MM:SS±ZZZZ`), where the time zone may be omitted. However, if it is not explicitly specified the time must be in UTC. _TODO: should we instead strictly follow ISO8601, i.e. use a format of `%Y-%m-%DT%H:%M:%SZ` or `%Y-%m-%DT%H:%M:%S±%z` (i.e. trailing `Z` is UTC)?_

Strings containing `,` or `\n` must be wrapped in quotes.

## `dst_deployment.csv`: DST deployment data

The DST deployment data describes the start and end of the natural behavior of the fish, beginning with the release after the tagging and ending on its death.

It must have four columns: `event_name`, `time`, `latitude`, and `longitude`.

The used events are: `release`, `recapture`, `fish_death` (more are possible but will be ignored). The file must contain entries for `release` and one of `recapture` or `fish_death`.

`latitude` and `longitude` are fixed-point representations of the position in degree, with a `.` as separator. `latitude` must be in a range of `-90°` to `90°`, while `longitude` may be given in either `0°` to `360°` or `-180°` to `180°`. If the position is unknown, both `latitude` and `longitude` must be set to `NA`.

For example:

```csv
event_name,time,latitude,longitude
release,2023-07-13 13:21:57,48.21842,-4.08578
recapture,2023-09-17 05:21:07,47.37423,-3.87582
```

or

```csv
event_name,time,latitude,longitude
release,2023-07-13 13:21:57,48.21842,-4.08578
fish_death,2023-09-17 05:21:07,NA,NA
```
