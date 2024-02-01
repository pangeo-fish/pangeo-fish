# tag format

The sensors involved are produced by a range of manufacturers, and thus dumps are usually not in a unified format. To make data ingestion easier, this document defines a standardized format that any such data should be translated to.

## overall structure

The data repository contains one CSV file detailing all the acoustic receiver deployments involved (`stations.csv`), and one directory per tag deployment:

```
tag
├── A19124
│   ├── acoustic.csv
│   ├── dst.csv
│   ├── tagging_events.csv
│   └── metadata.json
├── ...
└── stations.csv
```

Each tag deployment directory must contain the DST log (`dst.csv`) and the lifetime data (`tagging_events.csv`). It may also contain a file with all the acoustic detections (`acoustic.csv`) and additional metadata (`metadata.json`).

## csv formatting

All CSV files must separate columns using `,` and new lines using `\n` characters.

All floating point (or fixed-point) values must use `.` as a decimal separator.

All time values must be given in ISO8601 format (`YYYY-MM-DDTHH:MM:SS±ZZZZ` or `YYYY-MM-DDTHH:MM:SS`). However, time zone-naive datetime data (i.e. without timezone information) will be assumed to be UTC.

Strings containing `,` or `\n` must be wrapped in double quotes (`""`).

## `dst.csv`: DST log

The DST log contains the data measured by the DST tag.

The file must have three columns: `time`, `temperature`, and `pressure`.

For example:

```csv
time,temperature,pressure
2022-07-21 12:12:30,10.1,14.3
2022-08-01 07:08:01,1.3,17.3
```

## `tagging_events.csv`: lifetime data

The lifetime data describes the start and end of the natural behavior of the fish, beginning with the release after the tagging and ending with its death.

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

## `acoustic.csv`: Acoustic detections

This file contains information about acoustic detections.

It must contain at least two columns: `deployment_id` and `time`.

It may contain additional columns, such as the position of detection.

For example:

```csv
time,deployment_id
2022-08-10 22:11:00,176492
```

or

```csv
time,deployment_id,longitude,latitude
2022-08-10 22:11:00,176492,-3.88402,47.78820
2022-08-10 23:25:31,176492,-3.68471,47.81740
```

## `metadata.json`: arbitrary metadata

This file may contain additional metadata.

It must be in JSON format and the top-level structure must be an object. Additionally, the keys must be strings. Any valid JSON value is allowed.

For example:

```json
{ "pit_tag_id": "A19124", "acoustic_tag_id": "OPI-372" }
```
