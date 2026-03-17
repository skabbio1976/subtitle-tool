# TODO

## Translation quality

- [x] Investigate too-fast text — fixed: translate now uses `_wrap_subtitle_line()` (display-only line wrapping) instead of `_split_long_segment` which changed segment count/timing
- [x] Swedish translation goes out of sync — fixed: translate pipeline no longer runs `_split_long_segment`/`_merge_short_segments`; timestamps preserved 1:1 from source
- [x] Consider skipping or simplifying post-processing for translated files — done: translate uses `_wrap_subtitle_line()` only
- [x] Validate that line count in == line count out after translation — done: `_parse_numbered_response()` with explicit fallback to original text and warnings
- [ ] Investigate letting Haiku handle post-processing via prompt instead of regex/code — more flexible, can handle line length, natural Swedish, timing-awareness
- [ ] Add a flag (e.g. `--quality`) to choose Haiku post-processing vs fast regex — regex as default for batch/speed, Haiku for quality on individual files
- Test file: `/media/win/big/tv/Dallas(1978)/Dallas S09E26 Nothing's Ever Perfect/`
