# TODO

## Translation quality

- [ ] Investigate too-fast text — some translated subtitles display for too short a time relative to the amount of text
- [ ] Swedish translation goes out of sync — suspected cause: post-processing (`_split_long_segment` / `_merge_short_segments`) changes segment count/timing after translation
- [ ] Consider skipping or simplifying post-processing for translated files (they already have correct timing from the source)
- [ ] Validate that line count in == line count out after translation (the Claude batch has a fallback but it can mask problems)
- [ ] Investigate letting Haiku handle post-processing via prompt instead of regex/code — more flexible, can handle line length, natural Swedish, timing-awareness
- [ ] Add a flag (e.g. `--quality`) to choose Haiku post-processing vs fast regex — regex as default for batch/speed, Haiku for quality on individual files
- Test file: `/media/win/big/tv/Dallas(1978)/Dallas S09E26 Nothing's Ever Perfect/`
