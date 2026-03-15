# TODO

## Translation quality

- [ ] Undersök för snabba texter — vissa översatta subtitles visas för kort tid relativt textmängden
- [ ] Svensk översättning hamnar ur sync — misstänkt orsak: post-processing (`_split_long_segment` / `_merge_short_segments`) ändrar antal segment/timing efter översättning
- [ ] Överväg att skippa eller förenkla post-processing för översatta filer (de har redan korrekt timing från källan)
- [ ] Validera att antal rader in == antal rader ut efter översättning (Claude-batchen har fallback men den kan maska problem)
- [ ] Undersök att låta Haiku sköta post-processing via prompt istället för regex/kod — mer flexibelt, kan hantera radlängd, naturlig svenska, timing-awareness
- [ ] Lägg till växel (t.ex. `--quality`) för att välja Haiku-postprocessing vs snabb regex — regex som default för batch/speed, Haiku för kvalitet på enstaka filer
- Testfil: `/media/win/big/tv/Dallas(1978)/Dallas S09E26 Nothing's Ever Perfect/`
