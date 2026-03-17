# subtitle_tool.py — Förbättringsförslag

Baserat på hands-on-arbete med en typisk problematisk SRT-fil (The Master S01E01) där originalet hade fragmenterade rader, megablock utan interpunktion, och talspråkig dialog. Sorterat efter prioritet.

---

## 1. System-prompt + bättre user-prompt (translate_batch_claude)

**Rad:** ~768–793  
**Problem:** Haiku får minimal vägledning. Ingen system-prompt alls. User-prompten säger bara "translate" utan att specificera domän, register, eller formatkrav.  
**Effekt:** Störst kvalitetslyft per arbetsinsats.

### System-prompt (lägg till i body)

```python
body = json.dumps({
    "model": model,
    "max_tokens": 8192,
    "system": (
        "You are an expert subtitle translator for film and television. "
        "You produce natural, idiomatic translations that sound like "
        "real spoken dialogue — not written text. "
        "You match the register and tone of each character: "
        "slang stays slang, formal stays formal. "
        "You adapt idioms to equivalent expressions in the target language "
        "rather than translating literally. "
        "You never explain, add notes, or deviate from the numbered format."
    ),
    "messages": [{"role": "user", "content": prompt}],
}).encode()
```

### Förbättrad user-prompt

```python
prompt = (
    f"Translate these subtitle lines from {source_name} to {target_name}.\n\n"
    f"Rules:\n"
    f"- Return ONLY numbered translations, matching input numbering exactly.\n"
    f"- This is spoken dialogue from a film/TV show. Use natural, "
    f"colloquial {target_name} appropriate for the tone.\n"
    f"- Adapt idioms and slang to equivalent {target_name} expressions "
    f"rather than translating literally.\n"
    f"- Keep proper nouns, place names, and titles unchanged.\n"
    f"- Keep each line under 42 characters when possible (subtitle display limit).\n"
    f"- If a line has multiple speakers, preserve that structure.\n"
    f"- Do not add explanations, notes, or commentary.\n\n"
    f"{numbered}"
)
```

### Varför det hjälper

Utan system-prompt behandlar Haiku uppgiften generiskt. Med den vet modellen att:
- Det är talat språk (inte prosa)
- Register ska matchas (Max slang ≠ McAllisters värdighet)
- Idiom ska anpassas ("snuggle up" → inte "mysa ihop")
- 42 tecken är en hård gräns

---

## 2. Förbearbeta SRT-fragment innan översättning

**Rad:** ~864 (i `translate_subtitles`, direkt efter `parse_srt`)  
**Problem:** `_merge_short_segments` och `_split_long_segment` körs bara på Whisper-output. Befintliga SRT-filer som skickas till `translate_subtitles` kan ha exakt samma fragmentering — en mening splittrad på 2–3 block med 50ms gap.  
**Effekt:** Färre tokens (billigare), och Haiku får kompletta meningar istället för fragment.

### Ny funktion

```python
def _srt_ts_to_seconds(ts: str) -> float:
    """Convert SRT timestamp '01:02:03,456' to seconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _preprocess_for_translation(segments: list[dict]) -> list[dict]:
    """Merge mid-sentence fragments before sending to translation.
    
    Many SRT files split a single sentence across 2-3 entries with
    tiny gaps (< 150ms). This merges them back so the translator
    sees complete thoughts.
    """
    if not segments:
        return segments

    merged = []
    for seg in segments:
        text = seg["text"].replace("\n", " ").strip()

        if merged:
            prev = merged[-1]
            gap = _srt_ts_to_seconds(seg["start"]) - _srt_ts_to_seconds(prev["end"])
            prev_text = prev["text"]

            # Merge if: tiny gap + previous doesn't end a sentence + short fragment
            should_merge = (
                gap < 0.15
                and prev_text
                and prev_text[-1] not in '.!?"\')'
                and (len(text.split()) <= 3 or text[0].islower())
            )

            if should_merge:
                prev["text"] = f"{prev_text} {text}"
                prev["end"] = seg["end"]
                continue

        merged.append({**seg, "text": text})

    return merged
```

### Var den ska kallas

```python
def translate_subtitles(srt_path, target_lang, api_key, model, force):
    # ... (befintlig kod) ...
    segments = parse_srt(srt_path)
    
    # NYTT: förbearbeta innan översättning
    original_count = len(segments)
    segments = _preprocess_for_translation(segments)
    if len(segments) < original_count:
        print(f"  Merged {original_count - len(segments)} fragment(s)")
    
    # ... resten av funktionen ...
```

### Exempel från verkligheten

Originalfil hade:
```
6:  00:00:26,540 --> 00:00:27,997  "The kind who build things and care about"
7:  00:00:27,997 --> 00:00:28,180  "them."
```
Gap: 0ms. Haiku ser "them." isolerat och har ingen aning om sammanhanget. Efter merge: en rad, en komplett mening.

---

## 3. Kontextfönster mellan batchar

**Rad:** ~887–912  
**Problem:** Batch 1 (rad 1–200) och batch 2 (rad 201–400) skickas helt isolerade. Om rad 201 säger "I told *him*...", vet Haiku inte vem "him" syftar på. Pronomen, tonläge och karaktärsröster tappar kontinuitet.  
**Effekt:** Märkbar förbättring vid batchgränser, särskilt för dialogtunga scener.

### Implementation

```python
# I translate_subtitles, inuti batch-loopen:

context_window = 10  # antal föregående rader att inkludera som kontext
batch_size = 200

for batch_idx in range(0, len(all_texts), batch_size):
    batch = all_texts[batch_idx:batch_idx + batch_size]
    batch_num = batch_idx // batch_size + 1

    # Bygg numrerad input
    numbered = "\n".join(
        f"{i+1}: {t.replace(chr(10), ' ')}" for i, t in enumerate(batch)
    )

    # Lägg till kontext från föregående batch
    if batch_idx > 0:
        ctx_start = max(0, batch_idx - context_window)
        context_lines = all_texts[ctx_start:batch_idx]
        # Inkludera redan översatta versioner om tillgängliga
        if len(translated_texts) >= batch_idx:
            ctx_translated = translated_texts[ctx_start:batch_idx]
            context_str = "\n".join(
                f"[{i+1}] {orig} → {trans}"
                for i, (orig, trans) in enumerate(zip(context_lines, ctx_translated))
            )
        else:
            context_str = "\n".join(
                f"[{i+1}] {t.replace(chr(10), ' ')}"
                for i, t in enumerate(context_lines)
            )

        prompt = (
            f"For context, here are the preceding subtitle lines "
            f"and their translations (do NOT re-translate these):\n"
            f"{context_str}\n\n"
            f"Now translate these lines:\n{numbered}"
        )
    else:
        prompt = f"Translate these subtitle lines ...\n\n{numbered}"
```

### Notera

Att skicka med redan översatta rader (`orig → trans`) är bättre än bara original — det ger Haiku både semantisk kontext och stilkontinuitet. Kostar några extra input-tokens men värt det.

---

## 4. Post-processing: radbrytning av översatta rader

**Var:** Efter batch-översättning, innan `write_srt` kallas  
**Problem:** Haiku respekterar inte alltid 42-teckensgränsen trots instruktionen. Svenska är ~15% längre än engelska, så rader som var okej på engelska spiller över.  
**Effekt:** Garanterar att alla rader visas korrekt i videospelaren.

### Implementation

```python
def _wrap_subtitle_line(text: str, max_chars: int = 42) -> str:
    """Wrap a subtitle line to max 2 lines, balanced around the middle."""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    # Redan tvårading? Kolla om varje rad är OK
    if "\n" in text:
        lines = text.split("\n")
        if all(len(l) <= max_chars for l in lines) and len(lines) <= 2:
            return text

    # Platta ut och hitta bästa brytpunkt
    flat = text.replace("\n", " ")
    mid = len(flat) / 2
    best_pos = -1
    best_diff = len(flat)

    # Sök mellanslag nära mitten
    for i, ch in enumerate(flat):
        if ch == ' ':
            diff = abs(i - mid)
            if diff < best_diff:
                best_diff = diff
                best_pos = i

    if best_pos > 0:
        line1 = flat[:best_pos].rstrip()
        line2 = flat[best_pos + 1:].lstrip()
        # Kolla att ingen rad fortfarande är för lång
        if len(line1) <= max_chars and len(line2) <= max_chars:
            return f"{line1}\n{line2}"

    return flat  # fallback: returnera platt om inget funkar


# Anropa efter översättningen, innan write_srt:
for seg in output_segments:
    seg["text"] = _wrap_subtitle_line(seg["text"])
```

---

## 5. Robustare output-parsing

**Rad:** ~828–840  
**Problem:** Nuvarande parser hanterar bara `N: text`. Haiku kan returnera `N. text`, `N) text`, extra blankrader, eller hoppa över nummer. Nuvarande kod tappar rader tyst.

### Ersätt nuvarande parsing

```python
def _parse_numbered_response(response: str, expected_count: int,
                              fallback_texts: list[str]) -> list[str]:
    """Parse numbered translation response with fallback strategies."""
    lines = response.strip().split("\n")
    result = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Matcha: "N: text", "N. text", "N) text", "N - text"
        m = re.match(r'^(\d+)\s*[.:)\-]\s*(.*)', line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            if text:  # Skippa tomma
                result[idx] = text

    # Bygg output med fallback till originaltext för saknade rader
    output = []
    missing = []
    for i in range(1, expected_count + 1):
        if i in result:
            output.append(result[i])
        else:
            missing.append(i)
            # Fallback: behåll originaltexten
            output.append(fallback_texts[i - 1] if i - 1 < len(fallback_texts) else "")

    if missing:
        print(f"  {C_YELLOW}Warning: {len(missing)} line(s) missing from response, "
              f"kept original: {missing[:10]}{'...' if len(missing) > 10 else ''}{C_RESET}")

    return output
```

### Anropa istället för befintlig parsing

```python
result = _parse_numbered_response(response_text, len(batch), batch)
```

---

## 6. Whisper-parametrar: balans mellan koherens och hallucinationer

**Rad:** ~349–353  
**Problem:** `condition_on_previous_text=False` förhindrar hallucinationsloops men kostar koherens — inkonsekvent casing, fragmenterade meningar, namn som stavas olika.

### Förslag: aktivera med skyddsräcken

```python
kwargs = {
    "condition_on_previous_text": True,      # Behåll koherens
    "no_speech_threshold": 0.6,
    "hallucination_silence_threshold": 2,
    "repetition_penalty": 1.1,               # Motverka loops
    "compression_ratio_threshold": 2.4,      # Filtrera komprimerad hallucinering
    "log_prob_threshold": -1.0,              # Skippa segment med låg konfidens
}
```

### Vad varje parameter gör

| Parameter | Värde | Effekt |
|-----------|-------|--------|
| `condition_on_previous_text` | `True` | Whisper ser föregående text → bättre flöde |
| `repetition_penalty` | `1.1` | Straffar upprepade tokens → bryter loops |
| `compression_ratio_threshold` | `2.4` | Filtrerar segment som är "för komprimerade" (typiskt hallucinationer) |
| `log_prob_threshold` | `-1.0` | Skippar segment där modellen är osäker |

### Notera

Dessa parametrar är beroende av innehållstyp. Musik/tyst film kräver mer aggressiv filtrering. Dialog-tung TV (som The Master) klarar sig med mildare inställningar. Kan vara värt att exponera som CLI-flaggor eller presets:

```
--whisper-preset dialogue   # condition_on_previous_text=True, mild filtering
--whisper-preset music      # condition_on_previous_text=False, aggressive filtering
```

---

## Bonustips

### A. Batch-storlek

200 rader per batch (rad 882) är konservativt. Med Haiku 4.5 och 8192 max_tokens kan du ofta köra 300–400 rader. Men: med kontextfönstret (punkt 3) äter du input-tokens, så 150–200 kan vara sweet spot.

### B. Temperatur

Du sätter ingen `temperature`. Default (1.0) ger variation som är bra för kreativ text men kan ge inkonsekvent terminologi i undertexter. Prova:

```python
"temperature": 0.3,  # Mer deterministisk, konsekvent ordval
```

### C. Felåterhämtning per batch

Om batch 3 av 5 failar, tappar du allt. Lägg till retry per batch med checkpoint:

```python
# Spara progress löpande
checkpoint_path = output_path.with_suffix('.tmp')
# ... efter varje lyckad batch:
_write_partial(output_segments, translated_texts, checkpoint_path)
```

### D. Kvalitetsvalidering

Lägg till en enkel sanity-check efter översättning:

```python
def _translation_sanity_check(original: str, translated: str) -> list[str]:
    """Flagga potentiella problem."""
    warnings = []
    if len(translated) > len(original) * 2.5:
        warnings.append("suspiciously long")
    if len(translated) < len(original) * 0.3:
        warnings.append("suspiciously short")
    if translated.strip() == original.strip():
        warnings.append("untranslated")
    return warnings
```
