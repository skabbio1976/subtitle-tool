jag ska leka lite med qwen eller nån modell och försöka regex ut en konversation och skicka konversationer i batchar för då blir det inte så tung context att ta hänsyn till, kan det hjälpa?

2:58 PM

Utvärderade konversationsgränsdetektering och batchstrategi för optimerad kontextkvalitet
Utvärderade konversationsgränsdetektering och batchstrategi för optimerad kontextkvalitet
The user is asking about an interesting idea: using regex to detect conversation boundaries in subtitles and sending conversations as batches rather than arbitrary fixed-size batches. This way each batch would be a natural dialogue unit, giving the model better context without needing the context-window trick from the improvements doc.

This is a smart idea. Let me think about the pros and cons.

Pros:

Natural context boundaries - the model sees a complete scene/conversation
No need for overlapping context windows between batches
Better translation of pronouns, references, tone shifts
Smaller batches = less chance of the model losing track
Cons/challenges:

Detecting conversation boundaries from SRT alone is tricky
Could end up with very uneven batch sizes
The most reliable approach would be looking for time gaps between subtitles—anything longer than 3-5 seconds typically signals a scene change—along with markers like music or sound effects and shifts in who's speaking. The time gap method seems pretty solid for this.

Show more
Done
Ja, det är en smart idé — och enklare än du kanske tror. Du behöver inte ens regex egentligen, tidsgapet mellan undertexter är den starkaste signalen. Ett hopp på 3–5+ sekunder betyder nästan alltid scenbyte.

Något i stil med:

python
def split_into_scenes(segments, gap_threshold=4.0):
    scenes = []
    current = [segments[0]]
    
    for prev, seg in zip(segments, segments[1:]):
        gap = srt_to_seconds(seg["start"]) - srt_to_seconds(prev["end"])
        if gap > gap_threshold:
            scenes.append(current)
            current = []
        current.append(seg)
    
    if current:
        scenes.append(current)
    return scenes
Sedan slår du ihop småscener tills du når din batch-gräns, typ 150 rader, men aldrig bryta mitt i en scen. Då får modellen alltid en komplett dialog att jobba med.

Det löser två problem på en gång — du slipper kontextfönstret mellan batchar (för varje batch är redan en sammanhängande kontext), och en lokal 8B-modell som Qwen klarar en 30-raders scen mycket bättre än 200 slumpmässiga rader där den tappar tråden halvvägs.
