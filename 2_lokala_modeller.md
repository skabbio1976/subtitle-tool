En stegad approach med tre backends:

Helsinki-NLP/opus-mt-en-sv som lättvikts-fallback — pip-installeras, kräver ingen GPU, körs överallt, ~300MB. Kvaliteten är "Google Translate-nivå" men den är gratis, snabb, och kräver noll setup. Perfekt default för folk utan Ollama.
Qwen3 8B via Ollama som mellanläge — för folk som redan har Ollama uppsatt.
Anthropic API för bäst kvalitet — för de som har nyckel.

Helsinki-NLP-modellen är det mest intressanta tillägget tycker jag, just för att den inte kräver att användaren installerar Ollama. Bara pip install transformers sentencepiece och tre rader Python:
pythonfrom transformers import pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-sv")
result = translator("How do I get myself into these things?")
Den saknar helt kontextmedvetenhet och idiom-hantering, men för din use case (undertexter som redan är korta meningar) fungerar den förvånansvärt bra på rak dialog. Och den kan vara det snabbaste sättet att ge alla användare någon form av översättning utan att de behöver ett konto eller tung mjukvara.
