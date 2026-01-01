# TODO: Create Persian Speaker Profile for OuteTTS

## Problem

OuteTTS uses `EN-FEMALE-1-NEUTRAL` (English speaker) by default. When generating Persian text, the model inherits the English accent and prosody, resulting in poor quality output.

## Solution

Create a Persian speaker profile from a Persian audio sample.

## How to Create a Speaker Profile

```python
speaker = interface.create_speaker(
    audio_path="persian_sample.wav",
    transcript="متن دقیق فارسی که در فایل صوتی گفته شده",  # Optional but recommended
    whisper_model="turbo",
    whisper_device="cuda",
)
interface.save_speaker(speaker, "FA-FEMALE-1.json")
```

## Audio Requirements

| Requirement | Details |
|------------|---------|
| Duration | 10-15 seconds (max 20s) |
| Word count | ~20 words / 2-3 natural sentences |
| Content | Flowing sentences, NOT individual words |
| Language | Must be Persian |
| Quality | Clean, no clipping, no background noise |
| Format | WAV recommended |
| Sample rate | 16kHz (Whisper resamples internally) |

## What to Record

- Natural, flowing Persian sentences
- The emotion/accent/tone you want in output
- One continuous recording (not spliced)
- Consistent volume throughout

Example content:
```
سلام، من یک نمونه صدای فارسی هستم که برای ساخت پروفایل گوینده استفاده می‌شود.
این جمله‌ها باید طبیعی و روان باشند.
```

## What NOT to Do

- Individual words with pauses
- English audio for Persian output
- Multiple takes spliced together
- Whispered or shouted speech
- Audio with background noise

## Why Sentences, Not Words

OuteTTS uses Whisper transcription internally to align audio with text. Individual words give poor alignment and limited phonetic coverage.

## Whisper Language Detection Issue

For short non-English audio, Whisper can misidentify the language. Workaround: provide the transcript manually to bypass auto-detection.

GitHub issue requesting fix: https://github.com/edwko/OuteTTS/issues/74

## Quick Test Option

1. Use the Speaker Creator tool: https://bricksdisplay-outetts-speaker-creator.hf.space/
2. Upload 10-15 seconds of Persian audio
3. Download the speaker.json
4. Test locally or add to Replicate

## Integration with Replicate

Once we have a working Persian speaker profile:
1. Add FA-FEMALE-1.json to the Replicate model
2. Update predict.py to support speaker selection
3. Update persian-voice provider to offer Persian speaker option

## Resources

- OuteTTS GitHub: https://github.com/edwko/OuteTTS
- HuggingFace Model: https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B
- Speaker Creator Tool: https://bricksdisplay-outetts-speaker-creator.hf.space/
- Language support issue: https://github.com/edwko/OuteTTS/issues/90

## Persian Language Status in OuteTTS

Persian is in the "Moderate Training Data" tier - works but with "occasional limitations". Creating a native Persian speaker profile should significantly improve quality.
