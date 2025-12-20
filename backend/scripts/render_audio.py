from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persian_voice.providers.registry import load_providers
from persian_voice.schema import Clip, ModelVariant, Word
from persian_voice.env import load_dotenv
from persian_voice.storage import public_dir, read_json, write_json_atomic
from persian_voice.text import text_for_kind


def load_words(path: Path) -> list[Word]:
    raw = read_json(path)
    words: list[Word] = []
    for item in raw.get("words", []):
        words.append(
            Word(
                id=item["id"],
                fa=item["fa"],
                ar=item.get("ar") or item["fa"],
                fa_diac=item.get("fa_diac") or item["fa"],
                latn=item.get("latn") or "",
                gloss_en=item.get("gloss_en"),
                rarity=item.get("rarity"),
            )
        )
    return words


def load_clips(path: Path) -> list[Clip]:
    if not path.exists():
        return []
    raw = read_json(path)
    clips: list[Clip] = []
    for item in raw.get("clips", []):
        clips.append(
            Clip(
                word_id=item["word_id"],
                model_id=item["model_id"],
                text=item["text"],
                audio_path=item["audio_path"],
                created_at=item["created_at"],
                meta=item.get("meta") or {},
            )
        )
    return clips


_SAFE_SEG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_path_segment(value: str | None, *, fallback: str = "default") -> str:
    seg = (value or "").strip()
    if not seg:
        seg = fallback
    seg = seg.replace("/", "_").replace("\\", "_")
    seg = _SAFE_SEG_RE.sub("_", seg)
    seg = seg.strip("._-")
    return seg or fallback


def short_error_message(exc: BaseException, *, limit: int = 240) -> str:
    msg = str(exc).strip().replace("\n", " ")
    msg = re.sub(r"\s+", " ", msg)
    if len(msg) <= limit:
        return msg
    return msg[: max(0, limit - 1)] + "…"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Render missing audio clips for all configured providers (idempotent).")
    parser.add_argument("--words", default="data/words.json", help="Words JSON path relative to public dir.")
    parser.add_argument("--models", default="data/models.json", help="Models JSON path relative to public dir.")
    parser.add_argument("--clips", default="data/clips.json", help="Clips JSON path relative to public dir.")
    parser.add_argument(
        "--providers",
        default=None,
        help="Comma-separated provider ids (default: $PERSIAN_VOICE_PROVIDERS or 'all').",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first synthesis error.")
    parser.add_argument("--verbose", action="store_true", help="Print per-word skip/regenerate details.")
    args = parser.parse_args()

    pub = public_dir()
    words_path = pub / args.words
    models_path = pub / args.models
    clips_path = pub / args.clips

    words = load_words(words_path)

    providers = load_providers(args.providers)

    model_variants: list[ModelVariant] = []
    model_variants_by_provider: dict[str, list[ModelVariant]] = {}
    for provider in providers:
        variants = list(provider.list_model_variants())
        model_variants_by_provider[provider.provider_id] = variants
        model_variants.extend(variants)

    print(f"Words: {len(words)} ({words_path})")
    print(f"Providers: {len(providers)} ({', '.join(p.provider_id for p in providers)})")
    for provider in providers:
        available, reason = provider.is_available()
        variants = model_variants_by_provider.get(provider.provider_id, [])
        available_variants = sum(1 for v in variants if v.available)
        suffix = ""
        if not available:
            suffix = f" — unavailable: {reason or 'unknown reason'}"
        print(f"  - {provider.provider_id}: {available_variants}/{len(variants)} model-variants{suffix}")

    available_models = sum(1 for m in model_variants if m.available)
    unavailable_models = len(model_variants) - available_models
    print(f"Model variants: {len(model_variants)} (available={available_models}, unavailable={unavailable_models})")

    write_json_atomic(
        models_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": [
                {
                    "id": m.id,
                    "provider_id": m.provider_id,
                    "provider_label": m.provider_label,
                    "engine_id": m.engine_id,
                    "voice_id": m.voice_id,
                    "input_kind": m.input_kind,
                    "label": m.label,
                    "group": m.group,
                    "audio_format": m.audio_format,
                    "available": m.available,
                    "unavailable_reason": m.unavailable_reason,
                }
                for m in model_variants
            ],
        },
    )

    provider_by_id = {p.provider_id: p for p in providers}

    existing = load_clips(clips_path)
    existing_map: dict[tuple[str, str], Clip] = {(c.word_id, c.model_id): c for c in existing}

    updated: dict[tuple[str, str], Clip] = dict(existing_map)

    failures: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    missing_audio = 0
    failed = 0

    for model in model_variants:
        if not model.available:
            if args.verbose:
                print(f"[skip] {model.id}: unavailable: {model.unavailable_reason or 'unknown reason'}")
            continue
        provider = provider_by_id.get(model.provider_id)
        if not provider:
            if args.verbose:
                print(f"[skip] {model.id}: provider not loaded ({model.provider_id})")
            continue

        model_generated = 0
        model_skipped = 0
        model_missing_audio = 0
        model_failed = 0

        provider_seg = safe_path_segment(model.provider_id)
        engine_seg = safe_path_segment(model.engine_id)
        voice_seg = safe_path_segment(model.voice_id)
        for word in words:
            key = (word.id, model.id)
            word_seg = safe_path_segment(word.id, fallback="word")
            audio_rel = f"audio/{provider_seg}/{engine_seg}/{voice_seg}/{model.input_kind}/{word_seg}.{model.audio_format}"
            audio_abs = (pub / audio_rel).resolve()
            pub_resolved = pub.resolve()
            if pub_resolved not in audio_abs.parents:
                raise RuntimeError(f"Refusing to write outside public dir: {audio_abs}")
            audio_path_for_web = audio_rel

            existing_clip = updated.get(key)
            if audio_abs.exists() and not existing_clip:
                text = text_for_kind(word, model.input_kind)
                updated[key] = Clip(
                    word_id=word.id,
                    model_id=model.id,
                    text=text,
                    audio_path=audio_path_for_web,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    meta={
                        "provider_id": model.provider_id,
                        "engine_id": model.engine_id,
                        "voice_id": model.voice_id,
                        "input_kind": model.input_kind,
                    },
                )
                skipped += 1
                model_skipped += 1
                if args.verbose:
                    print(f"[skip] exists (no clip): {audio_rel}")
                continue
            if existing_clip and audio_abs.exists():
                skipped += 1
                model_skipped += 1
                if args.verbose:
                    print(f"[skip] exists: {audio_rel}")
                continue

            if not audio_abs.exists() and existing_clip:
                missing_audio += 1
                model_missing_audio += 1
                if args.verbose:
                    print(f"[warn] missing audio on disk, regenerating: {audio_rel}")

            text = text_for_kind(word, model.input_kind)
            try:
                provider.synthesize(model=model, text=text, out_path=audio_abs)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                model_failed += 1
                failure = {
                    "word_id": word.id,
                    "model_id": model.id,
                    "provider_id": model.provider_id,
                    "engine_id": model.engine_id,
                    "voice_id": model.voice_id,
                    "input_kind": model.input_kind,
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": short_error_message(exc),
                }
                failures.append(failure)
                print(f"[error] {word.id} · {model.id}: {failure['error_type']}: {failure['error']}", file=sys.stderr)
                if args.fail_fast:
                    raise
                continue

            clip = Clip(
                word_id=word.id,
                model_id=model.id,
                text=text,
                audio_path=audio_path_for_web,
                created_at=datetime.now(timezone.utc).isoformat(),
                meta={
                    "provider_id": model.provider_id,
                    "engine_id": model.engine_id,
                    "voice_id": model.voice_id,
                    "input_kind": model.input_kind,
                },
            )
            updated[key] = clip
            generated += 1
            model_generated += 1
            print(f"Generated {audio_rel}")

        print(
            f"[model] {model.id}: generated={model_generated} skipped={model_skipped}"
            f" missing_audio={model_missing_audio} failed={model_failed}"
        )

    # Persist clips.json
    clips_by_model: dict[str, list[Clip]] = defaultdict(list)
    for clip in updated.values():
        clips_by_model[clip.model_id].append(clip)

    all_clips: list[Clip] = []
    for model_id in sorted(clips_by_model.keys()):
        all_clips.extend(sorted(clips_by_model[model_id], key=lambda c: c.word_id))

    write_json_atomic(
        clips_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats": {"generated": generated, "skipped": skipped, "missing_audio": missing_audio, "failed": failed},
            "failures": failures,
            "clips": [
                {
                    "word_id": c.word_id,
                    "model_id": c.model_id,
                    "text": c.text,
                    "audio_path": c.audio_path,
                    "created_at": c.created_at,
                    "meta": c.meta,
                }
                for c in all_clips
            ],
        },
    )

    print(f"Stats: generated={generated} skipped={skipped} missing_audio={missing_audio} failed={failed}")
    print(f"Wrote {models_path}")
    print(f"Wrote {clips_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
