---
name: youtube-source
description: Catalog a YouTube video, obtain an authorized transcript, and write a source summary for the book.
---

# YouTube Source

Use this skill when asked to add a YouTube video, transcript, or video summary
to the book's source collection.

## Workflow

1. Read `AGENTS.md`, `sources/youtube/template_youtube.md`, the video index,
   and nearby YouTube entries in `references.bib`. Check the working tree and
   preserve existing changes. Normalize the supplied URL to a single video ID
   and canonical `https://www.youtube.com/watch?v=VIDEO_ID` URL; do not treat
   a playlist as a single video.
2. Obtain metadata from the video itself, for example with `yt-dlp
   --no-playlist --skip-download --dump-single-json URL`. Verify the title,
   channel, publication date, duration, and any named speakers. Do not infer a
   speaker from the channel name. Check the index, bibliography, and existing
   files for the ID before adding anything; update an existing entry instead
   of creating duplicates.
3. Check whether a transcript already exists in
   `sources/youtube/transcripts/`. Before downloading, transcribing, or storing
   copyrighted audio or text, establish that the user has rights or permission
   for that use, or verify a suitable license. Record the basis for reuse and
   attribution. If rights are unclear, ask the user; do not save or generate a
   full transcript merely because captions are publicly accessible.
4. When authorized, inspect available tracks with `yt-dlp --list-subs
   --no-playlist URL`. Prefer an original-language, creator-provided caption
   track; use automatic captions only when a suitable manual track is absent.
   Download the selected track using `--skip-download --write-subs` for
   manual captions or `--write-auto-subs` for automatic captions, with
   `--sub-langs LANGUAGE --sub-format vtt` and an output template in a
   temporary directory. yt-dlp normally adds a language suffix to the filename;
   move the selected file to `sources/youtube/transcripts/VIDEO_ID.vtt` only
   after confirming its language and that it is nonempty and contains
   timestamped cues. Identify its language and origin in the summary.
5. If no suitable captions exist and rights permit transcription, use a
   locally available speech-to-text tool on the video audio to create a
   timestamped VTT at the same path. Do not send audio or transcript text to a
   third-party service. Check the output against the audio when possible and
   label it as machine-generated. If tools, access, or rights are missing,
   report the blocker instead of inventing text or timestamps.
6. Write `sources/youtube/videos/VIDEO_ID.md` using the front matter and
   sections in `sources/youtube/template_youtube.md`. Summarize in original
   language, distinguish the speaker's claims from established facts, and
   use timestamp ranges only where supported by the transcript or video
   chapter markers. State what was actually reviewed and any limitations.
   Identify the book chapters where the source is useful; do not invent facts,
   speakers, or precise timings.
7. Keep `sources/youtube/video-id.json` as a JSON array of records with
   `video_id`, `title`, `url`, `transcript`, and `summary` fields. Paths are
   repository-relative. Preserve existing records, including the legacy
   single-object sample if present, when converting it to an array. Use
   `null` for a transcript that could not be saved, not a nonexistent path.
   Add a matching `@misc` entry in `references.bib`, following existing
   YouTube entries and escaping BibTeX special characters.
8. Validate the JSON, paths, unique IDs and bibliography key, and consistency
   between metadata, index, summary, and any transcript. Render the Quarto
   project as required by `AGENTS.md`, using a safe output directory outside
   `docs/`. Report the resulting files and any unresolved limitations.

Never edit `docs/`, `_extensions/`, or `appendices/glossary.yml` directly.
