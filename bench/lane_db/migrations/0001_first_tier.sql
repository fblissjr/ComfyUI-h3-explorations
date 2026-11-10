-- The masked lane's record, first tier: what a person said and decided, and a project's closing state.
-- One file for the lane (data/lane.sqlite, never in git). The plan: docs/masked_v2v/2026-11-10_simplification_plan.md, step 1.
--
-- Rules every table here keeps:
--   * STRICT tables, typed columns, a list as a child table (its place in the list is `ord`).
--   * A video's footage facts hang off `video` and are shared by every project on it; a replacement's facts hang off
--     `project`. Each table below says which it is.
--   * Frame numbers are the video's source frames, and a range is inclusive.
--   * A file is a place (`location`: a named root and a relative path) and, once its bytes are known, an `artifact`
--     (its sha256). No absolute path is stored outside import_source.raw.
--   * Nothing is edited where history matters: a later row or state supersedes, and "what stands" is a view.
--   * JSON sits in import_source.raw only (and, from the second tier, step.argv). No tool reads it.
-- The second tier (previews, loads' settings, steps, what each step read, how each link is known) is the next migration.

-- places and bytes ----------------------------------------------------------------------------------------------------
CREATE TABLE root (                                    -- resolved at run time (bench/lane_db/db.py, ROOTS), never stored
  name TEXT PRIMARY KEY, what TEXT NOT NULL
) STRICT;
CREATE TABLE artifact (                                -- a file's bytes: the sha256 is the identity
  id INTEGER PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE CHECK (length(sha256) = 64), bytes INTEGER
) STRICT;
CREATE TABLE location (                                -- a place a record names; which bytes are there is said once
  id INTEGER PRIMARY KEY, root TEXT NOT NULL REFERENCES root(name),
  relpath TEXT NOT NULL CHECK (relpath NOT LIKE '/%'),
  artifact_id INTEGER REFERENCES artifact(id),         -- the bytes a hashing pass found there; NULL until one has looked.
                                                       -- What a RECORD says was there is on the row that cites the place.
  size INTEGER, mtime REAL, verified_at TEXT, missing_at TEXT,
  UNIQUE (root, relpath)
) STRICT;
CREATE TRIGGER location_is_written_once BEFORE UPDATE OF artifact_id ON location
WHEN OLD.artifact_id IS NOT NULL AND NEW.artifact_id IS NOT OLD.artifact_id
BEGIN SELECT RAISE(ABORT, 'a location is written once: other bytes at the same place are a new place'); END;

-- the footage, and the projects on it ---------------------------------------------------------------------------------
CREATE TABLE video (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, source_name TEXT NOT NULL, frames INTEGER NOT NULL,
  film_words TEXT,                                     -- plan text.film: what the footage is, for every text
  shots_copy_id INTEGER REFERENCES copy(id)            -- the copy the shots were cut on
) STRICT;
CREATE TABLE project (                                 -- a set of replacements on one video; two projects can share a video
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, video_id INTEGER NOT NULL REFERENCES video(id),
  folder TEXT NOT NULL UNIQUE,                         -- its records' folder under the data root
  starts_at TEXT, ends_before TEXT,                    -- a dated record of the video is the project's when starts_at <= at < ends_before
  said_by TEXT NOT NULL,
  load_prefixes_by TEXT
) STRICT;
CREATE TABLE subject (                                 -- footage: a person of the video, by label
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES video(id), label TEXT NOT NULL, ord INTEGER NOT NULL,
  UNIQUE (video_id, label)
) STRICT;
CREATE TABLE alias (                                   -- footage: an older name of a subject (plan aliases)
  id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL REFERENCES subject(id), alias TEXT NOT NULL, ord INTEGER NOT NULL,
  said_by TEXT, UNIQUE (subject_id, alias)
) STRICT;
CREATE TABLE copy (                                    -- footage: the source file and the working copies made from it
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES video(id),
  kind TEXT NOT NULL CHECK (kind IN ('source', 'working')), name TEXT NOT NULL,
  location_id INTEGER REFERENCES location(id),
  rate REAL NOT NULL, frames INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
  bytes INTEGER NOT NULL, head_sha256 TEXT NOT NULL, first_source_frame INTEGER,
  checked TEXT, made_by TEXT, entered_at TEXT, entered_by TEXT,
  replaced_at TEXT, replaced_by TEXT, why TEXT, ord INTEGER NOT NULL,
  UNIQUE (video_id, name)
) STRICT;

CREATE TABLE shot (                                    -- footage: a cut of the video
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES video(id),
  first INTEGER NOT NULL, last INTEGER NOT NULL CHECK (last >= first),
  scene_score REAL, note TEXT,
  ord INTEGER                                          -- its place in the shots list; NULL for a shot another replaced
) STRICT;
CREATE TABLE shot_from (                               -- footage: the earlier shots a shot was merged, split or moved from
  shot_id INTEGER NOT NULL REFERENCES shot(id), from_shot_id INTEGER NOT NULL REFERENCES shot(id),
  how TEXT NOT NULL CHECK (how IN ('merged', 'split', 'moved')), ord INTEGER NOT NULL,
  PRIMARY KEY (shot_id, from_shot_id)
) STRICT;
CREATE VIEW shot_standing AS SELECT * FROM shot WHERE ord IS NOT NULL;

CREATE TABLE appearance (                              -- footage: whether a subject is on a shot, per a person's word
  id INTEGER PRIMARY KEY, shot_id INTEGER NOT NULL REFERENCES shot(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  present TEXT NOT NULL CHECK (present IN ('yes', 'no', 'unsure')),
  how TEXT, said_by TEXT NOT NULL, frames_from TEXT,
  answered_over_at TEXT, answered_over_by TEXT,        -- set on an answer a later one replaced (shots answered_over[])
  answered_over_ord INTEGER
) STRICT;
CREATE VIEW appearance_standing AS SELECT * FROM appearance WHERE answered_over_ord IS NULL;
CREATE TABLE appearance_frame (                        -- footage: who.<subject>.frames_on[] and frames_boxed[]
  appearance_id INTEGER NOT NULL REFERENCES appearance(id),
  kind TEXT NOT NULL CHECK (kind IN ('on', 'boxed')), ord INTEGER NOT NULL, frame INTEGER NOT NULL,
  PRIMARY KEY (appearance_id, kind, ord)
) STRICT;
CREATE TABLE seen_look (                               -- footage: a part of a subject seen or not on a span, by a look
  id INTEGER PRIMARY KEY, appearance_id INTEGER NOT NULL REFERENCES appearance(id), ord INTEGER NOT NULL,
  first INTEGER NOT NULL, last INTEGER NOT NULL, part TEXT NOT NULL, is_seen TEXT NOT NULL,
  why TEXT, said_by TEXT NOT NULL, entered_by TEXT, at TEXT NOT NULL
) STRICT;
CREATE TABLE seed (                                    -- footage: cast.jsonl, a box naming who is who on a frame
  id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL REFERENCES subject(id), line INTEGER NOT NULL UNIQUE,
  frame INTEGER NOT NULL, x REAL, y REAL, w REAL, h REAL, picture_w INTEGER, picture_h INTEGER,
  answer TEXT NOT NULL CHECK (answer IN ('here', 'none of the outlined people', 'withdrawn')),
  encloses TEXT CHECK (encloses IN ('head', 'body')), drawn TEXT,
  from_table TEXT NOT NULL, said_by TEXT NOT NULL, at TEXT NOT NULL,
  CHECK ((x IS NULL) = (y IS NULL) AND (x IS NULL) = (w IS NULL) AND (x IS NULL) = (h IS NULL))
) STRICT;
CREATE TABLE head_mark (                               -- footage: plan head_marks[], where a head is hidden or whose hair is whose
  id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL REFERENCES subject(id), ord INTEGER NOT NULL,
  first INTEGER NOT NULL, last INTEGER NOT NULL,
  mark TEXT NOT NULL CHECK (mark IN ('hidden', 'hers', 'not hers')), said_by TEXT NOT NULL
) STRICT;
CREATE TABLE voice_span (                              -- footage: plan voice[], who sings or talks where
  id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL REFERENCES subject(id), ord INTEGER NOT NULL,
  first INTEGER NOT NULL, last INTEGER NOT NULL, voice TEXT NOT NULL, said_by TEXT NOT NULL
) STRICT;
CREATE TABLE facial_hair (                             -- footage: plan source_facial_hair, the filmed subject has facial hair
  subject_id INTEGER PRIMARY KEY REFERENCES subject(id), has INTEGER NOT NULL CHECK (has IN (0, 1)), said_by TEXT NOT NULL
) STRICT;
CREATE TABLE not_person (                              -- footage: plan not_people[], a thing detectors take for a person
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES video(id), ord INTEGER NOT NULL,
  what TEXT NOT NULL, counts_as_no_one TEXT, said_by TEXT NOT NULL
) STRICT;
CREATE TABLE not_person_span (
  not_person_id INTEGER NOT NULL REFERENCES not_person(id), ord INTEGER NOT NULL,
  first INTEGER NOT NULL, last INTEGER NOT NULL, PRIMARY KEY (not_person_id, ord)
) STRICT;

-- what a person said: append only -------------------------------------------------------------------------------------
CREATE TABLE answer (                                  -- one line of owner_answers.jsonl
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES video(id),
  project_id INTEGER REFERENCES project(id),           -- set when the answer is about a replacement; NULL for the footage
  about TEXT NOT NULL CHECK (about IN ('footage', 'render', 'plan', 'not an answer', 'not classed')),
  line INTEGER NOT NULL UNIQUE,                        -- its line in the file, as records cite it (owner_answers.jsonl:N)
  at TEXT NOT NULL, who TEXT NOT NULL,
  what TEXT NOT NULL CHECK (what IN ('choice', 'words', 'parts', 'words_mouth', 'words_face', 'removed')),
  page_row TEXT, tile TEXT,                            -- the page's row id, and `n`, the tile on it
  shot_said INTEGER,                                   -- `shot` as the page wrote it: an index on old pages, a first frame on later ones
  subject_id INTEGER REFERENCES subject(id),
  frame INTEGER, source_frame INTEGER,
  source_frame_default INTEGER NOT NULL DEFAULT 0 CHECK (source_frame_default IN (0, 1)),   -- `source_frame: "default"`
  chosen TEXT, words TEXT,
  person_tile INTEGER, person_label TEXT,              -- `person`: a tile's number on the who page, or a label
  table_name TEXT, table_frame INTEGER,
  clip_location_id INTEGER REFERENCES location(id), clip_artifact_id INTEGER REFERENCES artifact(id), clip_not_known TEXT,
  render_location_id INTEGER REFERENCES location(id), render_artifact_id INTEGER REFERENCES artifact(id),
  said_in TEXT, taken_down_by TEXT, relayed_by TEXT, relayed_from TEXT, corrects TEXT,
  CHECK (person_tile IS NULL OR person_label IS NULL)
) STRICT;
CREATE TABLE answer_frame (                            -- `frames` (one span the answer is about) and `bad_frames` (spans it calls wrong)
  answer_id INTEGER NOT NULL REFERENCES answer(id), role TEXT NOT NULL CHECK (role IN ('about', 'bad')),
  ord INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL, PRIMARY KEY (answer_id, role, ord)
) STRICT;
CREATE TABLE answer_part (                             -- a parts answer: the part classes to replace, keep out or leave
  answer_id INTEGER NOT NULL REFERENCES answer(id), role TEXT NOT NULL CHECK (role IN ('replace', 'keep_out', 'leave')),
  ord INTEGER NOT NULL, part_class TEXT NOT NULL, PRIMARY KEY (answer_id, role, ord)
) STRICT;

-- a project's plan, per subject ---------------------------------------------------------------------------------------
CREATE TABLE subject_plan (                            -- project: plan subjects.<s>, what is replaced and from which still
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  region TEXT NOT NULL, said_by TEXT NOT NULL, inputs_by TEXT,
  still_location_id INTEGER NOT NULL REFERENCES location(id),
  margin_px INTEGER NOT NULL, zoom_up_to REAL NOT NULL,
  crop TEXT NOT NULL CHECK (crop IN ('head', 'face', 'cranium')),
  open_mouth TEXT NOT NULL, closed_mouth TEXT NOT NULL, recipe_by TEXT NOT NULL,
  depth_on INTEGER NOT NULL CHECK (depth_on IN (0, 1)), depth_why TEXT,
  look_by TEXT, restore_by TEXT, keep_outs_by TEXT,
  UNIQUE (project_id, subject_id)
) STRICT;
CREATE TABLE plan_keep_out (                           -- project: plan keep_outs[], the keep-outs a subject's loads are made with
  subject_plan_id INTEGER NOT NULL REFERENCES subject_plan(id), ord INTEGER NOT NULL, name TEXT NOT NULL,
  PRIMARY KEY (subject_plan_id, ord)
) STRICT;
CREATE TABLE load_prefix (                             -- project: plan load_prefixes, how a load's name says whose it is
  project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  ord INTEGER NOT NULL, prefix TEXT NOT NULL, PRIMARY KEY (project_id, subject_id, ord)
) STRICT;
CREATE TABLE plan_part (                               -- project: plan parts[]
  subject_plan_id INTEGER NOT NULL REFERENCES subject_plan(id), ord INTEGER NOT NULL, part TEXT NOT NULL,
  PRIMARY KEY (subject_plan_id, ord)
) STRICT;
CREATE TABLE subject_text (                            -- project: plan text.*, the words a load's text is made from
  subject_plan_id INTEGER PRIMARY KEY REFERENCES subject_plan(id),
  who TEXT, she TEXT, her TEXT, looks TEXT, looks_by TEXT, hair TEXT,
  hair_words TEXT, hair_words_by TEXT, picture_hair TEXT, picture_hair_by TEXT,
  sings INTEGER CHECK (sings IN (0, 1))
) STRICT;
CREATE TABLE plan_removal (                            -- project: a thing the plan held and a person took out, with their words
  subject_plan_id INTEGER NOT NULL REFERENCES subject_plan(id),
  what TEXT NOT NULL CHECK (what IN ('keep_outs', 'hair_band', 'pieces', 'restore', 'look.stay', 'text.hair_change')),
  note TEXT NOT NULL, PRIMARY KEY (subject_plan_id, what)
) STRICT;
CREATE TABLE shot_override (                           -- project: plan shot_overrides[], a span whose recipe differs
  id INTEGER PRIMARY KEY, subject_plan_id INTEGER NOT NULL REFERENCES subject_plan(id), ord INTEGER NOT NULL,
  first INTEGER NOT NULL, last INTEGER NOT NULL,
  margin_px INTEGER, zoom_up_to REAL, open_mouth TEXT, closed_mouth TEXT, why TEXT NOT NULL, said_by TEXT NOT NULL
) STRICT;
CREATE TABLE clothing_words (                          -- project: plan clothing[], her clothing in words on a span
  id INTEGER PRIMARY KEY, subject_plan_id INTEGER NOT NULL REFERENCES subject_plan(id), ord INTEGER NOT NULL,
  first INTEGER NOT NULL, last INTEGER NOT NULL, words TEXT NOT NULL, said_by TEXT NOT NULL
) STRICT;
CREATE TABLE look_change (                             -- project: plan look.change[]
  id INTEGER PRIMARY KEY, subject_plan_id INTEGER NOT NULL REFERENCES subject_plan(id), ord INTEGER NOT NULL,
  room TEXT NOT NULL, said_by TEXT NOT NULL
) STRICT;
CREATE TABLE restore_rule (                            -- project: plan restore[], classes given back to the source on a span
  id INTEGER PRIMARY KEY, subject_plan_id INTEGER NOT NULL REFERENCES subject_plan(id), ord INTEGER NOT NULL,
  first INTEGER NOT NULL, last INTEGER NOT NULL, touching TEXT, exception TEXT, exception_by TEXT
) STRICT;
CREATE TABLE restore_class (
  restore_rule_id INTEGER NOT NULL REFERENCES restore_rule(id), ord INTEGER NOT NULL, part_class TEXT NOT NULL,
  PRIMARY KEY (restore_rule_id, ord)
) STRICT;
CREATE TABLE restore_load (                            -- the loads a restore is pinned to, by name
  restore_rule_id INTEGER NOT NULL REFERENCES restore_rule(id), ord INTEGER NOT NULL, load_name TEXT NOT NULL,
  PRIMARY KEY (restore_rule_id, ord)
) STRICT;
CREATE TABLE departure (                               -- project: plan departures[], a span a subject's own rule does not hold on
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  ord INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL, note TEXT NOT NULL, said_by TEXT NOT NULL
) STRICT;
CREATE TABLE shot_order (                              -- project: shots order[], which subject is made first on a shot
  project_id INTEGER NOT NULL REFERENCES project(id), shot_id INTEGER NOT NULL REFERENCES shot(id),
  subject_id INTEGER NOT NULL REFERENCES subject(id), ord INTEGER NOT NULL, said_by TEXT NOT NULL,
  PRIMARY KEY (project_id, shot_id, subject_id)
) STRICT;

-- what a project laid -------------------------------------------------------------------------------------------------
CREATE TABLE load (                                    -- project: a load, by name; its settings come with the second tier
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), name TEXT NOT NULL,
  UNIQUE (project_id, name)
) STRICT;
CREATE TABLE laid (                                    -- project: a render laid on frames for a subject
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id),
  subject_id INTEGER NOT NULL REFERENCES subject(id), load_id INTEGER NOT NULL REFERENCES load(id),
  first INTEGER NOT NULL, last INTEGER NOT NULL,       -- the shot it is laid on
  render_location_id INTEGER NOT NULL REFERENCES location(id), render_artifact_id INTEGER NOT NULL REFERENCES artifact(id),
  zoomed_artifact_id INTEGER REFERENCES artifact(id),
  added_at TEXT NOT NULL, added_by TEXT NOT NULL,
  no_rebuild_why TEXT, partial_lay_why TEXT, seeded_from TEXT,
  owner_before_correction TEXT, owner_corrected_why TEXT,    -- laid owner_was / owner_corrected: a status corrected with no history entry
  list_line INTEGER UNIQUE,                            -- its line in the list as it stands; NULL once removed
  removed_line INTEGER UNIQUE                          -- its line in the removed list
) STRICT;
CREATE TABLE laid_state (                              -- the owner's standing on a piece over time; the last one stands
  laid_id INTEGER NOT NULL REFERENCES laid(id), ord INTEGER NOT NULL,
  owner_status TEXT NOT NULL CHECK (owner_status IN ('kept', 'kept in a joined clip', 'unconfirmed')),
  why TEXT NOT NULL,
  word_file TEXT, word_line INTEGER,                   -- the line the status rests on, as cited (file:line)
  answer_id INTEGER REFERENCES answer(id),             -- that line, where it is a line of the answers file
  at TEXT NOT NULL, by_session TEXT, no_rebuild_why TEXT,
  PRIMARY KEY (laid_id, ord)
) STRICT;
CREATE TABLE laid_frames (                             -- the spans a piece is laid on
  laid_id INTEGER NOT NULL REFERENCES laid(id), ord INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL,
  PRIMARY KEY (laid_id, ord)
) STRICT;
CREATE TABLE laid_replaces (                           -- the loads a piece replaced when it was laid, by name
  laid_id INTEGER NOT NULL REFERENCES laid(id), ord INTEGER NOT NULL, load_name TEXT NOT NULL,
  sha256_start TEXT,                                   -- as the long form says it: "(sha256 7ceada0c..., laid on [[604, 948]])"
  PRIMARY KEY (laid_id, ord)
) STRICT;
CREATE TABLE laid_replaces_frames (                    -- the spans the long form says the replaced piece was laid on
  laid_id INTEGER NOT NULL, replaces_ord INTEGER NOT NULL, ord INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL,
  PRIMARY KEY (laid_id, replaces_ord, ord), FOREIGN KEY (laid_id, replaces_ord) REFERENCES laid_replaces(laid_id, ord)
) STRICT;
CREATE TABLE laid_look (                               -- the owner's looks on a piece, in words
  laid_id INTEGER NOT NULL REFERENCES laid(id), ord INTEGER NOT NULL, words TEXT NOT NULL, PRIMARY KEY (laid_id, ord)
) STRICT;
CREATE TABLE laid_removal (                            -- a piece taken off the list
  laid_id INTEGER PRIMARY KEY REFERENCES laid(id), at TEXT NOT NULL, by_session TEXT NOT NULL, why TEXT NOT NULL,
  no_rebuild_why TEXT, carved_whole_by TEXT
) STRICT;
CREATE TABLE laid_carve (                              -- frames taken off a piece by a later load's keep
  laid_id INTEGER NOT NULL REFERENCES laid(id), ord INTEGER NOT NULL, load_name TEXT NOT NULL,
  reason TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY (laid_id, ord)
) STRICT;
CREATE TABLE laid_carve_frames (
  laid_id INTEGER NOT NULL, carve_ord INTEGER NOT NULL, ord INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL,
  PRIMARY KEY (laid_id, carve_ord, ord), FOREIGN KEY (laid_id, carve_ord) REFERENCES laid_carve(laid_id, ord)
) STRICT;
CREATE VIEW laid_standing AS
  SELECT l.*, s.owner_status, s.why, s.word_file, s.word_line, s.answer_id
  FROM laid l JOIN laid_state s ON s.laid_id = l.id
  WHERE l.list_line IS NOT NULL AND s.ord = (SELECT max(ord) FROM laid_state WHERE laid_id = l.id);

-- a project's cuts ----------------------------------------------------------------------------------------------------
CREATE TABLE cut (                                     -- project: a whole-video cut assembled from the laid list
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), stamp TEXT NOT NULL,
  location_id INTEGER NOT NULL REFERENCES location(id), artifact_id INTEGER NOT NULL REFERENCES artifact(id),
  first INTEGER NOT NULL, last INTEGER NOT NULL,
  made_at TEXT NOT NULL,
  laid_list_sha256_start TEXT NOT NULL, laid_list_lines INTEGER NOT NULL,   -- the laid list it was built from, as its table says
  table_made_at_commit TEXT NOT NULL, source_copy_id INTEGER NOT NULL REFERENCES copy(id),
  is_final INTEGER NOT NULL DEFAULT 0 CHECK (is_final IN (0, 1)), final_said_by TEXT,
  UNIQUE (project_id, stamp)
) STRICT;
CREATE TABLE cut_piece (                               -- a row of the cut's delivery table: a laid piece on a run of frames
  id INTEGER PRIMARY KEY, cut_id INTEGER NOT NULL REFERENCES cut(id), ord INTEGER NOT NULL,
  laid_id INTEGER NOT NULL REFERENCES laid(id),
  first INTEGER NOT NULL, last INTEGER NOT NULL, piece_first INTEGER NOT NULL,
  region TEXT NOT NULL,
  capture_folder TEXT, run_name TEXT,                  -- the capture run the row is tied to, as the table names it
  UNIQUE (cut_id, ord)
) STRICT;
CREATE TABLE cut_piece_restore (                       -- the row's restore= tokens, in order
  cut_piece_id INTEGER NOT NULL REFERENCES cut_piece(id), ord INTEGER NOT NULL, token TEXT NOT NULL,
  PRIMARY KEY (cut_piece_id, ord)
) STRICT;

-- where every row came from -------------------------------------------------------------------------------------------
CREATE TABLE import_run (                              -- the one place a build's own time is kept
  id INTEGER PRIMARY KEY, at TEXT NOT NULL, tool_commit TEXT, project TEXT NOT NULL
) STRICT;
CREATE TABLE import_file (                             -- a record file a build read: a place, and the bytes it held then
  id INTEGER PRIMARY KEY, location_id INTEGER NOT NULL UNIQUE REFERENCES location(id),
  artifact_id INTEGER NOT NULL REFERENCES artifact(id), source TEXT NOT NULL    -- its name in bench/lane_db/mapping.tsv
) STRICT;
CREATE TABLE import_source (                           -- every imported record, verbatim, so a row traces back to its line
  id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES import_file(id), line INTEGER,
  raw TEXT NOT NULL,                                   -- JSON, read by nothing
  into_table TEXT, into_id INTEGER
) STRICT;
CREATE INDEX import_source_into ON import_source(into_table, into_id);
