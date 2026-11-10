-- The lane's record, second migration: more than one project, and more than one video, in one file.
--
-- The first tier held a line's number as unique across the whole database (a seed's, an answer's, a laid line's). A
-- second project has its own laid list and, once the owner reviews its renders, its own answers file; a second video
-- has its own cast file. A line is now a line OF A FILE: each of the three tables names the file it was read from
-- (import_file), and a line number is unique within it. SQLite cannot change a column's UNIQUE in place, so each table
-- is made again and its rows copied (db.connect runs migrations with foreign keys off, and checks them after).
--
-- And the pieces a project keeps from another project on the same footage: fun2 remakes only the lead, and lays
-- fun's second and third subjects as they stand.

CREATE TABLE seed_new (                                    -- footage: cast.jsonl, a box naming who is who on a frame
  id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL REFERENCES subject(id), file_id INTEGER NOT NULL REFERENCES import_file(id), line INTEGER NOT NULL,
  frame INTEGER NOT NULL, x REAL, y REAL, w REAL, h REAL, picture_w INTEGER, picture_h INTEGER,
  answer TEXT NOT NULL CHECK (answer IN ('here', 'none of the outlined people', 'withdrawn')),
  encloses TEXT CHECK (encloses IN ('head', 'body')), drawn TEXT,
  from_table TEXT NOT NULL, said_by TEXT NOT NULL, at TEXT NOT NULL,
  UNIQUE (file_id, line),
  CHECK ((x IS NULL) = (y IS NULL) AND (x IS NULL) = (w IS NULL) AND (x IS NULL) = (h IS NULL))
) STRICT;
INSERT INTO seed_new (id, subject_id, line, frame, x, y, w, h, picture_w, picture_h, answer, encloses, drawn, from_table, said_by, at, file_id)
  SELECT id, subject_id, line, frame, x, y, w, h, picture_w, picture_h, answer, encloses, drawn, from_table, said_by, at, (SELECT id FROM import_file WHERE source = 'cast') FROM seed;
DROP TABLE seed;
ALTER TABLE seed_new RENAME TO seed;

CREATE TABLE answer_new (                                  -- one line of owner_answers.jsonl
  id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES video(id),
  project_id INTEGER REFERENCES project(id),           -- set when the answer is about a replacement; NULL for the footage
  about TEXT NOT NULL CHECK (about IN ('footage', 'render', 'plan', 'not an answer', 'not classed')),
  file_id INTEGER NOT NULL REFERENCES import_file(id), -- the answers file it is a line of
  line INTEGER NOT NULL,                               -- its line in that file, as records cite it (owner_answers.jsonl:N)
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
  UNIQUE (file_id, line),
  CHECK (person_tile IS NULL OR person_label IS NULL)
) STRICT;
INSERT INTO answer_new (id, video_id, project_id, about, line, at, who, what, page_row, tile, shot_said, subject_id, frame, source_frame, source_frame_default, chosen, words, person_tile, person_label, table_name, table_frame, clip_location_id, clip_artifact_id, clip_not_known, render_location_id, render_artifact_id, said_in, taken_down_by, relayed_by, relayed_from, corrects, file_id)
  SELECT id, video_id, project_id, about, line, at, who, what, page_row, tile, shot_said, subject_id, frame, source_frame, source_frame_default, chosen, words, person_tile, person_label, table_name, table_frame, clip_location_id, clip_artifact_id, clip_not_known, render_location_id, render_artifact_id, said_in, taken_down_by, relayed_by, relayed_from, corrects, (SELECT id FROM import_file WHERE source = 'answers') FROM answer;
DROP TABLE answer;
ALTER TABLE answer_new RENAME TO answer;

DROP VIEW laid_standing;
CREATE TABLE laid_new (                                    -- project: a render laid on frames for a subject
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id),
  subject_id INTEGER NOT NULL REFERENCES subject(id), load_id INTEGER NOT NULL REFERENCES load(id),
  first INTEGER NOT NULL, last INTEGER NOT NULL,       -- the shot it is laid on
  render_location_id INTEGER NOT NULL REFERENCES location(id), render_artifact_id INTEGER NOT NULL REFERENCES artifact(id),
  zoomed_artifact_id INTEGER REFERENCES artifact(id),
  added_at TEXT NOT NULL, added_by TEXT NOT NULL,
  no_rebuild_why TEXT, partial_lay_why TEXT, seeded_from TEXT,
  owner_before_correction TEXT, owner_corrected_why TEXT,    -- laid owner_was / owner_corrected: a status corrected with no history entry
  file_id INTEGER NOT NULL REFERENCES import_file(id), -- the list it is a line of: a laid list, or a removed list
  list_line INTEGER,                                   -- its line in the laid list as it stands; NULL once removed
  removed_line INTEGER,                                -- its line in the removed list
  UNIQUE (file_id, list_line), UNIQUE (file_id, removed_line)
) STRICT;
INSERT INTO laid_new (id, project_id, subject_id, load_id, first, last, render_location_id, render_artifact_id, zoomed_artifact_id, added_at, added_by, no_rebuild_why, partial_lay_why, seeded_from, owner_before_correction, owner_corrected_why, list_line, removed_line, file_id)
  SELECT id, project_id, subject_id, load_id, first, last, render_location_id, render_artifact_id, zoomed_artifact_id, added_at, added_by, no_rebuild_why, partial_lay_why, seeded_from, owner_before_correction, owner_corrected_why, list_line, removed_line, (SELECT id FROM import_file WHERE source = CASE WHEN list_line IS NOT NULL THEN 'laid' ELSE 'laid-removed' END) FROM laid;
DROP TABLE laid;
ALTER TABLE laid_new RENAME TO laid;

CREATE VIEW laid_standing AS
  SELECT l.*, s.owner_status, s.why, s.word_file, s.word_line, s.answer_id
  FROM laid l JOIN laid_state s ON s.laid_id = l.id
  WHERE l.list_line IS NOT NULL AND s.ord = (SELECT max(ord) FROM laid_state WHERE laid_id = l.id);

CREATE TABLE project_keeps (                           -- project: a subject whose standing pieces of another project on the same
  project_id INTEGER NOT NULL REFERENCES project(id),  -- footage this project lays as they stand, and does not remake
  from_project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  said_by TEXT NOT NULL, PRIMARY KEY (project_id, subject_id), CHECK (project_id <> from_project_id)
) STRICT;
CREATE VIEW project_laid AS                            -- what a project lays: its own standing pieces, and the ones it keeps
  SELECT l.project_id AS for_project_id, 0 AS kept_from_another, l.* FROM laid_standing l
  UNION ALL
  SELECT k.project_id, 1, l.* FROM project_keeps k JOIN laid_standing l ON l.project_id = k.from_project_id AND l.subject_id = k.subject_id;
