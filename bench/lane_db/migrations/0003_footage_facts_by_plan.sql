-- The lane's record, third migration: a footage fact said in a plan file is that plan's word, and the latest stands.
--
-- Head marks, voice spans, facial hair, not-people, aliases and the film's words are facts about the footage, and
-- today each project's plan file says them. The first two migrations kept the first project's word and only noted
-- where a later plan said otherwise. Now every plan's word is kept, under the project whose plan it is, and the
-- `*_standing` views give the word of the last project on the video that has a plan. A project with no plan yet
-- reads the standing word, so footage facts still carry; a later plan that adds a head mark supersedes, and the
-- earlier plan's rows stay as what stood before. The same fact said by two plans is two rows: what each file says.
-- "Last" is the order of data/projects.json, which is the order projects are numbered in.

CREATE TABLE head_mark_new (
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  ord INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL,
  mark TEXT NOT NULL CHECK (mark IN ('hidden', 'hers', 'not hers')), said_by TEXT NOT NULL
) STRICT;
INSERT INTO head_mark_new (id, project_id, subject_id, ord, first, last, mark, said_by)
  SELECT h.id, (SELECT min(p.id) FROM project p JOIN subject s ON s.video_id = p.video_id WHERE s.id = h.subject_id),
         h.subject_id, h.ord, h.first, h.last, h.mark, h.said_by FROM head_mark h;
DROP TABLE head_mark;
ALTER TABLE head_mark_new RENAME TO head_mark;

CREATE TABLE voice_span_new (
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  ord INTEGER NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL, voice TEXT NOT NULL, said_by TEXT NOT NULL
) STRICT;
INSERT INTO voice_span_new (id, project_id, subject_id, ord, first, last, voice, said_by)
  SELECT v.id, (SELECT min(p.id) FROM project p JOIN subject s ON s.video_id = p.video_id WHERE s.id = v.subject_id),
         v.subject_id, v.ord, v.first, v.last, v.voice, v.said_by FROM voice_span v;
DROP TABLE voice_span;
ALTER TABLE voice_span_new RENAME TO voice_span;

CREATE TABLE facial_hair_new (
  project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  has INTEGER NOT NULL CHECK (has IN (0, 1)), said_by TEXT NOT NULL, PRIMARY KEY (project_id, subject_id)
) STRICT;
INSERT INTO facial_hair_new (project_id, subject_id, has, said_by)
  SELECT (SELECT min(p.id) FROM project p JOIN subject s ON s.video_id = p.video_id WHERE s.id = f.subject_id), f.subject_id, f.has, f.said_by
  FROM facial_hair f;
DROP TABLE facial_hair;
ALTER TABLE facial_hair_new RENAME TO facial_hair;

CREATE TABLE alias_new (
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), subject_id INTEGER NOT NULL REFERENCES subject(id),
  alias TEXT NOT NULL, ord INTEGER NOT NULL, said_by TEXT, UNIQUE (project_id, subject_id, alias)
) STRICT;
INSERT INTO alias_new (id, project_id, subject_id, alias, ord, said_by)
  SELECT a.id, (SELECT min(p.id) FROM project p JOIN subject s ON s.video_id = p.video_id WHERE s.id = a.subject_id),
         a.subject_id, a.alias, a.ord, a.said_by FROM alias a;
DROP TABLE alias;
ALTER TABLE alias_new RENAME TO alias;

CREATE TABLE not_person_new (
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES project(id), video_id INTEGER NOT NULL REFERENCES video(id),
  ord INTEGER NOT NULL, what TEXT NOT NULL, counts_as_no_one TEXT, said_by TEXT NOT NULL
) STRICT;
INSERT INTO not_person_new (id, project_id, video_id, ord, what, counts_as_no_one, said_by)
  SELECT n.id, (SELECT min(p.id) FROM project p WHERE p.video_id = n.video_id), n.video_id, n.ord, n.what, n.counts_as_no_one, n.said_by
  FROM not_person n;
DROP TABLE not_person;
ALTER TABLE not_person_new RENAME TO not_person;

ALTER TABLE project ADD COLUMN film_words TEXT;        -- plan text.film, as this project's plan says it
UPDATE project SET film_words = (SELECT v.film_words FROM video v WHERE v.id = project.video_id)
  WHERE id = (SELECT min(p.id) FROM project p WHERE p.video_id = project.video_id);
ALTER TABLE video DROP COLUMN film_words;

-- A file built before this migration held one word per video, the first project's, and it stood for every project.
-- To keep that meaning, each later project that has a plan is given the first project's word as its own. A fresh
-- build reads each plan file instead.
CREATE TEMP TABLE later_plan AS
  SELECT p.id AS project_id, p.video_id, (SELECT min(q.id) FROM project q WHERE q.video_id = p.video_id) AS first_id
  FROM project p WHERE EXISTS (SELECT 1 FROM subject_plan sp WHERE sp.project_id = p.id)
    AND p.id <> (SELECT min(q.id) FROM project q WHERE q.video_id = p.video_id);
INSERT INTO head_mark (project_id, subject_id, ord, first, last, mark, said_by)
  SELECT l.project_id, h.subject_id, h.ord, h.first, h.last, h.mark, h.said_by FROM head_mark h JOIN later_plan l ON l.first_id = h.project_id;
INSERT INTO voice_span (project_id, subject_id, ord, first, last, voice, said_by)
  SELECT l.project_id, v.subject_id, v.ord, v.first, v.last, v.voice, v.said_by FROM voice_span v JOIN later_plan l ON l.first_id = v.project_id;
INSERT INTO facial_hair (project_id, subject_id, has, said_by)
  SELECT l.project_id, f.subject_id, f.has, f.said_by FROM facial_hair f JOIN later_plan l ON l.first_id = f.project_id;
INSERT INTO alias (project_id, subject_id, alias, ord, said_by)
  SELECT l.project_id, a.subject_id, a.alias, a.ord, a.said_by FROM alias a JOIN later_plan l ON l.first_id = a.project_id;
INSERT INTO not_person (project_id, video_id, ord, what, counts_as_no_one, said_by)
  SELECT l.project_id, n.video_id, n.ord, n.what, n.counts_as_no_one, n.said_by FROM not_person n JOIN later_plan l ON l.first_id = n.project_id;
INSERT INTO not_person_span (not_person_id, ord, first, last)
  SELECT mine.id, s.ord, s.first, s.last
  FROM not_person_span s JOIN not_person theirs ON theirs.id = s.not_person_id JOIN later_plan l ON l.first_id = theirs.project_id
  JOIN not_person mine ON mine.project_id = l.project_id AND mine.ord = theirs.ord;
UPDATE project SET film_words = (SELECT q.film_words FROM project q JOIN later_plan l ON l.first_id = q.id WHERE l.project_id = project.id)
  WHERE id IN (SELECT project_id FROM later_plan);
DROP TABLE later_plan;

CREATE VIEW plan_standing AS                           -- per video, the project whose plan is the latest word on its footage facts
  SELECT v.id AS video_id,
         (SELECT max(p.id) FROM project p WHERE p.video_id = v.id AND EXISTS (SELECT 1 FROM subject_plan sp WHERE sp.project_id = p.id)) AS project_id
  FROM video v;
CREATE VIEW head_mark_standing AS
  SELECT h.* FROM head_mark h JOIN subject s ON s.id = h.subject_id JOIN plan_standing ps ON ps.video_id = s.video_id AND ps.project_id = h.project_id;
CREATE VIEW voice_span_standing AS
  SELECT v.* FROM voice_span v JOIN subject s ON s.id = v.subject_id JOIN plan_standing ps ON ps.video_id = s.video_id AND ps.project_id = v.project_id;
CREATE VIEW facial_hair_standing AS
  SELECT f.* FROM facial_hair f JOIN subject s ON s.id = f.subject_id JOIN plan_standing ps ON ps.video_id = s.video_id AND ps.project_id = f.project_id;
CREATE VIEW alias_standing AS
  SELECT a.* FROM alias a JOIN subject s ON s.id = a.subject_id JOIN plan_standing ps ON ps.video_id = s.video_id AND ps.project_id = a.project_id;
CREATE VIEW not_person_standing AS
  SELECT n.* FROM not_person n JOIN plan_standing ps ON ps.video_id = n.video_id AND ps.project_id = n.project_id;
CREATE VIEW film_words_standing AS
  SELECT ps.video_id, p.film_words FROM plan_standing ps JOIN project p ON p.id = ps.project_id;
