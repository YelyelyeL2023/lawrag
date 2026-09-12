-- lawrag schema.
--
-- Two ideas drive the design:
--
--   1. Bitemporality. Every text row carries the range over which the norm was
--      in force (valid_from/valid_to) *and* the corpus snapshot it came from
--      (release_id). That is what lets the system answer "what was the rule on
--      2025-03-14?" and, six months later, reconstruct exactly what it saw when
--      it answered a given query.
--
--   2. Structure is data. The article tree, the citation graph and the glossary
--      are first-class tables, not something re-derived from text at query
--      time. Hydration is then a join, not an LLM guess.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- btree_gist lets the exclusion constraints below mix equality on text columns
-- with range overlap on dates.
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Open-ended validity uses this sentinel rather than NULL so BETWEEN works and
-- the index stays usable.
-- 9999-12-31

-- ---------------------------------------------------------------- releases --
CREATE TABLE IF NOT EXISTS corpus_release (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    manifest_sha256 TEXT NOT NULL,
    note            TEXT NOT NULL DEFAULT '',
    imported_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------- acts --
CREATE TABLE IF NOT EXISTS act (
    code        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    official_id TEXT NOT NULL DEFAULT '',
    norm_rank   SMALLINT NOT NULL,
    title_ru    TEXT NOT NULL DEFAULT '',
    title_kk    TEXT NOT NULL DEFAULT '',
    adopted_on  DATE,
    url         TEXT
);

-- ------------------------------------------------------------------ nodes --
-- Language-independent: the structural skeleton is shared by the Russian and
-- Kazakh texts of the same act, which is what makes bilingual alignment a
-- primary-key lookup rather than a similarity match.
CREATE TABLE IF NOT EXISTS node (
    id         TEXT PRIMARY KEY,
    act_code   TEXT NOT NULL REFERENCES act(code) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    number     TEXT,
    parent_id  TEXT REFERENCES node(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS node_parent_idx ON node (parent_id);
CREATE INDEX IF NOT EXISTS node_act_idx    ON node (act_code, kind);

CREATE TABLE IF NOT EXISTS node_text (
    node_id    TEXT NOT NULL REFERENCES node(id) ON DELETE CASCADE,
    lang       TEXT NOT NULL,
    valid_from DATE NOT NULL,
    valid_to   DATE NOT NULL DEFAULT DATE '9999-12-31',
    heading    TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    sha256     TEXT NOT NULL,
    release_id TEXT REFERENCES corpus_release(id),
    PRIMARY KEY (node_id, lang, valid_from)
);
CREATE INDEX IF NOT EXISTS node_text_validity_idx
    ON node_text (node_id, lang, valid_from, valid_to);

-- Two redactions of the same norm may never both be in force. Enforced by the
-- database rather than by ingest discipline: the ingest code closes superseded
-- rows correctly, but "correctly" depended on snapshot import order, and a
-- silently overlapping range means two different answers are both "current
-- law". Failing the import is the right outcome.
DO $$ BEGIN
    ALTER TABLE node_text ADD CONSTRAINT node_text_no_overlap
        EXCLUDE USING gist (
            node_id WITH =, lang WITH =,
            daterange(valid_from, valid_to, '[]') WITH &&
        );
EXCEPTION WHEN duplicate_table OR duplicate_object THEN NULL;
END $$;

-- ----------------------------------------------------------------- chunks --
CREATE TABLE IF NOT EXISTS chunk (
    id               TEXT PRIMARY KEY,
    node_id          TEXT NOT NULL,
    article_node_id  TEXT NOT NULL,
    act_code         TEXT NOT NULL REFERENCES act(code) ON DELETE CASCADE,
    lang             TEXT NOT NULL,
    valid_from       DATE NOT NULL,
    valid_to         DATE NOT NULL DEFAULT DATE '9999-12-31',
    heading          TEXT NOT NULL DEFAULT '',
    hierarchy_path   TEXT NOT NULL DEFAULT '',
    context_sentence TEXT NOT NULL DEFAULT '',
    -- The article's opening disposition, kept as its own column so that jobs
    -- which rebuild embed_text (the contextual-retrieval batch) cannot silently
    -- drop it.
    lead_in          TEXT NOT NULL DEFAULT '',
    -- Repeal stubs stay indexed — users ask about repealed articles — but are
    -- flagged so the packet can say so out loud.
    repealed         BOOLEAN NOT NULL DEFAULT false,
    display_text     TEXT NOT NULL,
    embed_text       TEXT NOT NULL,
    token_estimate   INTEGER NOT NULL DEFAULT 0,
    aligned_chunk_id TEXT,
    release_id       TEXT REFERENCES corpus_release(id),
    -- Russian gets a real stemmer. Kazakh has no Postgres dictionary, so it
    -- falls back to 'simple' plus the trigram index below; pretending otherwise
    -- would silently degrade half the corpus.
    tsv              tsvector GENERATED ALWAYS AS (
                        to_tsvector(
                            CASE WHEN lang = 'ru' THEN 'russian'::regconfig
                                 ELSE 'simple'::regconfig END,
                            embed_text)
                     ) STORED
);
CREATE INDEX IF NOT EXISTS chunk_tsv_idx        ON chunk USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunk_trgm_idx       ON chunk USING gin (embed_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS chunk_validity_idx   ON chunk (lang, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS chunk_article_idx    ON chunk (article_node_id, lang);
CREATE INDEX IF NOT EXISTS chunk_act_idx        ON chunk (act_code, lang);
CREATE INDEX IF NOT EXISTS chunk_node_idx       ON chunk (node_id, lang, valid_from);

DO $$ BEGIN
    ALTER TABLE chunk ADD CONSTRAINT chunk_no_overlap
        EXCLUDE USING gist (
            node_id WITH =, lang WITH =,
            daterange(valid_from, valid_to, '[]') WITH &&
        );
EXCEPTION WHEN duplicate_table OR duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS chunk_embedding (
    chunk_id  TEXT PRIMARY KEY REFERENCES chunk(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    embedding vector(1024) NOT NULL
);
-- HNSW is built after the first bulk load; see index/indexer.py.
-- CREATE INDEX chunk_embedding_hnsw ON chunk_embedding
--     USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- --------------------------------------------------------- citation graph --
-- The graph is versioned too. An article's references change when it is
-- amended, and hydrating a 2024 question through the 2026 citation graph pulls
-- in norms the 2024 text never mentioned.
CREATE TABLE IF NOT EXISTS xref (
    id            BIGSERIAL PRIMARY KEY,
    src_node_id   TEXT NOT NULL,
    lang          TEXT NOT NULL,
    raw           TEXT NOT NULL,
    dst_act_code  TEXT,
    dst_article   TEXT,
    dst_node_id   TEXT,
    kind          TEXT NOT NULL DEFAULT 'reference',
    valid_from    DATE NOT NULL DEFAULT DATE '1900-01-01',
    valid_to      DATE NOT NULL DEFAULT DATE '9999-12-31',
    release_id    TEXT REFERENCES corpus_release(id)
);
CREATE INDEX IF NOT EXISTS xref_src_idx ON xref (src_node_id, lang, valid_from, valid_to);
CREATE INDEX IF NOT EXISTS xref_dst_idx ON xref (dst_node_id);
-- Unresolved references are kept on purpose; this view is the corpus-health
-- signal that tells you which acts are still missing from the snapshot.
CREATE OR REPLACE VIEW xref_unresolved AS
    SELECT dst_act_code, count(*) AS n
      FROM xref WHERE dst_node_id IS NULL
     GROUP BY 1 ORDER BY 2 DESC;

-- ---------------------------------------------------------------- glossary --
CREATE TABLE IF NOT EXISTS term (
    id         BIGSERIAL PRIMARY KEY,
    term       TEXT NOT NULL,
    lang       TEXT NOT NULL,
    act_code   TEXT NOT NULL REFERENCES act(code) ON DELETE CASCADE,
    node_id    TEXT NOT NULL,
    definition TEXT NOT NULL,
    valid_from DATE NOT NULL,
    valid_to   DATE NOT NULL DEFAULT DATE '9999-12-31',
    release_id TEXT REFERENCES corpus_release(id)
);
CREATE INDEX IF NOT EXISTS term_lookup_idx ON term (lang, term);
CREATE INDEX IF NOT EXISTS term_trgm_idx   ON term USING gin (term gin_trgm_ops);

-- -------------------------------------------------------------- audit log --
-- Every answer is reconstructible: the question, the snapshot in force, what
-- was retrieved, what was cited, and whether verification passed.
CREATE TABLE IF NOT EXISTS query_log (
    id            BIGSERIAL PRIMARY KEY,
    asked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question      TEXT NOT NULL,
    lang          TEXT NOT NULL,
    as_of         DATE NOT NULL,
    release_id    TEXT,
    retrieved     JSONB NOT NULL DEFAULT '[]'::jsonb,
    cited         JSONB NOT NULL DEFAULT '[]'::jsonb,
    verdict       TEXT NOT NULL DEFAULT '',
    abstained     BOOLEAN NOT NULL DEFAULT false,
    answer        TEXT NOT NULL DEFAULT '',
    latency_ms    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS query_log_time_idx ON query_log (asked_at DESC);
