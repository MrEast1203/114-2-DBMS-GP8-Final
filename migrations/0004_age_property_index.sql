-- ResearchDB Phase 1 · 0004_age_property_index
--
-- AGE 1.6.0 ships labels without any index on the `properties` agtype
-- column. Cypher `MATCH (n:Paper {pid: X})` compiles to a `properties
-- @> '{"pid": X}'::agtype` filter, and without an index the plan is a
-- Seq Scan over all vertices.
--
-- §D10/§9 limitations Route C: building a GIN index on the agtype
-- properties (using the built-in `gin_agtype_ops`) makes containment
-- queries Bitmap Index Scans. On the 50K corpus this cuts a single
-- MATCH-by-pid from 4.66 ms → 0.22 ms (21×) and brings the full
-- ingest/load_graph.py edge projection from "90 min unfinished" to
-- 84 s end-to-end (~65× wall-clock).
--
-- We also add btree indexes on the CITES edge's start_id and end_id
-- columns — these mirror what AGE 1.7 (#2117) auto-creates on label
-- creation. They do not by themselves rescue per-query BFS (AGE 1.6's
-- variable-length edge function `age_vle` blocks PG planner pushdown
-- regardless), but they are required for any AGE-side traversal at
-- depth ≥ 2 to even consider an index path.
--
-- Idempotent: all CREATE INDEX use IF NOT EXISTS.

LOAD 'age';

CREATE INDEX IF NOT EXISTS paper_props_gin
    ON citations_g."Paper" USING gin (properties);

CREATE INDEX IF NOT EXISTS cites_start_idx
    ON citations_g."CITES" (start_id);

CREATE INDEX IF NOT EXISTS cites_end_idx
    ON citations_g."CITES" (end_id);
