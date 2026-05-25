-- ResearchDB Phase 1 · 0002_age_graph_init
--
-- Creates the AGE graph `citations_g` and the Paper / CITES labels.
-- Bulk projection from the relational tables happens in the Python loader
-- (ingest/load_graph.py) using UNWIND batches — that is N× faster than a
-- per-row plpgsql loop and avoids EXECUTE/format quoting hazards.
--
-- Required for the Phase 1 Week 1 AGE smoke test.
-- Runs inside sqlx::migrate!()'s implicit transaction.

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create the graph if not present. AGE 1.6 has no IF NOT EXISTS and
-- raises with several SQLSTATEs across versions, so we swallow all
-- errors here; the only failure mode that matters is the success-state
-- side effect (graph exists).
DO $boot$
BEGIN
    PERFORM create_graph('citations_g');
EXCEPTION WHEN OTHERS THEN NULL;
END $boot$;

-- Create labels if not present. Critically we do NOT drop+recreate
-- here: that would wipe any projected graph data. Re-running this
-- migration must be a no-op. The Python loader has its own
-- reset_age_labels() for when a fresh projection is intended.
DO $labels$
BEGIN
    BEGIN PERFORM create_vlabel('citations_g', 'Paper'); EXCEPTION WHEN OTHERS THEN NULL; END;
    BEGIN PERFORM create_elabel('citations_g', 'CITES'); EXCEPTION WHEN OTHERS THEN NULL; END;
END $labels$;
