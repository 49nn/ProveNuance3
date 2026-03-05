-- Migracja: dodaje kolumnę source_span_text do tabel ontologicznych
-- Additive, bezpieczna — IF NOT EXISTS nie zmienia istniejących wierszy
-- Uruchomić raz: psql -d provenuance -f db/migrate_span_text.sql

ALTER TABLE entity_types          ADD COLUMN IF NOT EXISTS source_span_text TEXT;
ALTER TABLE predicate_definitions ADD COLUMN IF NOT EXISTS source_span_text TEXT;
ALTER TABLE cluster_definitions   ADD COLUMN IF NOT EXISTS source_span_text TEXT;
ALTER TABLE rules                 ADD COLUMN IF NOT EXISTS source_span_text TEXT;
