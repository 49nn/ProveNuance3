-- Migracja: pola potrzebne dla ontologii generowanej przez LLM jako jedynego źródła prawdy.

ALTER TABLE entity_types
    ADD COLUMN IF NOT EXISTS source_span_text TEXT;

ALTER TABLE predicate_definitions
    ADD COLUMN IF NOT EXISTS source_span_text TEXT;

ALTER TABLE cluster_definitions
    ADD COLUMN IF NOT EXISTS entity_role_name TEXT,
    ADD COLUMN IF NOT EXISTS value_role_name TEXT,
    ADD COLUMN IF NOT EXISTS source_span_text TEXT;

UPDATE cluster_definitions cd
SET entity_role_name = et.name
FROM entity_types et
WHERE cd.entity_type_id = et.id
  AND cd.entity_role_name IS NULL;

UPDATE cluster_definitions
SET value_role_name = CASE
    WHEN name LIKE '%\_type' ESCAPE '\' THEN 'TYPE'
    WHEN name = 'payment_method' THEN 'METHOD'
    ELSE 'VALUE'
END
WHERE value_role_name IS NULL;

ALTER TABLE cluster_definitions
    ALTER COLUMN entity_role_name SET NOT NULL,
    ALTER COLUMN value_role_name SET NOT NULL;

ALTER TABLE rules
    ADD COLUMN IF NOT EXISTS source_span_text TEXT;
