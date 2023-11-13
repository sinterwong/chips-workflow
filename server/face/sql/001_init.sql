CREATE TABLE AppUser (
  id INTEGER PRIMARY KEY,
  id_number VARCHAR UNIQUE,
  feature_base64 VARCHAR
);

-- Path: server/face/sql/002_insert.sql
-- INSERT INTO
--   AppUser (id_number, feature_base64)
-- VALUES
--   (
--     '1427011xxxxxxxxxxx',
--     'data:image/jpeg;base64,XXXX'
--   );