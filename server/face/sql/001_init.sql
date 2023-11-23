CREATE TABLE AppUser (
  id INTEGER PRIMARY KEY,
  idNumber VARCHAR UNIQUE,
  libName VARCHAR,
  feature VARCHAR
);

-- Path: server/face/sql/002_insert.sql
-- INSERT INTO
--   AppUser (idNumber, feature)
-- VALUES
--   (
--     '1427011xxxxxxxxxxx',
--     'data:image/jpeg;base64,XXXX'
--   );