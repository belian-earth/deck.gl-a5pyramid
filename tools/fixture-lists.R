# test/fixtures/lists/duckdb-v2.parquet: a LIST<TINYINT> column of 64 as
# DuckDB writes it with PARQUET_VERSION V2 (delta-packed, v2 pages): row i,
# element e holds ((i * 37 + e * 11) %% 256) - 128.
#
#   Rscript tools/fixture-lists.R
suppressMessages(library(duckdb))
con <- dbConnect(duckdb())
dir.create(file.path("test", "fixtures", "lists"), showWarnings = FALSE)
dbExecute(con, "COPY (SELECT i::UBIGINT AS cell, list_transform(range(64), e -> ((i * 37 + e * 11) % 256 - 128)::TINYINT) AS v
                        FROM range(1000) t(i) ORDER BY i)
                  TO 'test/fixtures/lists/duckdb-v2.parquet' (FORMAT parquet, PARQUET_VERSION V2, COMPRESSION snappy)")
