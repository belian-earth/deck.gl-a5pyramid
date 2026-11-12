# The vector fixture (tools/fixtures.R) with its fixed-binary vector rewritten as LIST<INT8>,
# tile for tile (one row group per original row group) and with the same
# footer key-value metadata, for the list path (src/fixed-list.ts):
#
#   Rscript tools/fixture-vector-list.R
#
# arrow writes it, as store schema v4 writes its vectors (LIST<INT8>,
# dictionary encoded); snappy, so the tests need no other codec.
suppressMessages(library(arrow))
src <- file.path("test", "fixtures", "vector")
dst <- file.path("test", "fixtures", "vector-list")
set <- yyjsonr::read_json_file(file.path(src, "set.json"), opts = yyjsonr::opts_read_json(arr_of_objs_to_df = FALSE))
for (i in seq_along(set$shards)) {
  sh <- set$shards[[i]]
  pf <- ParquetFileReader$create(file.path(src, sh$path))
  sch <- pf$GetSchema()
  md <- sch$metadata; md[["ARROW:schema"]] <- NULL
  fields <- sch$fields; fields[[match("embedding", names(sch))]] <- field("embedding", list_of(int8()))
  out_schema <- schema(fields)$WithMetadata(md)
  dir.create(dirname(file.path(dst, sh$path)), recursive = TRUE, showWarnings = FALSE)
  sink <- FileOutputStream$create(file.path(dst, sh$path))
  w <- ParquetFileWriter$create(out_schema, sink, properties = ParquetWriterProperties$create(names(out_schema), compression = "snappy"))
  for (g in seq_len(pf$num_row_groups) - 1L) {
    tb <- pf$ReadRowGroup(g)
    raw <- tb$embedding$as_vector()
    lst <- lapply(raw, function(r) as.integer(readBin(r, "integer", size = 1, n = length(r), signed = TRUE)))
    cols <- lapply(names(out_schema), function(nm) if (nm == "embedding") Array$create(lst, type = list_of(int8())) else tb[[nm]])
    names(cols) <- names(out_schema)
    w$WriteTable(Table$create(!!!cols, schema = out_schema), chunk_size = tb$num_rows)
  }
  w$Close(); sink$close()
  p <- file.path(dst, sh$path); fs <- file.size(p)
  con <- file(p, "rb"); seek(con, fs - 8); flen <- readBin(con, "integer", 1, 4, endian = "little"); close(con)
  set$shards[[i]]$bytes <- fs; set$shards[[i]]$footer_bytes <- flen + 8
  cat(sh$path, pf$num_row_groups, "row groups ->", fs, "bytes\n")
}
yyjsonr::write_json_file(set, file.path(dst, "set.json"), auto_unbox = TRUE, digits = NA)
