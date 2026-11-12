# A store v4 set in miniature (belian-production docs/store-v4.md), written
# by a5pyramid (a5pyramid#6, format version 4), for the renderer's reading
# of shard tables, partial rows and single-level tiers:
#
#   Rscript tools/fixture-store-v4.R
#
# The leaves and codes are the vector fixture's (tools/fixtures.R), with
# pixel counts. The "partitions" are the leaves' supercells at resolution 13
# (three of them): each partition's levels 15 to 17 are built from its own
# leaves, so a cell split by a partition edge has a partial row in each,
# with `owner`; its leaves (18) are a file of their own, without lod or
# ncell. Levels 12 to 14 are keyed by resolution-11 cells and 9 to 11 by
# resolution-0 cells, built whole from every leaf. Every file holds one
# level, a tier of its own; vectors are lists; tiles of 16 rows in row
# groups of up to 64. set.parquet lists the files, one empty partition and
# the writer's own columns (year, build); there is no set.json. Snappy, so
# the tests need no other codec.

suppressMessages({
  library(a5R)
  library(a5pyramid)
})

centre <- a5_lonlat_to_cell(117.8, 4.96, 18)
leaves <- a5_uncompact(a5_spherical_cap(centre, 400))
ll <- a5_cell_to_lonlat(leaves, as_dataframe = TRUE)
u <- (ll$lon - mean(ll$lon)) / max(abs(ll$lon - mean(ll$lon)))
v <- (ll$lat - mean(ll$lat)) / max(abs(ll$lat - mean(ll$lat)))
code <- function(x) as.integer(pmax(-127, pmin(127, round(x))))
k <- 0:63
x <- data.frame(cell = leaves)
x$mean_embedding <- t(vapply(seq_along(leaves), function(i) code(90 * sin(u[i] * (k + 1) / 9 + v[i] * (k %% 7) / 3)), integer(64)))
x$n_pixels <- 3L + seq_along(leaves) %% 5L

out <- file.path("test", "fixtures", "store-v4")
unlink(out, recursive = TRUE)
dir.create(out, recursive = TRUE)
build <- function(rows, lods) {
  a5_pyramid(rows, mean_embedding, lods = lods, weights = "n_pixels", weights_type = "uint32",
             reducers = c(mean_embedding = "mean"), codecs = c(mean_embedding = "aef"))
}
tiers <- data.frame(lod_min = 9:18, lod_max = 9:18,
                    shard_resolution = c(0, 0, 0, 11, 11, 11, 13, 13, 13, 13),
                    tile_size = 16L, row_group_size = 64L)
tier_of <- function(lod) match(lod, tiers$lod_min) - 1L
files <- list()
write <- function(rows, lod, key, res, owner) {
  path <- sprintf("partition=R%d_%s/resolution=R_%d/embeddings.parquet", res, a5_u64_to_hex(key), lod)
  dir.create(dirname(file.path(out, path)), recursive = TRUE, showWarnings = FALSE)
  a5_write_pyramid_level(
    rows, file.path(out, path), mean_embedding, data_resolution = 18,
    ncell = if (lod < 18) "ncell", weights = "n_pixels",
    reducers = c(mean_embedding = "mean"), codecs = c(mean_embedding = "aef"), storage = c(mean_embedding = "list"),
    shard = list(tier = tier_of(lod), cell = key, resolution = res), owner = owner,
    tile_size = 16, row_group_size = 64, compression = "snappy",
    metadata = c(a5pyramids = '{"region":"fixture","vector":"mean_embedding","colour":"pca"}')
  )
  files[[length(files) + 1L]] <<- list(path = path, tier = tier_of(lod), cell = key)
}

# partitions: levels 15-17 from each partition's own leaves, and its leaves
part <- a5_cell_to_supercell(x$cell, 13)
for (p in unique(a5_u64_to_hex(part))) {
  mine <- x[a5_u64_to_hex(part) == p, ]
  key <- a5_cell(p)
  lv <- build(mine, 15:17)
  for (lod in 15:17) write(lv[lv$lod == lod, ], lod, key, 13, owner = TRUE)
  write(mine, 18, key, 13, owner = FALSE)
}
# coarse levels, whole, each cell in its key
whole <- build(x, 9:14)
for (lod in 9:14) {
  res <- if (lod >= 12) 11 else 0
  rows <- whole[whole$lod == lod, ]
  key <- a5_cell_to_supercell(rows$cell, res)
  for (kk in unique(a5_u64_to_hex(key))) write(rows[a5_u64_to_hex(key) == kk, ], lod, a5_cell(kk), res, owner = FALSE)
}

shards <- do.call(rbind, lapply(files, function(f) {
  info <- a5_pyramid_info(file.path(out, f$path))
  data.frame(tier = f$tier, path = f$path, rows = sum(info$tiles$rows), tiles = nrow(info$tiles), bytes = info$bytes,
             footer_bytes = info$footer_bytes, xmin = info$bbox[["xmin"]], ymin = info$bbox[["ymin"]],
             xmax = info$bbox[["xmax"]], ymax = info$bbox[["ymax"]])
}))
shards$cell <- do.call(c, lapply(files, `[[`, "cell"))
# a partition built empty: listed, with no file
empty <- shards[shards$tier == tier_of(18), ][1, ]
empty$cell <- a5_cell_to_supercell(a5_lonlat_to_cell(117.9, 5.2, 18), 13)
empty$path <- NA_character_
empty[c("rows", "tiles", "bytes", "footer_bytes")] <- 0
empty[c("xmin", "ymin", "xmax", "ymax")] <- NA_real_
shards <- rbind(shards, empty)
shards <- shards[order(shards$tier, a5_u64_to_hex(shards$cell)), ]
shards$year <- 2021L
shards$build <- "fixture"
a5_write_pyramid_set_table(shards, file.path(out, "set.parquet"), template = file.path(out, files[[1]]$path),
                           tiers = tiers, metadata = c(a5pyramids = '{"region":"fixture","vector":"mean_embedding","colour":"pca"}'))
print(a5_pyramid_set_info(out)$tiers[, c("tier", "lod_min", "shard_resolution", "shards", "rows", "bytes")])
