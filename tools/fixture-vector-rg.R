# The vector fixture (tools/fixtures.R) in format version 4: row groups of up
# to 64 rows, each cut into tiles of 16 rows that are its data pages, with
# the page index, for the reading of tiles by page (src/tile-core.ts):
#
#   Rscript tools/fixture-vector-rg.R
#
# Needs a5pyramid with row_group_size (a5pyramid#6). The leaves, levels and
# codes are those of the vector fixture; only the tiling differs.

suppressMessages({
  library(a5R)
  library(a5pyramid)
  library(arrow)
})

centre <- a5_lonlat_to_cell(117.8, 4.96, 18)
leaves <- a5_uncompact(a5_spherical_cap(centre, 400))
ll <- a5_cell_to_lonlat(leaves, as_dataframe = TRUE)
u <- (ll$lon - mean(ll$lon)) / max(abs(ll$lon - mean(ll$lon)))
v <- (ll$lat - mean(ll$lat)) / max(abs(ll$lat - mean(ll$lat)))
code <- function(x) as.integer(pmax(-127, pmin(127, round(x))))
order_ <- order(a5_cell_to_arrow(leaves)$as_vector())

k <- 0:63
emb <- lapply(seq_along(leaves), function(i) code(90 * sin(u[i] * (k + 1) / 9 + v[i] * (k %% 7) / 3)))
vec <- arrow_table(
  cell = a5_cell_to_arrow(leaves),
  embedding = Array$create(emb, fixed_size_list_of(int8(), 64L))
)
vec <- vec[order_, ]

tiers <- a5_pyramid_set_tiers(18L)
tiers$tile_size <- 16L
tiers$row_group_size <- 64L
meta <- function(x) c(a5pyramids = yyjsonr::write_json_str(x, opts = yyjsonr::opts_write_json(auto_unbox = TRUE)))
dest <- file.path("test", "fixtures", "vector-rg")
unlink(dest, recursive = TRUE)
a5_write_pyramid_set(
  vec, dest, embedding,
  tiers = tiers,
  metadata = meta(list(region = "fixture", vector = "embedding", bands = sprintf("A%02d", k),
                       decode = "sign(x) * (x / 127.5)^2", colour = "pca"))
)
print(a5_pyramid_set_info(dest)$shards[, c("tier", "rows", "tiles", "bytes", "footer_bytes")])
