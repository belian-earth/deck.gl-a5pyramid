# The test fixtures: two small a5pyramid sets in test/fixtures, written by
# a5pyramid itself, so the reader is tested against the writer's output.
#
#   Rscript tools/fixtures.R
#
# Needs a5pyramid with vector columns (format version 3, a5pyramid#4) and
# a5R 1.0. Each set holds the leaves (resolution 18) within 400 m of a point
# near Danum Valley, about 1,000 cells, with tiles of at most 64 rows so
# every level of the finer tiers spans several tiles.
#
#   bands   three int8 columns A, B and C, a band stretch in the metadata
#   vector  one int8 vector column of 64 (as AEF embeddings), colour "pca"
#
# Values are a fixed function of each leaf's position, so the files are the
# same on every run.

suppressMessages({
  library(a5R)
  library(a5pyramid)
  library(arrow)
})

out <- file.path("test", "fixtures")
centre <- a5_lonlat_to_cell(117.8, 4.96, 18)
leaves <- a5_uncompact(a5_spherical_cap(centre, 400))
ll <- a5_cell_to_lonlat(leaves, as_dataframe = TRUE)
# positions within the patch, -1 to 1
u <- (ll$lon - mean(ll$lon)) / max(abs(ll$lon - mean(ll$lon)))
v <- (ll$lat - mean(ll$lat)) / max(abs(ll$lat - mean(ll$lat)))
code <- function(x) as.integer(pmax(-127, pmin(127, round(x))))
order_ <- order(a5_cell_to_arrow(leaves)$as_vector())
cat(length(leaves), "leaves\n")

tiers <- a5_pyramid_set_tiers(18L)
tiers$tile_size <- 64L

meta <- function(x) c(a5pyramids = yyjsonr::write_json_str(x, opts = yyjsonr::opts_write_json(auto_unbox = TRUE)))

# -- bands ---------------------------------------------------------------------
bands <- arrow_table(
  cell = a5_cell_to_arrow(leaves),
  A = Array$create(code(100 * u), int8()),
  B = Array$create(code(100 * v), int8()),
  C = Array$create(code(60 * u * v + 20), int8())
)
bands <- bands[order_, ]
dest <- file.path(out, "bands")
unlink(dest, recursive = TRUE)
a5_write_pyramid_set(
  bands, dest, A, B, C,
  tiers = tiers, aggregate = "rep_child", types = c(A = "int8", B = "int8", C = "int8"),
  metadata = meta(list(region = "fixture", bands = c("A", "B", "C"),
                       stretch = list(A = c(-100, 100), B = c(-100, 100), C = c(-40, 80))))
)
print(a5_pyramid_set_info(dest)$shards[, c("tier", "rows", "tiles", "bytes")])

# -- vector --------------------------------------------------------------------
k <- 0:63
emb <- lapply(seq_along(leaves), function(i) code(90 * sin(u[i] * (k + 1) / 9 + v[i] * (k %% 7) / 3)))
vec <- arrow_table(
  cell = a5_cell_to_arrow(leaves),
  embedding = Array$create(emb, fixed_size_list_of(int8(), 64L))
)
vec <- vec[order_, ]
dest <- file.path(out, "vector")
unlink(dest, recursive = TRUE)
a5_write_pyramid_set(
  vec, dest, embedding,
  tiers = tiers,
  metadata = meta(list(region = "fixture", vector = "embedding", bands = sprintf("A%02d", k),
                       decode = "sign(x) * (x / 127.5)^2", colour = "pca"))
)
print(a5_pyramid_set_info(dest)$shards[, c("tier", "rows", "tiles", "bytes")])
