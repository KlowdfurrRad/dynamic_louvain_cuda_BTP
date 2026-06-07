# Benchmark Graphs for Louvain Evaluation

| Dataset | Nodes | Edges | Type | Download |
|---------|------:|------:|------|----------|
| com-LiveJournal | 4.0M | 34.7M | Social | [SNAP](https://snap.stanford.edu/data/com-LiveJournal.html) |
| com-Orkut | 3.1M | 117.2M | Social | [SNAP](https://snap.stanford.edu/data/com-Orkut.html) |
| indochina-2004 | 7.4M | 194.1M | Web | [SuiteSparse](https://sparse.tamu.edu/LAW/indochina-2004) |
| asia_osm | 12.0M | 12.7M | Road | [SuiteSparse](https://sparse.tamu.edu/DIMACS10/asia_osm) |
| uk-2002 | 18.5M | 298.1M | Web | [SuiteSparse](https://sparse.tamu.edu/LAW/uk-2002) |
| arabic-2005 | 22.7M | 640.0M | Web | [SuiteSparse](https://sparse.tamu.edu/LAW/arabic-2005) |
| uk-2005 | 39.5M | 936.4M | Web | [SuiteSparse](https://sparse.tamu.edu/LAW/uk-2005) |
| it-2004 | 41.3M | 1.15B | Web | [SuiteSparse](https://sparse.tamu.edu/LAW/it-2004) |
| europe_osm | 50.9M | 108.1M | Road | [SuiteSparse](https://sparse.tamu.edu/DIMACS10/europe_osm) |
| sk-2005 | 50.6M | 1.95B | Web | [SuiteSparse](https://sparse.tamu.edu/LAW/sk-2005) |
| webbase-2001 | 118.1M | 1.02B | Web | [SuiteSparse](https://sparse.tamu.edu/LAW/webbase-2001) |
| kmer_A2a | 170.7M | 360.6M | K-mer | [SuiteSparse](https://sparse.tamu.edu/GenBank/kmer_A2a) |
| kmer_V1r | 214.0M | 465.4M | K-mer | [SuiteSparse](https://sparse.tamu.edu/GenBank/kmer_V1r) |

# Real Graph Datasets for Louvain Benchmarking
References [SNAP](https://snap.stanford.edu/data/index.html)

## Classic Small Benchmarks

| Dataset | Nodes | Edges | Type | Link |
|---------|-------|-------|------|------|
| Zachary's Karate Club | 34 | 78 | Social | [KONECT](http://konect.cc/networks/ucidata-zachary/) |
| Dolphins | 62 | 159 | Animal social | [Newman](http://www-personal.umich.edu/~mejn/netdata/dolphins.zip) |
| Les Miserables | 77 | 254 | Co-appearance | [Newman](http://www-personal.umich.edu/~mejn/netdata/lesmis.zip) |
| College Football | 115 | 613 | Sports | [Newman](http://www-personal.umich.edu/~mejn/netdata/football.zip) |
| Jazz Musicians | 198 | 2,742 | Collaboration | [KONECT](http://konect.cc/networks/arenas-jazz/) |

## SNAP Datasets (Small-Medium)

| Dataset | Nodes | Edges | Type | Link |
|---------|-------|-------|------|------|
| ego-Facebook | 4,039 | 88,234 | Social | [SNAP](https://snap.stanford.edu/data/ego-Facebook.html) |
| CA-GrQc | 5,242 | 14,496 | Collaboration | [SNAP](https://snap.stanford.edu/data/ca-GrQc.html) |
| CA-HepTh | 9,877 | 25,998 | Collaboration | [SNAP](https://snap.stanford.edu/data/ca-HepTh.html) |
| CA-HepPh | 12,008 | 118,521 | Collaboration | [SNAP](https://snap.stanford.edu/data/ca-HepPh.html) |
| CA-AstroPh | 18,772 | 198,110 | Collaboration | [SNAP](https://snap.stanford.edu/data/ca-AstroPh.html) |
| Email-Enron | 36,692 | 183,831 | Communication | [SNAP](https://snap.stanford.edu/data/email-Enron.html) |

## SNAP Datasets (Medium-Large)

The `com-*` graphs come with ground-truth communities; `web-Google` is a web graph.

| Dataset | Nodes | Edges | Type | Link |
|---------|-------|-------|------|------|
| com-DBLP | 317,080 | 1,049,866 | Collaboration | [SNAP](https://snap.stanford.edu/data/com-DBLP.html) |
| com-Amazon | 334,863 | 925,872 | Co-purchasing | [SNAP](https://snap.stanford.edu/data/com-Amazon.html) |
| com-Youtube | 1,134,890 | 2,987,624 | Social | [SNAP](https://snap.stanford.edu/data/com-Youtube.html) |
| web-Google | 875,713 | 5,105,039 | Web | [SNAP](https://snap.stanford.edu/data/web-Google.html) |

## SNAP Temporal Networks

Edges are timestamped (`SRC DST UNIXTS`) and the same pair can recur over time.
**Temporal Edges** is the full timestamped stream; **Static Edges** is the unique
deduplicated graph. For the dynamic Louvain benchmarks the stream is split
chronologically into an initial graph (earliest 80 %) plus insertion batches (see
`real_graphs/snap_temporal/` and its `convert_snap_temporal_to_dynamic.py`).

| Dataset | Nodes | Temporal Edges | Static Edges | Link |
|---------|------:|---------------:|-------------:|------|
| email-Eu-core-temporal | 986 | 332,334 | 24,929 | [SNAP](https://snap.stanford.edu/data/email-Eu-core-temporal.html) |
| CollegeMsg | 1,899 | 59,835 | 20,296 | [SNAP](https://snap.stanford.edu/data/CollegeMsg.html) |
| sx-mathoverflow | 24,818 | 506,550 | 239,978 | [SNAP](https://snap.stanford.edu/data/sx-mathoverflow.html) |
| sx-askubuntu | 159,316 | 964,437 | 596,933 | [SNAP](https://snap.stanford.edu/data/sx-askubuntu.html) |
| sx-superuser | 194,085 | 1,443,339 | 924,886 | [SNAP](https://snap.stanford.edu/data/sx-superuser.html) |
| wiki-talk-temporal | 1,140,149 | 7,833,140 | 3,309,592 | [SNAP](https://snap.stanford.edu/data/wiki-talk-temporal.html) |
| sx-stackoverflow | 2,601,977 | 63,497,050 | 36,233,450 | [SNAP](https://snap.stanford.edu/data/sx-stackoverflow.html) |

Currently downloaded under `real_graphs/snap_temporal/graphs/`: CollegeMsg,
sx-mathoverflow, sx-askubuntu, sx-superuser.

## KONECT Datasets

| Dataset | Nodes | Edges | Type | Link |
|---------|-------|-------|------|------|
| Hamsterster | 2,426 | 16,630 | Social | [KONECT](http://konect.cc/networks/petster-hamster/) |
| PGP web of trust | 10,680 | 24,316 | Trust | [KONECT](http://konect.cc/networks/arenas-pgp/) |
| Brightkite | 58,228 | 214,078 | Location social | [KONECT](http://konect.cc/networks/loc-brightkite_edges/) |

## Notes

- SNAP files are tab-separated edge lists with `#` comment headers. Need to strip comments and remap to contiguous 0-indexed node IDs.
- Newman's files are in GML format. Convert using `networkx.read_gml()` + `convert_node_labels_to_integers()`.
- KONECT files typically have a header to skip and are space/tab-separated.
- All graphs above are undirected. Our input format expects `n_nodes m_undirected_edges` on the first line, then `u v` per undirected edge (stored as both directions internally).
- com-LiveJournal (currently in real_graphs/) has ~4M nodes / ~35M edges -- too large for quick iteration.

