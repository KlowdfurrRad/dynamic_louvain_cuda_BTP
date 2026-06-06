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

## SNAP Datasets (Medium-Large, with ground truth communities)

| Dataset | Nodes | Edges | Type | Link |
|---------|-------|-------|------|------|
| com-DBLP | 317,080 | 1,049,866 | Collaboration | [SNAP](https://snap.stanford.edu/data/com-DBLP.html) |
| com-Amazon | 334,863 | 925,872 | Co-purchasing | [SNAP](https://snap.stanford.edu/data/com-Amazon.html) |
| com-Youtube | 1,134,890 | 2,987,624 | Social | [SNAP](https://snap.stanford.edu/data/com-Youtube.html) |

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
