from pyspark.sql import SparkSession, functions as F, types as T
import pandas as pd
import json
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
meta_path = "/home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv"
json_path = "/home/wangyd/Projects/macs_thesis/yangyu/artist_demographics/demographic_location.json"
pair_path = "/home/wangyd/Projects/macs_thesis/yangyu/artwork_data/artwork_similarity_pairs_50.parquet"
output_pairs_path = "/home/wangyd/Projects/macs_thesis/yangyu/artwork_data/artwork_similarity_pairs_attributes_50geo.parquet"

# ---------------------------------------------------------------------
# 1) Load meta + demo; build image -> (artist, year) mappings
# ---------------------------------------------------------------------
meta = pd.read_csv(meta_path, low_memory=False)
meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce")

with open(json_path, "r") as f:
    demo = json.load(f)

def csv_artist_to_slug(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    if "/" in x:
        x = x.split("/")[-1]
    return x

meta["artist_slug"] = meta["Artist_name"].map(csv_artist_to_slug)
meta_filtered = meta[(meta["Year"] >= 1400) & (meta["artist_slug"].isin(demo.keys()))].copy()

img2artist = dict(zip(meta_filtered["image_n"], meta_filtered["artist_slug"]))
img2year   = dict(zip(meta_filtered["image_n"], meta_filtered["Year"]))

# ---------------------------------------------------------------------
# 2) Build dynamic graphs from JSON (with life-span constraints)
# ---------------------------------------------------------------------
def is_alive(person_id, year):
    info = demo.get(person_id, {})
    birth = info.get("birth_year")
    death = info.get("death_year")
    if birth is not None and year < birth:
        return False
    if death is not None and year > death:
        return False
    return True

pair_first_year = {}   # (u, v) -> earliest_year
pair_meta = {}         # (u, v) -> list of interaction dicts

for ego, info in demo.items():
    interactions = info.get("interactions") or []
    for inter in interactions:
        alter = inter.get("name")
        year = inter.get("year")
        if alter is None or year is None:
            continue
        if alter not in demo:
            continue

        u, v = sorted([ego, alter])
        pair = (u, v)

        if pair not in pair_first_year or year < pair_first_year[pair]:
            pair_first_year[pair] = year

        pair_meta.setdefault(pair, []).append(inter)

years = list(range(1400, 2024))
graphs_by_year = {}

for y in years:
    G = nx.Graph()
    alive_artists = [pid for pid in demo.keys() if is_alive(pid, y)]
    G.add_nodes_from(alive_artists)

    for (u, v), first_year in pair_first_year.items():
        if first_year <= y and is_alive(u, y) and is_alive(v, y):
            G.add_edge(u, v, first_year=first_year, interactions=pair_meta[(u, v)])

    graphs_by_year[y] = G

graph_stats = {}
for year, G in graphs_by_year.items():
    n = G.number_of_nodes()
    if n > 1:
        density = nx.density(G)
        clustering = nx.average_clustering(G)
    else:
        density = 0.0
        clustering = 0.0
    graph_stats[year] = {
        "nodeCount": int(n),
        "graphDensity": float(density),
        "avgClustering": float(clustering),
    }

# ---------------------------------------------------------------------
# Helpers for location extraction (used in saved node CSV too)
# ---------------------------------------------------------------------
def in_interval(year, start, end):
    if year is None:
        return False
    if start is not None and year < start:
        return False
    if end is not None and year > end:
        return False
    return True

def get_lat_lon(location_dict):
    if not location_dict or not isinstance(location_dict, dict):
        return (None, None)
    lat = location_dict.get("lat", location_dict.get("latitude"))
    lon = location_dict.get("lon", location_dict.get("longitude"))
    if lat is None or lon is None:
        return (None, None)
    try:
        return (float(lat), float(lon))
    except Exception:
        return (None, None)

def get_location_at_year_from_demo(pid, year, demo_dict):
    info = demo_dict.get(pid, {})

    # residences first
    residences = info.get("residences") or []
    matches = []
    for res in residences:
        start = res.get("start_year")
        end = res.get("end_year")
        if not in_interval(year, start, end):
            continue
        lat, lon = get_lat_lon(res.get("location"))
        if lat is None or lon is None:
            continue
        start_key = start if start is not None else -1
        matches.append((start_key, lat, lon))

    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        _, lat, lon = matches[0]
        return (lat, lon)

    # fallback birth_place
    lat, lon = get_lat_lon(info.get("birth_place"))
    if lat is not None and lon is not None:
        return (lat, lon)

    return (None, None)

# Save start-year graph

year0 = 1920
G0 = graphs_by_year.get(year0)

if G0 is not None:
    # -----------------------------
    # 1) Edge list
    # -----------------------------
    edge_rows = []
    for u, v, data in G0.edges(data=True):
        edge_rows.append({
            "source": u,
            "target": v,
            "first_year": data.get("first_year"),
        })

    edge_path = (
        "/home/wangyd/Projects/macs_thesis/data/"
        "artist_network_start_edgelist.csv"
    )
    pd.DataFrame(edge_rows).to_csv(edge_path, index=False)
    print(f"[INFO] Saved {len(edge_rows)} edges for year {year0} to {edge_path}")

    # -----------------------------
    # 2) Node list
    # -----------------------------
    node_rows = []
    for n, data in G0.nodes(data=True):
        lat, lon = get_location_at_year_from_demo(n, year0, demo)

        row = {
            "node": n,
            "year": year0,
            "latitude": lat,
            "longitude": lon,
        }

        # include all node attributes if any
        row.update(data)
        node_rows.append(row)

    node_path = (
        "/home/wangyd/Projects/macs_thesis/data/"
        "artist_network_start_nodelist.csv"
    )
    pd.DataFrame(node_rows).to_csv(node_path, index=False)
    print(f"[INFO] Saved {len(node_rows)} nodes for year {year0} to {node_path}")

else:
    print(f"[WARN] No graph found for year {year0}; edge/node lists not written.")

# ---------------------------------------------------------------------
# 3) Spark setup and broadcasts
# ---------------------------------------------------------------------
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

img2artist_bc = sc.broadcast(img2artist)
img2year_bc   = sc.broadcast(img2year)
demo_bc = sc.broadcast(demo)
graphs_by_year_bc = sc.broadcast(graphs_by_year)
graph_stats_bc = sc.broadcast(graph_stats)

pairs_sdf = spark.read.parquet(pair_path)

@F.udf(T.StringType())
def img_to_artist_udf(img_id):
    if img_id is None:
        return None
    return img2artist_bc.value.get(img_id)

@F.udf(T.IntegerType())
def img_to_year_udf(img_id):
    if img_id is None:
        return None
    y = img2year_bc.value.get(img_id)
    if y is None:
        return None
    try:
        return int(y)
    except Exception:
        return None

pairs_sdf = (
    pairs_sdf
    .withColumn("src_artist", img_to_artist_udf("src"))
    .withColumn("dst_artist", img_to_artist_udf("dst"))
    .withColumn("src_year", img_to_year_udf("src"))
    .withColumn("dst_year", img_to_year_udf("dst"))
    .where(
        F.col("src_artist").isNotNull() &
        F.col("dst_artist").isNotNull() &
        F.col("src_year").isNotNull() &
        F.col("dst_year").isNotNull()
    )
)

# ---------------------------------------------------------------------
# 4) UDF to compute features (NO geoAffinity; geoDistance uses src_year/dst_year)
# ---------------------------------------------------------------------
features_schema = T.StructType([
    T.StructField("self", T.BooleanType(), True),
    T.StructField("firstOrderTie", T.BooleanType(), True),
    T.StructField("secondOrderTie", T.BooleanType(), True),
    T.StructField("birthYearDiff", T.DoubleType(), True),
    T.StructField("geoDistance", T.DoubleType(), True),
    T.StructField("nationalityAffinity", T.BooleanType(), True),
    T.StructField("affiliationAffinity", T.BooleanType(), True),
    T.StructField("genderAffinity", T.BooleanType(), True),
    T.StructField("educationAffinity", T.BooleanType(), True),
    T.StructField("religionAffinity", T.BooleanType(), True),
    T.StructField("languageAffinity", T.BooleanType(), True),
    T.StructField("graphNodeCount", T.IntegerType(), True),
    T.StructField("graphDensity", T.DoubleType(), True),
    T.StructField("avgClustering", T.DoubleType(), True),
])

@F.udf(features_schema)
def compute_features_udf(src_artist, dst_artist, src_year, dst_year):
    import numpy as np

    if src_artist is None or dst_artist is None or src_year is None or dst_year is None:
        return {k: None for k in features_schema.fieldNames()}

    demo = demo_bc.value
    graphs_by_year = graphs_by_year_bc.value
    graph_stats = graph_stats_bc.value

    # ----------------- helpers -----------------
    def norm_str(s):
        if s is None:
            return None
        return str(s).strip().lower()

    def get_person(pid):
        return demo.get(pid, {}) if pid else {}

    def get_birth_year(pid):
        return get_person(pid).get("birth_year")

    def in_interval(year, start, end):
        if year is None:
            return False
        if start is not None and year < start:
            return False
        if end is not None and year > end:
            return False
        return True

    def get_lat_lon(location_dict):
        if not location_dict or not isinstance(location_dict, dict):
            return (None, None)
        lat = location_dict.get("lat", location_dict.get("latitude"))
        lon = location_dict.get("lon", location_dict.get("longitude"))
        if lat is None or lon is None:
            return (None, None)
        try:
            return (float(lat), float(lon))
        except Exception:
            return (None, None)

    def get_location_at_year(pid, year):
        info = get_person(pid)

        # residences first
        residences = info.get("residences") or []
        matches = []
        for res in residences:
            start = res.get("start_year")
            end = res.get("end_year")
            if not in_interval(year, start, end):
                continue
            lat, lon = get_lat_lon(res.get("location"))
            if lat is None or lon is None:
                continue
            # choose latest-start residence among matches
            start_key = start if start is not None else -1
            matches.append((start_key, lat, lon))

        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            _, lat, lon = matches[0]
            return (lat, lon)

        # fallback birth_place
        lat, lon = get_lat_lon(info.get("birth_place"))
        if lat is not None and lon is not None:
            return (lat, lon)

        return (None, None)

    def haversine_distance(lat1, lon1, lat2, lon2):
        if any(x is None for x in [lat1, lon1, lat2, lon2]):
            return None
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return float(6371.0 * c)

    def get_affiliation_groups_before(pid, year):
        info = get_person(pid)
        groups = set()
        entries = info.get("affiliations") or []
        for a in entries:
            group = norm_str(a.get("group"))
            if not group:
                continue
            start = a.get("start_year")
            if year is None or start is None or start <= year:
                groups.add(group)
        return groups

    def get_education_schools_before(pid, year):
        info = get_person(pid)
        schools = set()
        entries = info.get("education") or []
        for e in entries:
            school = norm_str(e.get("school"))
            if not school:
                continue
            ey = e.get("year")
            if ey is None or (year is not None and ey <= year):
                schools.add(school)
        return schools

    def get_religions(pid, year):
        info = get_person(pid)
        rel_field = info.get("religion")
        if rel_field is None:
            return set()

        if isinstance(rel_field, dict):
            entries = [rel_field]
        elif isinstance(rel_field, list):
            entries = rel_field
        else:
            name = norm_str(rel_field)
            return {name} if name else set()

        rels = set()
        for r in entries:
            if not isinstance(r, dict):
                name = norm_str(r)
                if name:
                    rels.add(name)
                continue
            name = norm_str(r.get("religion"))
            if not name:
                continue
            start = r.get("start_year")
            end = r.get("end_year")
            if in_interval(year, start, end):
                rels.add(name)
        return rels

    def get_languages(pid):
        info = get_person(pid)
        langs = info.get("languages_spoken") or []
        out = set()
        for l in langs:
            ln = norm_str(l)
            if ln:
                out.add(ln)
        return out

    def get_nationality(pid):
        return norm_str(get_person(pid).get("nationality"))

    def get_gender(pid):
        return norm_str(get_person(pid).get("gender"))

    # ----------------- self indicator -----------------
    self_flag = (src_artist == dst_artist)

    # ----------------- network ties (use src_year graph) -----------------
    G = graphs_by_year.get(int(src_year))
    firstOrderTie = False
    secondOrderTie = False
    if G is not None and src_artist in G and dst_artist in G:
        firstOrderTie = G.has_edge(src_artist, dst_artist)
        if not firstOrderTie:
            try:
                nbr_src = set(G.neighbors(src_artist))
                nbr_dst = set(G.neighbors(dst_artist))
                secondOrderTie = len(nbr_src.intersection(nbr_dst)) > 0
            except Exception:
                secondOrderTie = False

    # ----------------- birth year diff -----------------
    by1 = get_birth_year(src_artist)
    by2 = get_birth_year(dst_artist)
    birthYearDiff = float(abs(by1 - by2)) if (by1 is not None and by2 is not None) else None

    # ----------------- geo distance using each painting's year -----------------
    lat1, lon1 = get_location_at_year(src_artist, int(src_year))
    lat2, lon2 = get_location_at_year(dst_artist, int(dst_year))
    geoDistance = haversine_distance(lat1, lon1, lat2, lon2)

    # ----------------- nationality -----------------
    nat1 = get_nationality(src_artist)
    nat2 = get_nationality(dst_artist)
    nationalityAffinity = None if nat1 is None or nat2 is None else (nat1 == nat2)

    # ----------------- affiliation (use src_year cutoff) -----------------
    aff1 = get_affiliation_groups_before(src_artist, int(src_year))
    aff2 = get_affiliation_groups_before(dst_artist, int(src_year))
    affiliationAffinity = (len(aff1) > 0 and len(aff2) > 0 and len(aff1.intersection(aff2)) > 0)

    # ----------------- gender -----------------
    g1 = get_gender(src_artist)
    g2 = get_gender(dst_artist)
    genderAffinity = None if g1 is None or g2 is None else (g1 == g2)

    # ----------------- education (use src_year cutoff) -----------------
    edu1 = get_education_schools_before(src_artist, int(src_year))
    edu2 = get_education_schools_before(dst_artist, int(src_year))
    educationAffinity = (len(edu1) > 0 and len(edu2) > 0 and len(edu1.intersection(edu2)) > 0)

    # ----------------- religion (use src_year) -----------------
    rel1 = get_religions(src_artist, int(src_year))
    rel2 = get_religions(dst_artist, int(src_year))
    religionAffinity = (len(rel1) > 0 and len(rel2) > 0 and len(rel1.intersection(rel2)) > 0)

    # ----------------- language -----------------
    l1 = get_languages(src_artist)
    l2 = get_languages(dst_artist)
    languageAffinity = None if (not l1 or not l2) else (len(l1.intersection(l2)) > 0)

    # ----------------- graph-level stats (use src_year) -----------------
    stats = graph_stats.get(int(src_year))
    if stats is not None:
        graphNodeCount = stats["nodeCount"]
        graphDensity = stats["graphDensity"]
        avgClustering = stats["avgClustering"]
    else:
        graphNodeCount = None
        graphDensity = None
        avgClustering = None

    return {
        "self": bool(self_flag),
        "firstOrderTie": bool(firstOrderTie),
        "secondOrderTie": bool(secondOrderTie),
        "birthYearDiff": birthYearDiff,
        "geoDistance": geoDistance,
        "nationalityAffinity": nationalityAffinity,
        "affiliationAffinity": affiliationAffinity,
        "genderAffinity": genderAffinity,
        "educationAffinity": educationAffinity,
        "religionAffinity": religionAffinity,
        "languageAffinity": languageAffinity,
        "graphNodeCount": graphNodeCount,
        "graphDensity": graphDensity,
        "avgClustering": avgClustering,
    }

pairs_sdf = pairs_sdf.withColumn(
    "features",
    compute_features_udf("src_artist", "dst_artist", "src_year", "dst_year")
)

for name in features_schema.fieldNames():
    pairs_sdf = pairs_sdf.withColumn(name, F.col(f"features.{name}"))

# ---------------------------------------------------------------------
# Summary + write
# ---------------------------------------------------------------------
bool_cols = [
    "self",
    "firstOrderTie",
    "secondOrderTie",
    "nationalityAffinity",
    "affiliationAffinity",
    "genderAffinity",
    "educationAffinity",
    "religionAffinity",
    "languageAffinity",
]
agg_exprs = [F.mean(F.col(c).cast("double")).alias(c) for c in bool_cols]
agg_exprs += [
    F.mean("birthYearDiff").alias("meanBirthYearDiff"),
    F.mean("geoDistance").alias("meanGeoDistance"),
    F.mean("graphNodeCount").alias("meanGraphNodeCount"),
    F.mean("graphDensity").alias("meanGraphDensity"),
    F.mean("avgClustering").alias("meanAvgClustering"),
]

pairs_sdf.agg(*agg_exprs).show(truncate=False)

pairs_sdf.write.mode("overwrite").parquet(output_pairs_path)
pairs_sdf.show(10, truncate=False)