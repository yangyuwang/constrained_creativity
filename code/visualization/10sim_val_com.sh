module load python/anaconda-2022.05 spark/3.3.2 graphviz
source activate gallery_gpt

REAL_META="/home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv"
REAL_NPZ="/home/wangyd/Projects/macs_thesis/yangyu/artwork_data/artwork_style_embeddings.npz"

# Paris validation
python code/visualization/10sim_val_com.py \
  --sim-file /home/wangyd/Projects/macs_thesis/data/paris_simulation_style.npz \
  --real-meta $REAL_META \
  --real-npz  $REAL_NPZ \
  --outdir /home/wangyd/Projects/macs_thesis/yangyu/simulation_results/out_val_paris \
  --mode validation \
  --setting validation \
  --year-lo 1921 --year-hi 1925 \
  --round-min 0 --round-max 4 \
  --real-artist-field Artist_name \
  --slugify-real-artists \
  --panel-cols 4 --panel-rows 5

# Paris comparison
python code/visualization/10sim_val_com.py \
  --sim-file /home/wangyd/Projects/macs_thesis/data/paris_simulation_style.npz \
  --real-meta $REAL_META \
  --real-npz  $REAL_NPZ \
  --outdir /home/wangyd/Projects/macs_thesis/yangyu/simulation_results/out_com_paris \
  --mode comparison \
  --comparison-conditions edge01_smallmove edge01_largemove edge05_smallmove edge05_largemove \
  --year-lo 1921 --year-hi 1925 \
  --round-min 0 --round-max 4 \
  --real-artist-field Artist_name \
  --slugify-real-artists \
  --panel-cols 4 --panel-rows 5