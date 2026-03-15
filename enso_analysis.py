#!/usr/bin/env python3
"""
ENSO Diversity Analysis: EP vs CP ENSO pattern correlations
Works with any subset of models that have valid .nc files downloaded.

Pipeline:
1. Load HadISST (OBS) + any available CMIP6 model files
2. Subset 1980-2014, compute SSTA (remove monthly climatology)
3. Compute Nino 1+2 and Nino 4 indices (area-weighted)
4. EP ENSO: regress out Nino4, take EOF1 of residual SSTA
5. CP ENSO: regress out Nino1+2, take EOF1 of residual SSTA
6. Compute area-weighted pattern correlation (model vs OBS)
7. Produce: correlations.csv, eof_patterns.png, scatter_ep_cp.png,
            bar_ep.png, bar_cp.png
"""

import os, glob, warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
OBS_PATH  = "/Users/shiyang/Downloads/HadISST/OBS_HadISST_reanaly_1_Omon_tos_187001-202112.nc"
CMIP6_DIR = os.path.expanduser("~/Downloads/CMIP6")
OUT_DIR   = "output"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_START, TIME_END = "1980-01", "2014-12"
LAT_MIN, LAT_MAX = -30, 30
LON_MIN, LON_MAX = 120, 280   # 0-360 convention

# Nino index boxes (lat_min, lat_max, lon_min, lon_max) — 0-360 lon
NINO12_BOX = (-10,  0, 270, 280)
NINO4_BOX  = ( -5,  5, 160, 210)

# ── Helpers ────────────────────────────────────────────────────────────────────

def area_weights(lat):
    """1-D cos(lat) weights."""
    return np.cos(np.deg2rad(lat))


def weighted_mean(data, weights_lat):
    """
    Area-weighted spatial mean over (lat, lon) axes.
    data: (time, lat, lon)  weights_lat: (lat,)
    returns: (time,)
    """
    w = weights_lat[:, np.newaxis]          # (lat, 1)
    mask = np.isfinite(data)
    wsum = np.where(mask, w, 0).sum(axis=(1,2))
    return np.nansum(data * w, axis=(1,2)) / np.where(wsum==0, np.nan, wsum)


def remove_monthly_climatology(data):
    """data: (time, ...) — remove monthly mean."""
    nt = data.shape[0]
    out = data.copy()
    for m in range(12):
        idx = np.arange(m, nt, 12)
        clim = np.nanmean(data[idx], axis=0)
        out[idx] -= clim
    return out


def regress_and_residual(ssta, index):
    """
    Linear regression of ssta (time,lat,lon) on index (time,).
    Returns residual = ssta - beta*index.
    """
    idx_norm = index / np.std(index) if np.std(index) > 0 else index
    nt, nlat, nlon = ssta.shape
    idx2 = idx_norm[:, np.newaxis, np.newaxis]
    beta = (np.nansum(ssta * idx2, axis=0) /
            np.nansum(idx2**2, axis=0))           # (lat, lon)
    residual = ssta - beta[np.newaxis] * idx2
    return residual


def eof1(data, lat):
    """
    Compute EOF1 of data (time, lat, lon) using area-weighted SVD.
    Returns spatial pattern (lat, lon), normalized to unit variance.
    Sign convention: positive loading in Nino3 region (5S-5N, 210-270E).
    """
    w = np.sqrt(np.cos(np.deg2rad(lat)))[:, np.newaxis]   # (lat,1)
    nt, nlat, nlon = data.shape
    X = data.copy()
    # Fill NaN with 0 for SVD
    X = np.where(np.isfinite(X), X, 0) * w[np.newaxis]
    X2d = X.reshape(nt, nlat*nlon)
    # Remove time mean
    X2d -= X2d.mean(axis=0)
    try:
        # Use truncated SVD (much faster than full SVD for large grids)
        try:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vt = randomized_svd(X2d, n_components=1, random_state=42)
        except ImportError:
            from scipy.sparse.linalg import svds
            U, s, Vt = svds(X2d, k=1)
            Vt = Vt[::-1]   # svds returns smallest first
        pattern = (Vt[0].reshape(nlat, nlon) / w)          # unweight
    except Exception as e:
        print(f"    SVD fallback to full SVD: {e}")
        try:
            U, s, Vt = np.linalg.svd(X2d, full_matrices=False)
            pattern = (Vt[0].reshape(nlat, nlon) / w)
        except Exception as e2:
            print(f"    SVD error: {e2}")
            return np.full((nlat, nlon), np.nan)

    # Sign convention: EP pattern should be positive in Nino3 (5S-5N,210-270)
    # CP pattern should be positive in Nino4 (5S-5N, 160-210)
    return pattern


def fix_sign(pattern, lat, lon, ref_lat=(-5,5), ref_lon=(210,270)):
    """Flip sign so mean over reference box is positive."""
    j = np.where((lat >= ref_lat[0]) & (lat <= ref_lat[1]))[0]
    i = np.where((lon >= ref_lon[0]) & (lon <= ref_lon[1]))[0]
    if len(j)==0 or len(i)==0:
        return pattern
    ref_val = np.nanmean(pattern[np.ix_(j, i)])
    if ref_val < 0:
        return -pattern
    return pattern


def pattern_correlation(pat1, pat2, lat):
    """
    Area-weighted pattern correlation between two (lat,lon) arrays.
    Both must be on the same grid. NaNs excluded.
    """
    w = np.cos(np.deg2rad(lat))[:, np.newaxis]
    mask = np.isfinite(pat1) & np.isfinite(pat2)
    if mask.sum() < 10:
        return np.nan
    p1 = pat1[mask]
    p2 = pat2[mask]
    ww = np.broadcast_to(w, pat1.shape)[mask]
    p1c = p1 - np.average(p1, weights=ww)
    p2c = p2 - np.average(p2, weights=ww)
    num = np.sum(ww * p1c * p2c)
    den = np.sqrt(np.sum(ww * p1c**2) * np.sum(ww * p2c**2))
    return float(num/den) if den > 0 else np.nan


def interpolate_to_obs_grid(pattern, lat_src, lon_src, lat_obs, lon_obs):
    """Bilinear interpolation of model pattern onto OBS grid."""
    # Ensure monotonically increasing
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        pattern = pattern[::-1, :]
    if lon_src[0] > lon_src[-1]:
        lon_src = lon_src[::-1]
        pattern = pattern[:, ::-1]
    interp = RegularGridInterpolator(
        (lat_src, lon_src), pattern,
        method="linear", bounds_error=False, fill_value=np.nan
    )
    ll, ln = np.meshgrid(lat_obs, lon_obs, indexing="ij")
    return interp(np.stack([ll.ravel(), ln.ravel()], axis=1)).reshape(len(lat_obs), len(lon_obs))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_prepare(nc_path, label=""):
    """
    Load a NetCDF file, subset to 1980-2014 tropical Pacific, compute SSTA.
    Returns dict with keys: ssta, lat, lon, nino12, nino4, label
    or None on failure.
    """
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"  [ERROR] Cannot open {nc_path}: {e}")
        return None

    # Find SST variable
    tos = None
    for vname in ["tos", "sst", "SST", "TOS"]:
        if vname in ds:
            tos = ds[vname]
            break
    if tos is None:
        print(f"  [ERROR] No tos variable in {nc_path}")
        return None

    # Time subset
    try:
        tos = tos.sel(time=slice(TIME_START, TIME_END))
    except Exception:
        pass
    if len(tos.time) < 12:
        print(f"  [SKIP] {label}: only {len(tos.time)} time steps")
        return None

    # Determine grid type
    dims = tos.dims   # e.g. (time, lat, lon) or (time, j, i)
    has_regular = ("lat" in dims and "lon" in dims)

    if has_regular:
        lat = tos["lat"].values.astype(float)
        lon = tos["lon"].values.astype(float)
        # Normalize lon to 0-360
        lon = lon % 360
        tos = tos.assign_coords(lon=lon).sortby("lon").sortby("lat")
        lat = tos["lat"].values
        lon = tos["lon"].values
        # Spatial subset
        tos = tos.sel(lat=slice(LAT_MIN, LAT_MAX),
                      lon=slice(LON_MIN, LON_MAX))
        lat = tos["lat"].values
        lon = tos["lon"].values
        data = tos.values.astype(float)  # (time, lat, lon)

    else:
        # Curvilinear grid — regrid to 1° regular
        print(f"  [INFO] {label}: curvilinear grid → regridding to 1°")
        lat2d_coord, lon2d_coord = None, None
        for cn in ["latitude","lat","nav_lat","TLAT"]:
            if cn in ds.coords and ds[cn].ndim == 2:
                lat2d_coord = ds[cn].values; break
        for cn in ["longitude","lon","nav_lon","TLONG"]:
            if cn in ds.coords and ds[cn].ndim == 2:
                lon2d_coord = ds[cn].values % 360; break
        if lat2d_coord is None:
            print(f"  [ERROR] {label}: cannot find 2D lat/lon")
            return None

        # Mask to tropical Pacific
        mask = ((lat2d_coord >= LAT_MIN) & (lat2d_coord <= LAT_MAX) &
                (lon2d_coord >= LON_MIN) & (lon2d_coord <= LON_MAX))
        j_idx = np.where(mask.any(axis=1))[0]
        i_idx = np.where(mask.any(axis=0))[0]
        if len(j_idx)==0 or len(i_idx)==0:
            print(f"  [ERROR] {label}: spatial subset is empty")
            return None
        j_sl = slice(j_idx[0], j_idx[-1]+1)
        i_sl = slice(i_idx[0], i_idx[-1]+1)

        raw = tos.isel(**{tos.dims[1]: j_sl, tos.dims[2]: i_sl}).values.astype(float)
        lat2d = lat2d_coord[j_sl, i_sl]
        lon2d = lon2d_coord[j_sl, i_sl]

        # Target 1° grid
        target_lat = np.arange(LAT_MIN, LAT_MAX+1, 1.0)
        target_lon = np.arange(LON_MIN, LON_MAX+1, 1.0)
        from scipy.interpolate import griddata
        nt = raw.shape[0]
        data = np.full((nt, len(target_lat), len(target_lon)), np.nan, float)
        pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
        gl, gn = np.meshgrid(target_lat, target_lon, indexing="ij")
        # Vectorized IDW: build k-NN tree once, apply inverse-distance weights to all timesteps
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        query_pts = np.column_stack([gl.ravel(), gn.ravel()])
        dist, idx = tree.query(query_pts, k=4, workers=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_idw = 1.0 / np.where(dist == 0, 1e-12, dist)**2
            w_idw /= w_idw.sum(axis=1, keepdims=True)
        npts_out = gl.size
        raw_flat = raw.reshape(nt, -1)
        out_flat = np.full((nt, npts_out), np.nan)
        for t in range(nt):
            v = raw_flat[t]
            vals_k = v[idx]
            valid_k = np.isfinite(vals_k) & (np.abs(vals_k) < 400)
            wsum = (w_idw * valid_k).sum(axis=1)
            with np.errstate(invalid="ignore"):
                out_flat[t] = np.where(
                    wsum > 0,
                    (w_idw * np.where(valid_k, vals_k, 0)).sum(axis=1) / wsum,
                    np.nan
                )
        data = out_flat.reshape(nt, len(target_lat), len(target_lon))
        lat, lon = target_lat, target_lon

    # Replace land/fill values
    data = np.where(np.abs(data) > 100, np.nan, data)

    # Remove monthly climatology → SSTA
    ssta = remove_monthly_climatology(data)

    # Nino indices (area-weighted)
    def nino_index(ssta, lat, lon, box):
        la0,la1,lo0,lo1 = box
        jj = np.where((lat>=la0)&(lat<=la1))[0]
        ii = np.where((lon>=lo0)&(lon<=lo1))[0]
        if len(jj)==0 or len(ii)==0:
            return np.zeros(ssta.shape[0])
        sub = ssta[:, np.ix_(jj,ii)[0], np.ix_(jj,ii)[1]]
        sub = ssta[np.ix_(range(ssta.shape[0]), jj, ii)]
        wlat = area_weights(lat[jj])
        return weighted_mean(sub, wlat)

    nino12 = nino_index(ssta, lat, lon, NINO12_BOX)
    nino4  = nino_index(ssta, lat, lon, NINO4_BOX)

    return dict(ssta=ssta, lat=lat, lon=lon,
                nino12=nino12, nino4=nino4, label=label)


def compute_ep_cp_patterns(prepared):
    """Compute EP and CP EOF patterns from a prepared dataset."""
    ssta   = prepared["ssta"]
    lat    = prepared["lat"]
    lon    = prepared["lon"]
    nino12 = prepared["nino12"]
    nino4  = prepared["nino4"]

    # EP ENSO: remove Nino4 signal, EOF1 of residual
    resid_ep = regress_and_residual(ssta, nino4)
    ep_pat   = eof1(resid_ep, lat)
    ep_pat   = fix_sign(ep_pat, lat, lon, ref_lat=(-5,5), ref_lon=(210,270))

    # CP ENSO: remove Nino1+2 signal, EOF1 of residual
    resid_cp = regress_and_residual(ssta, nino12)
    cp_pat   = eof1(resid_cp, lat)
    cp_pat   = fix_sign(cp_pat, lat, lon, ref_lat=(-5,5), ref_lon=(160,210))

    return ep_pat, cp_pat


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_eof_patterns(obs_data, model_data_list, out_path):
    """
    Plot EP and CP ENSO patterns for OBS and each model side by side.
    Uses Cartopy if available, else plain matplotlib.
    """
    all_datasets = [obs_data] + model_data_list
    n = len(all_datasets)
    ncols = 2  # EP | CP
    nrows = n

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 3.5*nrows),
                             subplot_kw={"projection": ccrs.PlateCarree()} if USE_CARTOPY else {})
    if nrows == 1:
        axes = axes[np.newaxis, :]

    cmap = plt.cm.RdBu_r
    vmax = 0.6

    for row, dset in enumerate(all_datasets):
        ep_pat = dset["ep_pat"]
        cp_pat = dset["cp_pat"]
        lat    = dset["lat"]
        lon    = dset["lon"]
        label  = dset["label"]
        lon2d, lat2d = np.meshgrid(lon, lat)

        for col, (pat, title_sfx) in enumerate([(ep_pat,"EP ENSO"), (cp_pat,"CP ENSO")]):
            ax = axes[row, col]
            if USE_CARTOPY:
                ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                ax.coastlines(linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
                pcm = ax.pcolormesh(lon2d, lat2d, pat,
                                    cmap=cmap, vmin=-vmax, vmax=vmax,
                                    transform=ccrs.PlateCarree(), zorder=0)
            else:
                pcm = ax.pcolormesh(lon2d, lat2d, pat,
                                    cmap=cmap, vmin=-vmax, vmax=vmax)
                ax.set_xlim(LON_MIN, LON_MAX)
                ax.set_ylim(LAT_MIN, LAT_MAX)
            if row == 0:
                ax.set_title(title_sfx, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=9, labelpad=3)
            plt.colorbar(pcm, ax=ax, orientation="vertical",
                         fraction=0.03, pad=0.02, label="°C/std")

    fig.suptitle("ENSO Diversity: EP and CP Patterns", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_scatter(corr_ep, corr_cp, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    for i, (ep, cp, lab) in enumerate(zip(corr_ep, corr_cp, labels)):
        ax.scatter(ep, cp, color=colors[i], s=80, zorder=3, label=lab)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("EP ENSO Pattern Correlation", fontsize=12)
    ax.set_ylabel("CP ENSO Pattern Correlation", fontsize=12)
    ax.set_title("CMIP6 vs HadISST: EP and CP ENSO Pattern Correlations", fontsize=12)
    ax.set_xlim(-0.2, 1.05); ax.set_ylim(-0.2, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_bar(corr_vals, labels, title, ylabel, out_path, color):
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.7), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, corr_vals, color=color, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.3, 1.1)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, corr_vals):
        if np.isfinite(val):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("ENSO Diversity Analysis")
    print("="*60)

    # 1. Load OBS (HadISST)
    print("\n[1] Loading HadISST OBS...")
    obs = load_and_prepare(OBS_PATH, label="HadISST")
    if obs is None:
        print("FATAL: Cannot load HadISST. Aborting.")
        return
    obs["ep_pat"], obs["cp_pat"] = compute_ep_cp_patterns(obs)
    print(f"  OBS EP/CP patterns computed. lat={obs['lat'].shape}, lon={obs['lon'].shape}")

    obs_ep_on_obsgrid = obs["ep_pat"]
    obs_cp_on_obsgrid = obs["cp_pat"]
    obs_lat = obs["lat"]
    obs_lon = obs["lon"]

    # 2. Discover available model files
    print("\n[2] Scanning CMIP6 downloads...")
    model_files = []
    if os.path.isdir(CMIP6_DIR):
        for model_dir in sorted(os.listdir(CMIP6_DIR)):
            dpath = os.path.join(CMIP6_DIR, model_dir)
            if not os.path.isdir(dpath):
                continue
            nc_files = sorted(glob.glob(os.path.join(dpath, "*.nc")))
            valid = []
            for f in nc_files:
                if os.path.getsize(f) > 500_000:  # >0.5MB
                    valid.append(f)
            if valid:
                model_files.append((model_dir, valid))
                print(f"  Found: {model_dir} ({len(valid)} file(s))")
    if not model_files:
        print("  No CMIP6 model files available yet.")

    # 3. Process each model
    print("\n[3] Processing models...")
    results = []
    all_datasets_for_plot = []

    for model_name, nc_paths in model_files:
        print(f"\n  -- {model_name} --")

        # For multi-file models, concatenate
        if len(nc_paths) == 1:
            prepared = load_and_prepare(nc_paths[0], label=model_name)
        else:
            # Multiple files: try concatenating
            dss = []
            for p in nc_paths:
                d = load_and_prepare(p, label=model_name)
                if d is not None:
                    dss.append(d)
            if not dss:
                continue
            # Merge the SSTA arrays along time
            from itertools import chain
            # Use first dataset's grid (they should be consistent)
            prepared = dss[0]
            if len(dss) > 1:
                prepared["ssta"] = np.concatenate([d["ssta"] for d in dss], axis=0)
                prepared["nino12"] = np.concatenate([d["nino12"] for d in dss])
                prepared["nino4"] = np.concatenate([d["nino4"] for d in dss])

        if prepared is None:
            continue

        ep_pat, cp_pat = compute_ep_cp_patterns(prepared)
        prepared["ep_pat"] = ep_pat
        prepared["cp_pat"] = cp_pat

        # Interpolate to OBS grid for correlation
        ep_on_obs = interpolate_to_obs_grid(ep_pat,
                                            prepared["lat"], prepared["lon"],
                                            obs_lat, obs_lon)
        cp_on_obs = interpolate_to_obs_grid(cp_pat,
                                            prepared["lat"], prepared["lon"],
                                            obs_lat, obs_lon)

        corr_ep = pattern_correlation(obs_ep_on_obsgrid, ep_on_obs, obs_lat)
        corr_cp = pattern_correlation(obs_cp_on_obsgrid, cp_on_obs, obs_lat)

        print(f"  EP corr = {corr_ep:.3f} | CP corr = {corr_cp:.3f}")
        results.append({"model": model_name, "corr_ep": corr_ep, "corr_cp": corr_cp})
        all_datasets_for_plot.append(prepared)

    # 4. Save correlations.csv
    import csv
    csv_path = os.path.join(OUT_DIR, "correlations.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","corr_ep","corr_cp"])
        w.writeheader()
        w.writerows(results)
    print(f"\n[4] Saved: {csv_path}")

    # Print table
    print("\n  MODEL PATTERN CORRELATIONS vs HadISST")
    print(f"  {'Model':<25} {'EP corr':>10} {'CP corr':>10}")
    print("  " + "-"*47)
    for r in results:
        print(f"  {r['model']:<25} {r['corr_ep']:>10.3f} {r['corr_cp']:>10.3f}")

    # 5. Plots — only if we have at least 1 model
    if not results:
        print("\nNo model results — skipping plots. (Re-run once downloads complete.)")
        print("\n[DONE] OBS patterns computed. Awaiting CMIP6 downloads.")
        return

    print("\n[5] Generating plots...")

    # EOF patterns figure
    plot_eof_patterns(obs, all_datasets_for_plot,
                      os.path.join(OUT_DIR, "eof_patterns.png"))

    # Scatter: EP corr vs CP corr
    m_labels  = [r["model"] for r in results]
    ep_corrs  = [r["corr_ep"] for r in results]
    cp_corrs  = [r["corr_cp"] for r in results]
    plot_scatter(ep_corrs, cp_corrs, m_labels,
                 os.path.join(OUT_DIR, "scatter_ep_cp.png"))

    # Bar charts
    plot_bar(ep_corrs, m_labels,
             "EP ENSO Pattern Correlation (CMIP6 vs HadISST)",
             "Pattern Correlation",
             os.path.join(OUT_DIR, "bar_ep.png"),
             color="#E87722")

    plot_bar(cp_corrs, m_labels,
             "CP ENSO Pattern Correlation (CMIP6 vs HadISST)",
             "Pattern Correlation",
             os.path.join(OUT_DIR, "bar_cp.png"),
             color="#3A86FF")

    print("\n" + "="*60)
    print(f"DONE. {len(results)} model(s) processed.")
    print(f"Outputs in: {OUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
