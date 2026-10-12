cd /mnt/proj-shared/VASTdata/zdr/ELM_emulator

/mnt/proj-shared/VASTdata/zdr/envs/torch/bin/python -m elm_emulator replay \
  --regional \
  --universal \
  --device cuda \
  --hydrology-backend full \
  --met-dir met_forcing/Daymet_ERA5_TESSFA \
  --regional-met-format tessfa \
  --regional-surfdata data/surface_data/surfdata.TESSFA_DOMAIN2.4km.1d.NLCD.SSURGO.MODISLAI.c260814.nc4 \
  --clm-params data/params/clm_params_c180524.nc \
  --regional-physics-step-hours 3  --hydrology-backend full --disable-snow \
  --year 1980 \
  --month 1 \
  --nmonths 60 \
  --regional-min-lon -95 \
  --regional-max-lon -74 \
  --regional-min-lat 25 \
  --regional-max-lat 40 \
  --online-carbon \
  --regional-daily-output output/southeast_1980_6month_carbon_daily.nc \
  --regional-output-frequency daily \
  --regional-plot-dir output/southeast_1980_6month_carbon_plots \
--regional-plot-variable \
  gpp_gc_m2_d_mean \
  lai_mean \
  qflx_runoff_tot_mean \
  eflx_lh_tot_mean \
  eflx_sh_tot_mean \
  carbon_totsomc \
  carbon_totvegc \
  --no-compile \
  --no-compile-physics
