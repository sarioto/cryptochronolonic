from hist_service import HistWorker

hs = HistWorker()
hs.combine_polo_frames_vol_sorted(3)

print(next(iter(hs.currentHists.values())).head())

hs.currentHists = {}
hs.combine_binance_frames_vol_sorted(3)

print(next(iter(hs.currentHists.values())).head())