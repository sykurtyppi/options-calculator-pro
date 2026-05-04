#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/Users/tristanalejandro/Desktop/options_calculator_pro/.venv_arm64/bin/python"
PROJECT_ROOT="/Users/tristanalejandro/Downloads/options_calculator_pro"
DATA_ROOT="/Volumes/T9/market_data"
RUN_LABEL="top250_batch9_overnight_2023_2026"
START_DATE="2023-01-01"
END_DATE="2026-04-02"
SYMBOLS="MMC,AUTO,FI,WAB,WBD,WDAY,WELL,WMB,WM,XEL,XYL,YUM,ZBH,CVS,TGT,BA,SLB,CRWD,SNPS,DDOG,MRVL,ON,VRTX,DXCM,ALGN,TJX,DLTR,EBAY,SPGI,BK,FITB,CFG,T,VZ,TMUS,GOOG,NWS,NWSA,FOXA,FOX,LYV,TKO,OMC,TTD,ADP,ADSK,AKAM,APP,CDW,CTSH,DELL,EPAM,FFIV,FICO,FIS,FISV,GEN,GDDY,GPN,GLW,HPE,HPQ,INTC,IT,JKHY,KEYS,MPWR,MSCI"

cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/ivol_options_backfill.py" \
  --symbols "$SYMBOLS" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --data-root "$DATA_ROOT" \
  --include-underlying \
  --timeout-seconds 60 \
  --sleep-seconds 1.25 \
  --run-label "$RUN_LABEL"
