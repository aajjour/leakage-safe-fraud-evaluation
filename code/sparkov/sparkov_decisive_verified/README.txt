1) ./setup_and_run.sh debug
2) ./verify_and_package.sh
3) ./setup_and_run.sh full
4) ./verify_and_package.sh
Full design: 6 future test months x 4 label delays x 3 training policies x 2 seeds x 3 models x 3 calibration states = 1296 metric rows.
Primary article decision subset: delay=14, capped_weighted, raw discrimination; Platt/beta calibration; 0.5% and 1% alert/value capture.
