.PHONY: install test smoke replicate refs clean demo demo-clips figures
JOBS ?= 4
install:
	pip install -e ".[dev]"
	nbstripout --install
test:
	pytest -q
smoke:
	python -m sim.runner experiments/configs/smoke.yaml
refs:            ## known-competitive and known-collusive references (for calibrating Delta and impulse plots)
	python -m sim.runner experiments/configs/competitive_reference.yaml
	python -m sim.runner experiments/configs/grim_reference.yaml
replicate:       ## Colliard-Foucault-Lovo baseline: alpha x N grid, 10 seeds (long)
	python experiments/sweep.py experiments/configs/replicate_cfl_2mm.yaml \
	    --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9 --jobs $(JOBS)
demo:            ## record every demo scenario from the simulator; then open demo/index.html in a browser
	python experiments/trace.py --all
demo-clips:      ## GIF of every recorded scenario in results/demo/ (for slides without a browser)
	python analysis/trace_video.py --all --periods 240
figures:         ## every figure of the paper and reports, from the committed CSVs (no run directories or raw data needed)
	for f in report_figs best_response_fig defences_fig entry_fig relearn_fig identity_fig marklag_fig liquidity_fig \
	         fee_fig queue_fig no_regret_fig public_attacker_fig workshop2_figs frontier_fig \
	         makers_fig stress_fig identity_selection_fig longwindow_fig events_fig retail_fig; do \
	    python analysis/$$f.py || exit 1; done
clean:
	rm -rf results/*_seed* results/_configs
