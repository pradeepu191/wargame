.PHONY: install test smoke replicate refs clean
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
	    --set market.alpha=0.1,0.3,0.5 --set market.n_mm=2,3 --seeds 0-9
clean:
	rm -rf results/*_seed* results/_configs
