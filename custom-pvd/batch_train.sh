#!/bin/bash
python train_generation.py --category RESIDENTIALhouse --niter 2000
python train_generation.py --category COMMERCIALoffice_building -niter 2000
python train_generation.py --category RELIGIOUSmosque --niter 2000
