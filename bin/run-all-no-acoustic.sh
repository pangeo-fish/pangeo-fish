#!/bin/bash

rm -rf template

mkdir template
startfile=05a_estimator_scipy.yaml
startfile=06_state_probabilities.yaml
startfile=07_track_decoding.yaml
startfile=04_acoustic_ranges.yaml
startfile=08_verify_emission_state.yaml
startfile=05b_estimator_optuna.yaml
startfile=01_copernicus_diff.yaml

cd template
       ln -sf $startfile 02_healpix_regrid.yaml
       ln -sf $startfile 03_emission.yaml
       ln -sf $startfile 05a_estimator_scipy.yaml
       ln -sf $startfile 06_state_probabilities.yaml
       ln -sf $startfile 07_track_decoding.yaml
       ln -sf $startfile 08_verify_emission_state.yaml
cd ..

#for tagconf in `ls -1 ./notebooks/workflow/configuration/  `;do
tagsshort=' A18831 A18832 A18844 A19051 A19124 A19230'
tagslong='A18828 A18857 A19056 A19226'
tagslong='A18828 A18857 A19056 '
acoustic='A19124 A19051 A18831 A18832 A18828 A18857 A19056 '
tags=$tagslong
for tagconf in $tags;do
echo 'tagconf',$tagconf
       mkdir -p notebooks/workflow/configuration/$tagconf
rm ./notebooks/workflow/configuration/$tagconf/*yaml
rm ./notebooks/workflow/parametrized_no_acoustic/$tagconf/*ipynb
# computation without acoustic
       sed -e "s/A18828/${tagconf}/" -e  "s/\/acoustic\///" template.yaml >notebooks/workflow/configuration/$tagconf/$startfile
   cp -a template/*.yaml notebooks/workflow/configuration/$tagconf/.
#./bin/run-workflow.sh $tagconf  --environment /home1/datawork/todaka/micromamba/envs/pangeo-fish-1023  --html-root /home/datawork-taos-s/public/fish/newbase >$tagconf.log &
   ./bin/run-workflow.sh $tagconf  --environment /home1/datawork/todaka/micromamba/envs/pangeo-fish-0723  --executed-root /home1/datahome/todaka/DATAWORK/git/pangeo-fish/notebooks/workflow/executed_no_acoustic --parametrized-root /home1/datahome/todaka/DATAWORK/git/pangeo-fish/notebooks/workflow/parametrized_no_acoustic  --html-root /home/datawork-taos-s/public/fish/no_acoustic >$tagconf.log &
done
#
