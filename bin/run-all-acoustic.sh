#!/bin/bash

rm -rf template

mkdir template
startfile=05a_estimator_scipy.yaml
startfile=06_state_probabilities.yaml
startfile=07_track_decoding.yaml
startfile=04_acoustic_ranges.yaml

cd template
#       ln -sf $startfile 02_healpix_regrid.yaml
#       ln -sf $startfile 03_emission.yaml
#       ln -sf $startfile 04_acoustic_ranges.yaml
       ln -sf $startfile 05a_estimator_scipy.yaml
       ln -sf $startfile 06_state_probabilities.yaml
       ln -sf $startfile 07_track_decoding.yaml
       ln -sf $startfile 08_verify_emission_state.yaml
cd ..

#for tagconf in `ls -1 ./notebooks/workflow/configuration/  `;do
tagsshort=' A18831 A18832 A18844 A19051 A19124 A19230'
tagslong='A18828 A18857 A19056 A19226'
acoustic='A19124 A19051 A18831 A18832 A18828 A18857 A19056 '
tags=$acoustic
for tagconf in $tags;do
echo 'tagconf',$tagconf
       mkdir -p notebooks/workflow/configuration/$tagconf
rm ./notebooks/workflow/configuration/$tagconf/*yaml
rm ./notebooks/workflow/parametrized/$tagconf/*ipynb
# computation with acoustic
       sed -e "s/A18828/${tagconf}/"  template.yaml >notebooks/workflow/configuration/$tagconf/$startfile
   cp -a template/*.yaml notebooks/workflow/configuration/$tagconf/.
#./bin/run-workflow.sh $tagconf  --environment /home1/datawork/todaka/micromamba/envs/pangeo-fish-1023  --html-root /home/datawork-taos-s/public/fish/newbase >$tagconf.log &
   ./bin/run-workflow.sh $tagconf  --environment /home1/datawork/todaka/micromamba/envs/pangeo-fish-0723  --html-root /home/datawork-taos-s/public/fish/acoustic >$tagconf.log &
done
