#!/bin/sh


tag_id="SV_A11981"

healpy="healpy"

models="merged rang0 armor finis loire gironde adour armor seine  morbihan pdc"
year="2016"


for model in $models
do
notebook="repix_healpix_prepare"
echo 'submitting job with'  'tag_id' $tag_id  'healpy' $healpy  'notebook' $notebook 'model' $model 'year' $year

export notebook healpy tag_id model year
env healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $model$tag_id -V datarmor.pbs
done

exit


models="merged"
notebooks="marc_merge_diff  marc_emission estimator "

after=
for notebook in $notebooks
do
export notebook healpy tag_id model year
sub=`env healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $model$tag_id -V -W depend=afterany:$after datarmor.pbs `
after=`echo $sub |awk '{print $1}'`
echo $after
sleep 1
done
