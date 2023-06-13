#!/bin/sh


#SV_A11148 | 2016-06-22 18:30:00 | 2016-09-01 22:30:00
#SV_A12595 | 2016-06-23 12:30:00 | 2016-09-01 05:51:00
#SV_A11149 | 2016-06-22 11:20:00 | 2016-08-24 03:30:00
#SV_A11965 | 2016-06-21 17:20:00 | 2016-08-19 06:15:00
#SV_A11930 | 2016-06-22 18:30:00 | 2016-08-08 23:00:00
#SV_A11942 | 2016-06-22 11:20:00 | 2016-07-30 03:09:00
#SV_A11981 | 2016-06-21 17:20:00 | 2016-06-26 11:00:00
tag_id="SV_A11981"
tag_ids="SV_A11148 SV_A11149 SV_A11930 SV_A11942 SV_A11965 SV_A11981 SV_A12595 "
tag_ids="SV_A11942 SV_A11930"
healpy="healpy"

healpy="False"
models="rang0 armor finis loire gironde adour armor seine  morbihan pdc"
notebook="marc_diff"
year="2016"

after=

for tag_id in $tag_ids
do
for model in $models
do
echo 'submitting job with'  'tag_id' $tag_id  'healpy' $healpy  'notebook' $notebook 'model' $model 'year' $year
#echo $intakefilename

export notebook healpy tag_id model year
## todo later here, put chained condition
env healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $model$tag_id -V datarmor.pbs
#sub=`env healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $model$tag_id -V -W depend=afterany:$after toto.pbs `
#af=`echo $sub |awk '{print $1}'`
#echo 'coucou' $after 'af' $af
#after="af"
#echo 'coucou' $after 'af' $af
done
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
