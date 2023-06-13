#!/bin/sh


tag_id="SV_A11981"
tag_ids="SV_A11942 SV_A11930"

models="merged"
notebooks="estimator"
notebooks="marc_emission "
notebooks="marc_emission estimator "
notebooks="marc_merge_diff  repix_healpix_comp  marc_emission estimator "
notebooks=" repix_healpix_comp  marc_emission estimator "
notebooks="  marc_emission estimator "
#notebooks="repix_healpix_comp  marc_emission estimator "


after=
for tag_id in $tag_ids
do
model="merged"
year=2016
healpy="healpix"

notebook='marc_merge_diff'
notebook='repix_healpix_comp'
echo healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $tag_id -V datarmor.pbs
sub=`env healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $tag_id -V datarmor.pbs `
after=`echo $sub |awk '{print $1}'`
echo $after
sleep 2



model="merged"
for notebook in $notebooks
do
export notebook healpy tag_id model year
echo healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $tag_id -V -W depend=afterany:$after  datarmor.pbs
sub=`env healpy=$healpy model=$model tag_id=$tag_id notebook=$notebook year=$year qsub -N $tag_id -V -W depend=afterany:$after datarmor.pbs `
after=`echo $sub |awk '{print $1}'`
echo $after
sleep 2
done

done
