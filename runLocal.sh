run_dir=$(cat ./run_dir.txt)
cd $run_dir

rm $1_output/* 2> /dev/null

echo "cd $run_dir" > qsub_submit_script.sh
echo "./qsub_run_script.sh $1" >> qsub_submit_script.sh

./IonWake.exe $1_input/ $1_output/ $1  echo $! > $1_output/$1_jobID.txt

#./IonWake.exe $1_input/ $1_output/ $1 & echo $! > $1_output/$1_jobID.txt) > $1_output/qsub_stdout.txt 2> $1_output/qsub_stderr.txt


# Profile with ncu
#ncu --set full --target-processes all --export $1_output/$1_profile.ncu-rep ./IonWake.exe $1_input/ $1_output/ $1 & echo $! > $1_output/$1_jobID.txt
#ncu --set full -k calcIonAccels_102 -c 1 --export $1_output/$1_profile.ncu-rep ./IonWake.exe $1_input/ $1_output/ $1

# Profile with nsys
#nsys profile --stats=true --force-overwrite=true --output=$1_output/$1_profile.nsys-rep ./IonWake.exe $1_input/ $1_output/ $1 & echo $! > $1_output/$1_jobID.txt