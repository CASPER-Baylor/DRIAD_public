run_dir=$(cat ./run_dir.txt)
cd $run_dir

rm $1_output/* 2> /dev/null

echo "cd $run_dir" > qsub_submit_script.sh
echo "./qsub_run_script.sh $1" >> qsub_submit_script.sh

# Run in KODIAK
#qsub -N $1 -q gpu -l nodes=1:ppn=18 -o $1_output/qsub_stdout.txt -e $1_output/qsub_stderr.txt qsub_submit_script.sh 

# Run in SWINT
qsub -N $1 -q swint -l nodes=1:ppn=18 -o $1_output/qsub_stdout.txt -e $1_output/qsub_stderr.txt qsub_submit_script.sh