run_dir=$(cat ./run_dir.txt)
cd $run_dir

rm $1_output/* 2> /dev/null

echo "cd $run_dir" > qsub_submit_script.sh
echo "./qsub_run_script.sh $1" >> qsub_submit_script.sh

# Run in local
./local_submit_script.sh & > $1_output/qsub_stdout.txt 2> $1_output/qsub_stderr.txt