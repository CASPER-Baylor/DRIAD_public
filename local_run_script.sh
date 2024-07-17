# Select one of the GPU configurations(0,1,`idlegpu`)
#export CUDA_VISIBLE_DEVICES=`idlegpu`

echo "Start Time: `date`"
#echo "Node: `uname -n`"
#echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
./IonWake.exe $1_input/ $1_output/ $1 & echo $! > $1_output/$1_jobID.txt 

echo "End Time: `date`"