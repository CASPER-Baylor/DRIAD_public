module load cuda10.2/toolkit 

# Select one of the GPU configurations(0,1,`idlegpu`)
#export CUDA_VISIBLE_DEVICES=1

echo "Start Time: `date`"
#echo "Node: `uname -n`"
#echo "CUDA Device: $CUDA_VISIBLE_DEVICES"

./IonWake.exe $1_input/ $1_output/ $1

echo "End Time: `date`"