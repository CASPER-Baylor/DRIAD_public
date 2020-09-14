module load cuda10.1/toolkit/10.1.243 
export CUDA_VISIBLE_DEVICES=`idlegpu`
echo "Start Time: `date`"
echo "Node: `uname -n`"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
./IonWake.exe $1_input/ $1_output/ $1
echo "End Time: `date`"

