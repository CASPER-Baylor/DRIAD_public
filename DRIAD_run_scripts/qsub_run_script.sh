#module load cuda/12.3              
#module load cuda12.3/nsight/12.3.2
#ncu --version
#nsys --version

# Select one of the GPU configurations(0,1,`idlegpu`)
#export CUDA_VISIBLE_DEVICES=0

echo "Start Time: `date`"
#echo "Node: `uname -n`"
#echo "CUDA Device: $CUDA_VISIBLE_DEVICES"

./IonWake.exe $1_input/ $1_output/ $1

# Profile with ncu
#ncu --set full --target-processes all --export $1_output/$1_profile.ncu-rep ./IonWake.exe $1_input/ $1_output/ $1 & echo $! > $1_output/$1_jobID.txt
#ncu --set full -k calcIonAccels_102 -c 1 --export $1_output/$1_profile.ncu-rep ./IonWake.exe $1_input/ $1_output/ $1

# Profile with nsys
#nsys profile --stats=true --force-overwrite=true --output=$1_output/$1_profile.nsys-rep ./IonWake.exe $1_input/ $1_output/ $1 & echo $! > $1_output/$1_jobID.txt

echo "End Time: `date`"