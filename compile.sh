run_dir=$(cat ./run_dir.txt)
source_dir=$(cat ./source_dir.txt)

cd $source_dir
./compile.sh
cd bin
cp IonWake.exe $run_dir
cd $run_dir

