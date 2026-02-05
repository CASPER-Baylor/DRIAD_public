
# Get the path of directory to save the DRIAD output
pathRun=$(pwd)

# Get the path of the source directory
pathSource=$(echo $pathRun | sed 's/DRIAD_run_scripts/IonWake/')

# Save the source path in a file
echo $pathSource > source_dir.txt

# Save the run path in a file
echo $pathRun > run_dir.txt