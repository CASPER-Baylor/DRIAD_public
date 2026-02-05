# Get the ID of the job
jobID=$(cat $1_output/$1_jobID.txt)

#Check if the ID is valid
if [ -n "$jobID" ]; then
    
    # Try to kill the job
    kill -9 "$jobID"
    
    # Check if the job is deleted
    if [ $? -eq 0 ]; then
        #Print a message
        echo "Job is deleted"
    else
        #Print a message
        echo "Error: Job could not be deleted"
    fi    
# The ID is not valid
else
    #Print a message
    echo "It could not be obtained a valid job ID"   
fi         