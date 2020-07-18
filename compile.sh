
COLOR='\033[0;94m' # red
NC='\033[0m' # no color

##############################################

cd ./build/

if [ $# -ge 1 ]
then
	while [ $# -ge 1 ]
	do
		echo -e "${COLOR}Making: $1 ${NC}"
		make $1
		shift
	done
else
	echo -e "${COLOR}Making: All ${NC}"
	make
fi

cd ../
